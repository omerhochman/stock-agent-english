from langchain_core.messages import HumanMessage
import json
import numpy as np
import pandas as pd

from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.tools.api import prices_to_df, get_market_data
from src.utils.logging_config import setup_logger
from src.utils.api_utils import agent_endpoint
from src.calc.portfolio_optimization import portfolio_volatility
from src.tools.factor_data_api import get_market_returns, get_risk_free_rate, get_multiple_index_data
from src.calc.calculate_beta import calculate_beta

logger = setup_logger('valuation_agent')

@agent_endpoint("valuation", "估值分析师，使用DCF和所有者收益法评估公司内在价值")
def valuation_agent(state: AgentState):
    """负责估值分析，使用多种方法评估公司内在价值"""
    show_workflow_status("Valuation Agent")
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    metrics = data["financial_metrics"][0]
    current_financial_line_item = data["financial_line_items"][0]
    previous_financial_line_item = data["financial_line_items"][1]
    market_cap = data["market_cap"]
    ticker = data.get("ticker", "")
    
    # 提取价格数据用于计算Beta
    prices = data.get("prices", [])
    start_date = data.get("start_date")
    end_date = data.get("end_date")

    reasoning = {}

    # 计算营运资本变化
    working_capital_change = (current_financial_line_item.get(
        'working_capital') or 0) - (previous_financial_line_item.get('working_capital') or 0)

    # 1. 获取真实的市场数据
    # 转换价格数据为DataFrame
    prices_df = prices_to_df(prices)
    
    # 2. 获取市场数据，计算实际市场波动率
    market_data = {}
    market_volatility = None  # 初始化为None，以便后续判断是否需要使用默认值
    
    try:
        # 获取主要市场指数数据
        indices = ["000300", "399001", "399006"]  # 沪深300、深证成指、创业板指
        market_indices = get_multiple_index_data(indices, start_date, end_date)
        
        # 创建市场指数的收益率序列
        market_returns_dict = {}
        for idx_name, idx_data in market_indices.items():
            if isinstance(idx_data, pd.DataFrame) and 'close' in idx_data.columns:
                idx_df = idx_data.copy()
                idx_df['return'] = idx_df['close'].pct_change()
                market_returns_dict[idx_name] = idx_df['return'].dropna()
        
        # 使用沪深300作为主要市场基准
        if '000300' in market_returns_dict:
            market_returns = market_returns_dict['000300']
            # 计算实际市场波动率
            market_volatility = market_returns.std() * np.sqrt(252)
            market_data['market_returns'] = market_returns
            market_data['market_volatility'] = market_volatility
            logger.info(f"成功获取市场数据 - 市场波动率: {market_volatility:.2%}")
        else:
            logger.warning("无法获取沪深300指数数据，将尝试使用其他方法获取市场数据")
    except Exception as e:
        logger.error(f"获取市场数据失败: {e}")
    
    # 3. 获取风险指标 (市场回报率和无风险利率)
    try:
        # 获取真实市场回报率和无风险利率
        market_returns_data = get_market_returns(start_date=start_date, end_date=end_date)
        risk_free_data = get_risk_free_rate(start_date=start_date, end_date=end_date)
        
        # 使用实际数据
        market_return = market_returns_data.mean() * 252 if not market_returns_data.empty else 0.08
        risk_free_rate = risk_free_data.mean() if not risk_free_data.empty else 0.03
        
        logger.info(f"获取到真实市场数据: 市场回报率={market_return:.2%}, 无风险利率={risk_free_rate:.2%}")
    except Exception as e:
        # 在无法获取数据的情况下使用合理默认值
        logger.warning(f"无法获取市场回报率和无风险利率数据: {e}，使用默认值")
        market_return = 0.08  # 默认8%市场回报率
        risk_free_rate = 0.03  # 默认3%无风险利率

    # 4. 使用改进的Beta计算函数计算Beta
    beta_value = calculate_beta(ticker, "000300", start_date, end_date)  # 使用calc/calculate_beta.py中的函数
    logger.info(f"使用改进的Beta计算函数计算Beta值: {beta_value:.2f}")

    # 5. 如果市场波动率未成功获取，则使用组合波动率函数计算
    if market_volatility is None:
        if not prices_df.empty:
            try:
                # 使用portfolio_volatility计算股票波动率
                stock_returns = prices_df['close'].pct_change().dropna()
                stock_volatility = portfolio_volatility(np.ones(1), np.array([[stock_returns.var()]])) * np.sqrt(252)
                
                # 使用beta值和股票波动率反推市场波动率
                if beta_value > 0:
                    market_volatility = stock_volatility / beta_value
                else:
                    market_volatility = 0.15  # 默认值
                
                logger.info(f"使用portfolio_volatility和beta反推市场波动率: {market_volatility:.2%}")
            except Exception as e:
                logger.error(f"计算市场波动率失败: {e}")
                market_volatility = 0.15  # 默认值
        else:
            market_volatility = 0.15  # 默认值
            logger.warning("无价格数据，使用默认市场波动率: 15%")

    # 6. 计算加权平均资本成本 (WACC)
    required_return = calculate_required_return(
        ticker=ticker,
        beta=beta_value,
        risk_free_rate=risk_free_rate,
        market_premium=market_return - risk_free_rate
    )

    # 7. 使用所有者收益法估值 (改进的Buffett方法)
    owner_earnings_value = calculate_owner_earnings_value(
        net_income=current_financial_line_item.get('net_income'),
        depreciation=current_financial_line_item.get(
            'depreciation_and_amortization'),
        capex=current_financial_line_item.get('capital_expenditure'),
        working_capital_change=working_capital_change,
        growth_rate=metrics.get("earnings_growth", 0.05),
        required_return=required_return,
        margin_of_safety=0.25
    )

    # 8. 使用DCF估值 (多阶段模型)
    dcf_value = calculate_multistage_dcf(
        free_cash_flow=current_financial_line_item.get('free_cash_flow'),
        revenue_growth=metrics.get("revenue_growth", 0.05),
        earnings_growth=metrics.get("earnings_growth", 0.05),
        net_margin=metrics.get("net_margin", 0.1),
        wacc=required_return
    )
    
    # 9. 获取行业数据进行相对估值分析
    # 首先尝试从市场数据工具获取实际行业平均值
    try:
        # 获取行业数据
        industry_data = get_market_data(ticker)
        
        # 提取行业平均值
        industry_avg_pe = industry_data.get("industry_avg_pe", 15)
        industry_avg_pb = industry_data.get("industry_avg_pb", 1.5)
        industry_growth = industry_data.get("industry_growth", 0.05)
        
        logger.info(f"获取到行业数据: PE={industry_avg_pe}, PB={industry_avg_pb}, 增长率={industry_growth}")
    except Exception as e:
        logger.warning(f"无法获取行业数据: {e}，使用默认值")
        # 使用合理默认值
        industry_avg_pe = 15
        industry_avg_pb = 1.5
        industry_growth = 0.05
    
    # 进行相对估值分析
    pe_ratio = metrics.get("pe_ratio", 0)
    price_to_book = metrics.get("price_to_book", 0)
    
    # 根据公司和行业增长率调整PE
    company_growth = metrics.get("earnings_growth", 0.05)
    growth_premium = max(0, company_growth - industry_growth) * 10  # 每1%额外增长给予10PE溢价
    adjusted_industry_pe = industry_avg_pe + growth_premium
    
    # 计算相对PE比率和PB比率
    pe_relative = pe_ratio / adjusted_industry_pe if adjusted_industry_pe else 1
    pb_relative = price_to_book / industry_avg_pb if industry_avg_pb else 1
    
    # 基于相对估值的内在价值
    earnings_per_share = metrics.get("earnings_per_share", 0)
    relative_value = earnings_per_share * adjusted_industry_pe if earnings_per_share else 0
    
    # 10. 计算剩余收益模型估值 (Residual Income Model)
    book_value_per_share = metrics.get("book_value_per_share", 0)
    roi = metrics.get("return_on_equity", 0.1)
    residual_income_value = 0
    
    if book_value_per_share and roi > 0:
        # 计算超额收益 (ROE - 必要回报率)
        excess_return = max(0, roi - required_return)
        # 简化的剩余收益模型 (超额收益持续5年，然后线性衰减到0)
        residual_income = book_value_per_share * excess_return
        present_value = 0
        for i in range(1, 11):  # 10年期
            year_factor = 1 - (i/10) * 0.5 if i <= 5 else max(0, 1 - 0.5 - (i-5)/5 * 0.5)
            residual_income_year = residual_income * year_factor
            present_value += residual_income_year / ((1 + required_return) ** i)
        
        residual_income_value = book_value_per_share + present_value
    
    # 添加到估值列表
    all_valuations = [
        {"method": "DCF", "value": dcf_value, "weight": 0.35},
        {"method": "Owner Earnings", "value": owner_earnings_value, "weight": 0.35},
        {"method": "Relative Valuation", "value": relative_value, "weight": 0.15},
        {"method": "Residual Income", "value": residual_income_value, "weight": 0.15}
    ]
    
    # 计算加权平均估值
    weighted_sum = sum(v["value"] * v["weight"] for v in all_valuations)
    weight_sum = sum(v["weight"] for v in all_valuations)
    weighted_value = weighted_sum / weight_sum if weight_sum else 0
    
    # 计算估值差距
    dcf_gap = (dcf_value - market_cap) / market_cap if market_cap else 0
    owner_earnings_gap = (owner_earnings_value - market_cap) / market_cap if market_cap else 0
    relative_gap = (relative_value - market_cap) / market_cap if market_cap else 0
    residual_income_gap = (residual_income_value - market_cap) / market_cap if market_cap else 0
    weighted_gap = (weighted_value - market_cap) / market_cap if market_cap else 0

    # 11. 决定投资信号
    if weighted_gap > 0.15:  # 15%以上低估
        signal = 'bullish'
        confidence = min(abs(weighted_gap), 0.50)  # 最大50%置信度
    elif weighted_gap < -0.15:  # 15%以上高估
        signal = 'bearish'
        confidence = min(abs(weighted_gap), 0.50)  # 最大50%置信度
    else:
        signal = 'neutral'
        confidence = 0.25  # 中性信号较低置信度

    # 根据估值模型的一致性调整置信度
    # 计算各模型之间的方差，一致性高则提高置信度
    valuation_signals = []
    for val in all_valuations:
        gap = (val["value"] - market_cap) / market_cap if market_cap else 0
        if gap > 0.15:
            valuation_signals.append(1)  # 看多
        elif gap < -0.15:
            valuation_signals.append(-1)  # 看空
        else:
            valuation_signals.append(0)  # 中性
    
    # 计算信号的方差，方差小表示一致性高
    signal_variance = np.var(valuation_signals) if len(valuation_signals) > 1 else 1
    consistency_boost = max(0, 1 - signal_variance * 0.5)  # 方差为0时提升50%
    confidence = min(confidence * (1 + consistency_boost * 0.5), 1.0)  # 最高提升50%
    
    # 12. 记录推理
    reasoning["dcf_analysis"] = {
        "signal": "bullish" if dcf_gap > 0.15 else "bearish" if dcf_gap < -0.15 else "neutral",
        "details": f"内在价值: ${dcf_value:,.2f}, 市值: ${market_cap:,.2f}, 差异: {dcf_gap:.1%}",
        "model_details": {
            "stages": "多阶段DCF",
            "wacc": f"{required_return:.1%}",
            "beta": f"{beta_value:.2f}",
            "terminal_growth": "3.0%"
        }
    }

    reasoning["owner_earnings_analysis"] = {
        "signal": "bullish" if owner_earnings_gap > 0.15 else "bearish" if owner_earnings_gap < -0.15 else "neutral",
        "details": f"所有者收益价值: ${owner_earnings_value:,.2f}, 市值: ${market_cap:,.2f}, 差异: {owner_earnings_gap:.1%}",
        "model_details": {
            "required_return": f"{required_return:.1%}",
            "margin_of_safety": "25%",
            "growth_rate": f"{metrics.get('earnings_growth', 0.05):.1%}"
        }
    }
    
    reasoning["relative_valuation"] = {
        "signal": "bullish" if relative_gap > 0.15 else "bearish" if relative_gap < -0.15 else "neutral",
        "details": f"相对估值: ${relative_value:,.2f}, 市值: ${market_cap:,.2f}, 差异: {relative_gap:.1%}",
        "model_details": {
            "pe_ratio": f"{pe_ratio:.2f} (行业平均调整: {adjusted_industry_pe:.2f})",
            "pb_ratio": f"{price_to_book:.2f} (行业平均: {industry_avg_pb:.2f})",
            "growth_premium": f"{growth_premium:.1f}"
        }
    }
    
    reasoning["residual_income_valuation"] = {
        "signal": "bullish" if residual_income_gap > 0.15 else "bearish" if residual_income_gap < -0.15 else "neutral",
        "details": f"剩余收益价值: ${residual_income_value:,.2f}, 市值: ${market_cap:,.2f}, 差异: {residual_income_gap:.1%}",
        "model_details": {
            "book_value": f"${book_value_per_share:.2f}",
            "roe": f"{roi:.1%}",
            "excess_return": f"{max(0, roi - required_return):.1%}"
        }
    }
    
    reasoning["weighted_valuation"] = {
        "signal": signal,
        "details": f"加权估值: ${weighted_value:,.2f}, 市值: ${market_cap:,.2f}, 差异: {weighted_gap:.1%}",
        "weights": {v["method"]: f"{v['weight']*100:.0f}%" for v in all_valuations},
        "consistency": f"{consistency_boost:.2f}"
    }

    message_content = {
        "signal": signal,
        "confidence": confidence,
        "valuation_gap": weighted_gap,
        "all_valuations": all_valuations,
        "reasoning": reasoning,
        "capm_data": {
            "beta": beta_value,
            "required_return": required_return,
            "risk_free_rate": risk_free_rate,
            "market_return": market_return,
            "market_volatility": float(market_volatility)
        }
    }

    message = HumanMessage(
        content=json.dumps(message_content),
        name="valuation_agent",
    )

    if show_reasoning:
        show_agent_reasoning(message_content, "Valuation Analysis Agent")
        # 保存推理信息到metadata供API使用
        state["metadata"]["agent_reasoning"] = message_content

    show_workflow_status("Valuation Agent", "completed")
    return {
        "messages": [message],
        "data": {
            **data,
            "valuation_analysis": message_content
        },
        "metadata": state["metadata"],
    }


def calculate_required_return(ticker, beta, risk_free_rate=0.03, market_premium=0.055):
    """
    计算使用CAPM模型的必要收益率/资本成本
    
    Args:
        ticker: 股票代码
        beta: 已计算的beta值
        risk_free_rate: 无风险利率
        market_premium: 市场风险溢价
    
    Returns:
        float: 必要收益率/WACC估计值
    """
    # 使用CAPM公式: WACC = 无风险利率 + Beta * 市场风险溢价
    required_return = risk_free_rate + (beta * market_premium)
    
    # 确保合理范围
    required_return = max(min(required_return, 0.20), 0.05)  # 5%-20%的范围
    
    return required_return


def calculate_owner_earnings_value(
    net_income: float,
    depreciation: float,
    capex: float,
    working_capital_change: float,
    growth_rate: float = 0.05,
    required_return: float = 0.15,
    margin_of_safety: float = 0.25,
    num_years: int = 5
) -> float:
    """
    使用改进的所有者收益法计算公司价值，考虑边际递减增长。

    Args:
        net_income: 净利润
        depreciation: 折旧和摊销
        capex: 资本支出
        working_capital_change: 营运资金变化
        growth_rate: 初始预期增长率
        required_return: 要求回报率
        margin_of_safety: 安全边际比例
        num_years: 预测年数

    Returns:
        float: 基于所有者收益法的估值
    """
    try:
        # 数据有效性检查
        if not all(isinstance(x, (int, float)) for x in [net_income, depreciation, capex, working_capital_change]):
            return 0

        # 计算初始所有者收益
        owner_earnings = (
            net_income +
            depreciation -
            capex -
            working_capital_change
        )

        if owner_earnings <= 0:
            return 0

        # 调整增长率，确保合理性
        growth_rate = min(max(growth_rate, 0), 0.25)  # 限制在0-25%之间

        # 计算预测期收益现值
        future_values = []
        for year in range(1, num_years + 1):
            # 递减增长率模型 - 增长率随时间衰减
            year_growth = growth_rate * (1 - year / (2 * num_years))
            future_value = owner_earnings * (1 + year_growth) ** year
            discounted_value = future_value / (1 + required_return) ** year
            future_values.append(discounted_value)

        # 使用两阶段模型计算终值
        # 第一阶段：前num_years年，上面已计算
        # 第二阶段：永续增长阶段
        terminal_growth = min(growth_rate * 0.4, 0.03)  # 取增长率的40%或3%的较小值
        terminal_value = (
            future_values[-1] * (1 + terminal_growth) / (required_return - terminal_growth))
        terminal_value_discounted = terminal_value / (1 + required_return) ** num_years

        # 计算总价值并应用安全边际
        intrinsic_value = sum(future_values) + terminal_value_discounted
        value_with_safety_margin = intrinsic_value * (1 - margin_of_safety)

        return max(value_with_safety_margin, 0)  # 确保不返回负值

    except Exception as e:
        logger.error(f"所有者收益计算错误: {e}")
        return 0


def calculate_multistage_dcf(
    free_cash_flow: float,
    revenue_growth: float = 0.05,
    earnings_growth: float = 0.05,
    net_margin: float = 0.1,
    wacc: float = 0.10,
    high_growth_years: int = 5,
    transition_years: int = 5,
    terminal_growth_rate: float = 0.02,
) -> float:
    """
    使用三阶段DCF模型计算内在价值
    
    阶段1: 高增长期 - 使用提供的增长率
    阶段2: 过渡期 - 增长率线性降至永续增长率
    阶段3: 永续期 - 使用终值增长率

    Args:
        free_cash_flow: 当前自由现金流
        revenue_growth: 收入增长率
        earnings_growth: 盈利增长率
        net_margin: 净利润率
        wacc: 加权平均资本成本
        high_growth_years: 高增长阶段年数
        transition_years: 过渡期年数
        terminal_growth_rate: 永续期增长率

    Returns:
        float: 三阶段DCF估值
    """
    try:
        if not isinstance(free_cash_flow, (int, float)) or free_cash_flow <= 0:
            return 0

        # 使用净利润率调整的增长率来获得更准确的预测
        # 高增长率和盈利能力通常有正相关性
        adjusted_growth = (earnings_growth + revenue_growth) / 2
        adjusted_growth = min(max(adjusted_growth, 0), 0.25)  # 限制在0-25%范围
        
        # 根据净利润率调整预测的可靠性
        reliability_factor = min(max(net_margin / 0.15, 0.5), 1.5)  # 净利润率与行业平均(15%)的比值
        adjusted_growth = adjusted_growth * reliability_factor
        
        # 第一阶段：高增长期
        high_growth_values = []
        current_fcf = free_cash_flow
        
        for year in range(1, high_growth_years + 1):
            current_fcf = current_fcf * (1 + adjusted_growth)
            present_value = current_fcf / ((1 + wacc) ** year)
            high_growth_values.append(present_value)
            
        # 第二阶段：过渡期
        transition_values = []
        for year in range(1, transition_years + 1):
            # 线性降低增长率
            transition_growth = adjusted_growth - (adjusted_growth - terminal_growth_rate) * (year / transition_years)
            current_fcf = current_fcf * (1 + transition_growth)
            present_value = current_fcf / ((1 + wacc) ** (high_growth_years + year))
            transition_values.append(present_value)
            
        # 第三阶段：永续期
        final_fcf = current_fcf * (1 + terminal_growth_rate)
        terminal_value = final_fcf / (wacc - terminal_growth_rate)
        present_terminal_value = terminal_value / ((1 + wacc) ** (high_growth_years + transition_years))
        
        # 计算总现值
        total_value = sum(high_growth_values) + sum(transition_values) + present_terminal_value
        
        return max(total_value, 0)  # 确保不返回负值

    except Exception as e:
        logger.error(f"DCF计算错误: {e}")
        return 0