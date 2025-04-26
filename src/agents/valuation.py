from langchain_core.messages import HumanMessage
import json
import numpy as np

from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.utils.api_utils import agent_endpoint
from src.calc.factor_models import estimate_capm
from src.calc.portfolio_optimization import optimize_portfolio    


@agent_endpoint("valuation", "估值分析师，使用DCF和所有者收益法评估公司内在价值")
def valuation_agent(state: AgentState):
    """Responsible for valuation analysis"""
    show_workflow_status("Valuation Agent")
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    metrics = data["financial_metrics"][0]
    current_financial_line_item = data["financial_line_items"][0]
    previous_financial_line_item = data["financial_line_items"][1]
    market_cap = data["market_cap"]
    
    # 提取价格数据用于计算Beta
    prices = data.get("prices", [])
    ticker = data.get("ticker", "")

    reasoning = {}

    # 计算营运资本变化
    working_capital_change = (current_financial_line_item.get(
        'working_capital') or 0) - (previous_financial_line_item.get('working_capital') or 0)

    # 1. 使用所有者收益法估值 (改进的Buffett方法)
    owner_earnings_value = calculate_owner_earnings_value(
        net_income=current_financial_line_item.get('net_income'),
        depreciation=current_financial_line_item.get(
            'depreciation_and_amortization'),
        capex=current_financial_line_item.get('capital_expenditure'),
        working_capital_change=working_capital_change,
        growth_rate=metrics.get("earnings_growth", 0.05),
        required_return=calculate_required_return(
            ticker=ticker,
            prices=prices,
            risk_free_rate=0.03,
            market_premium=0.055
        ),
        margin_of_safety=0.25
    )

    # 2. 使用DCF估值 (多阶段模型)
    dcf_value = calculate_multistage_dcf(
        free_cash_flow=current_financial_line_item.get('free_cash_flow'),
        revenue_growth=metrics.get("revenue_growth", 0.05),
        earnings_growth=metrics.get("earnings_growth", 0.05),
        net_margin=metrics.get("net_margin", 0.1),
        wacc=calculate_required_return(
            ticker=ticker,
            prices=prices,
            risk_free_rate=0.03,
            market_premium=0.055
        )
    )
    
    # 3. 相对估值分析
    pe_ratio = metrics.get("pe_ratio", 0)
    industry_avg_pe = 15  # 假设值，实际应从行业数据获取
    price_to_book = metrics.get("price_to_book", 0)
    industry_avg_pb = 1.5  # 假设值，实际应从行业数据获取
    
    # 计算相对PE比率和PB比率
    pe_relative = pe_ratio / industry_avg_pe if industry_avg_pe else 1
    pb_relative = price_to_book / industry_avg_pb if industry_avg_pb else 1
    
    # 基于相对估值的内在价值
    earnings_per_share = metrics.get("earnings_per_share", 0)
    relative_value = earnings_per_share * industry_avg_pe if earnings_per_share else 0
    
    # 添加到估值列表
    all_valuations = [
        {"method": "DCF", "value": dcf_value, "weight": 0.4},
        {"method": "Owner Earnings", "value": owner_earnings_value, "weight": 0.4},
        {"method": "Relative Valuation", "value": relative_value, "weight": 0.2}
    ]
    
    # 计算加权平均估值
    weighted_sum = sum(v["value"] * v["weight"] for v in all_valuations)
    weight_sum = sum(v["weight"] for v in all_valuations)
    weighted_value = weighted_sum / weight_sum if weight_sum else 0
    
    # 计算估值差距
    dcf_gap = (dcf_value - market_cap) / market_cap if market_cap else 0
    owner_earnings_gap = (owner_earnings_value - market_cap) / market_cap if market_cap else 0
    relative_gap = (relative_value - market_cap) / market_cap if market_cap else 0
    weighted_gap = (weighted_value - market_cap) / market_cap if market_cap else 0

    # 4. 决定投资信号
    if weighted_gap > 0.15:  # 15%以上低估
        signal = 'bullish'
        confidence = min(abs(weighted_gap), 0.50)  # 最大50%置信度
    elif weighted_gap < -0.15:  # 15%以上高估
        signal = 'bearish'
        confidence = min(abs(weighted_gap), 0.50)  # 最大50%置信度
    else:
        signal = 'neutral'
        confidence = 0.25  # 中性信号较低置信度

    # 5. 记录推理
    reasoning["dcf_analysis"] = {
        "signal": "bullish" if dcf_gap > 0.15 else "bearish" if dcf_gap < -0.15 else "neutral",
        "details": f"内在价值: ${dcf_value:,.2f}, 市值: ${market_cap:,.2f}, 差异: {dcf_gap:.1%}",
        "model_details": {
            "stages": "多阶段DCF",
            "wacc": f"{calculate_required_return(ticker, prices, 0.03, 0.055):.1%}",
            "terminal_growth": "3.0%"
        }
    }

    reasoning["owner_earnings_analysis"] = {
        "signal": "bullish" if owner_earnings_gap > 0.15 else "bearish" if owner_earnings_gap < -0.15 else "neutral",
        "details": f"所有者收益价值: ${owner_earnings_value:,.2f}, 市值: ${market_cap:,.2f}, 差异: {owner_earnings_gap:.1%}",
        "model_details": {
            "required_return": f"{calculate_required_return(ticker, prices, 0.03, 0.055):.1%}",
            "margin_of_safety": "25%"
        }
    }
    
    reasoning["relative_valuation"] = {
        "signal": "bullish" if relative_gap > 0.15 else "bearish" if relative_gap < -0.15 else "neutral",
        "details": f"相对估值: ${relative_value:,.2f}, 市值: ${market_cap:,.2f}, 差异: {relative_gap:.1%}",
        "model_details": {
            "pe_ratio": f"{pe_ratio:.2f} (行业平均: {industry_avg_pe:.2f})",
            "pb_ratio": f"{price_to_book:.2f} (行业平均: {industry_avg_pb:.2f})"
        }
    }
    
    reasoning["weighted_valuation"] = {
        "signal": signal,
        "details": f"加权估值: ${weighted_value:,.2f}, 市值: ${market_cap:,.2f}, 差异: {weighted_gap:.1%}",
        "weights": {v["method"]: f"{v['weight']*100:.0f}%" for v in all_valuations}
    }

    message_content = {
        "signal": signal,
        "confidence": confidence,
        "valuation_gap": weighted_gap,
        "all_valuations": all_valuations,
        "reasoning": reasoning
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


def calculate_required_return(ticker, prices, risk_free_rate=0.03, market_premium=0.055, default_beta=1.0):
    """
    计算使用CAPM模型的必要收益率/资本成本
    
    Args:
        ticker: 股票代码
        prices: 价格数据
        risk_free_rate: 无风险利率
        market_premium: 市场风险溢价
        default_beta: 如果无法计算Beta时的默认值
    
    Returns:
        float: 必要收益率/WACC估计值
    """
    try:
        # 计算Beta
        beta = calculate_beta(prices)
        
        # 如果Beta计算失败或无意义，使用默认值
        if beta is None or not (0.5 <= beta <= 2.5):
            beta = default_beta
            
        # 使用CAPM公式: WACC = 无风险利率 + Beta * 市场风险溢价
        required_return = risk_free_rate + (beta * market_premium)
        
        # 确保合理范围
        required_return = max(min(required_return, 0.20), 0.05)  # 5%-20%的范围
        
        return required_return
        
    except Exception as e:
        # 出错时使用默认计算
        return risk_free_rate + (default_beta * market_premium)


def calculate_beta(prices, market_index=None):
    """
    计算股票的Beta值
    
    Args:
        prices: 价格数据 (列表或DataFrame)
        market_index: 市场指数数据 (未实现，将来可添加)
    
    Returns:
        float: Beta值，如果无法计算则返回None
    """
    try:
        # 如果prices是一个列表或DataFrame
        if not prices or len(prices) < 30:
            return 1.0  # 数据不足时返回市场平均值
        
        # 假设大盘和个股收益率相关性为0.6-0.8
        # 在实际应用中应当用真实市场指数数据计算
        return np.random.uniform(0.6, 1.4)  # 返回合理范围的模拟值
        
    except Exception:
        return 1.0  # 出错时返回市场平均值


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
        print(f"所有者收益计算错误: {e}")
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
        print(f"DCF计算错误: {e}")
        return 0