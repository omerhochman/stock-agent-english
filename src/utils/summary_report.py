import json
from typing import Dict, Any, Optional
from datetime import datetime

from src.utils.logging_config import setup_logger

# 日志设置
logger = setup_logger(__name__)

# ANSI颜色代码
COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "bg_red": "\033[41m",
    "bg_green": "\033[42m",
    "bg_yellow": "\033[43m",
    "bg_blue": "\033[44m"
}

# Agent名称映射
AGENT_NAMES = {
    "valuation_agent": "valuation",
    "fundamentals_agent": "fundamentals", 
    "sentiment_agent": "sentiment",
    "technical_analyst_agent": "technical",
    "researcher_bull_agent": "researcher_bull",
    "researcher_bear_agent": "researcher_bear",
    "debate_room_agent": "debate_room",
    "risk_management_agent": "risk_manager",
    "macro_analyst_agent": "macro",
    "portfolio_management_agent": "portfolio_manager",
    "ai_model_analyst_agent": "ai_model",
    "portfolio_analyzer_agent": "portfolio_analyzer"
}

def extract_agent_data(state: Dict[str, Any], agent_name: str) -> Optional[Dict[str, Any]]:
    """
    从状态中提取指定agent的数据
    
    Args:
        state: 分析状态字典
        agent_name: 要提取的agent名称
        
    Returns:
        提取的agent数据，如果找不到则返回None
    """
    # 从analysis字典中尝试获取
    analyses = state.get("analysis", {})
    if agent_name in analyses:
        return analyses[agent_name]
    
    # 尝试从messages中提取
    for msg in state.get("messages", []):
        if hasattr(msg, 'name') and msg.name == agent_name:
            try:
                # 尝试解析消息内容
                if hasattr(msg, 'content'):
                    content = msg.content
                    if isinstance(content, str):
                        # 如果是JSON字符串，尝试解析
                        if content.strip().startswith('{') and content.strip().endswith('}'):
                            return json.loads(content)
                        # 如果可能是包含在其他文本中的JSON，尝试提取
                        json_start = content.find('{')
                        json_end = content.rfind('}')
                        if json_start >= 0 and json_end > json_start:
                            json_str = content[json_start:json_end+1]
                            try:
                                return json.loads(json_str)
                            except json.JSONDecodeError:
                                pass
                    elif isinstance(content, dict):
                        return content
            except Exception as e:
                logger.warning(f"解析{agent_name}消息内容时出错: {str(e)}")
    
    # 检查data字典中是否有直接保存的数据
    data = state.get("data", {})
    if f"{agent_name.replace('_agent', '')}_analysis" in data:
        return data[f"{agent_name.replace('_agent', '')}_analysis"]
    elif f"{agent_name}_analysis" in data:
        return data[f"{agent_name}_analysis"]
    
    # 未找到数据
    return None

def safe_get(data: Dict[str, Any], *keys, default=None) -> Any:
    """
    安全地从嵌套字典中获取值
    
    Args:
        data: 要查询的字典
        *keys: 要按顺序查询的键
        default: 找不到时返回的默认值
        
    Returns:
        查询到的值，如果找不到则返回默认值
    """
    if not isinstance(data, dict):
        return default
    
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current

def format_confidence(value, default_text="未知") -> str:
    """
    格式化置信度为百分比
    
    Args:
        value: 置信度值
        default_text: 无法格式化时使用的默认文本
        
    Returns:
        格式化后的置信度字符串
    """
    if isinstance(value, (int, float)):
        # 检查值是否已经是百分比格式(>1)
        if 0 <= value <= 1:
            return f"{value*100:.1f}%"
        elif 1 < value <= 100:
            return f"{value:.1f}%"
    elif isinstance(value, str):
        # 尝试转换字符串为数值
        try:
            if value.endswith('%'):
                return value
            value_float = float(value)
            if 0 <= value_float <= 1:
                return f"{value_float*100:.1f}%"
            elif 1 < value_float <= 100:
                return f"{value_float:.1f}%"
        except ValueError:
            pass
    
    return default_text

def get_signal_color(signal: str) -> str:
    """
    根据信号类型返回对应的颜色
    
    Args:
        signal: 信号类型字符串
        
    Returns:
        带颜色的信号字符串
    """
    if not signal:
        return "未知"
    
    signal_lower = signal.lower()
    
    # 处理不同类型的信号
    if signal_lower in ["bullish", "buy", "positive"]:
        return f"{COLORS['green']}{signal}{COLORS['reset']}"
    elif signal_lower in ["bearish", "sell", "negative"]:
        return f"{COLORS['red']}{signal}{COLORS['reset']}"
    elif signal_lower in ["neutral", "hold"]:
        return f"{COLORS['yellow']}{signal}{COLORS['reset']}"
    
    return signal

def print_summary_report(state: Dict[str, Any]) -> None:
    """
    打印投资分析汇总报告
    
    Args:
        state: 包含所有Agent分析结果的状态字典
    """
    if not state:
        logger.warning("尝试生成报告但状态为空")
        return
    
    ticker = state.get("ticker", "未知")
    date_range = state.get("date_range", {})
    start_date = date_range.get("start", "未知")
    end_date = date_range.get("end", "未知")
    
    # 提取最终决策（从portfolio_management_agent获取）
    portfolio_manager_data = extract_agent_data(state, "portfolio_management_agent")
    final_decision = state.get("final_decision", {})
    
    # 如果final_decision为空，尝试直接从portfolio_manager_data获取
    if not final_decision and portfolio_manager_data:
        final_decision = portfolio_manager_data
    
    action = safe_get(final_decision, "action", default="未知")
    quantity = safe_get(final_decision, "quantity", default=0)
    confidence = safe_get(final_decision, "confidence", default=0)
    reasoning = safe_get(final_decision, "reasoning", default="未提供决策理由")
    
    # 打印报告标题
    width = 100
    print("\n" + "=" * width)
    print(f"{COLORS['bold']}{COLORS['bg_blue']}{'':^10}股票代码 {ticker} 投资分析汇总报告{'':^10}{COLORS['reset']}")
    print("=" * width)
    
    # 打印基本信息
    print(f"{COLORS['bold']}分析区间:{COLORS['reset']} {start_date} 至 {end_date}")
    print(f"{COLORS['bold']}分析日期:{COLORS['reset']} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * width)
    
    # 打印最终决策
    print(f"\n{COLORS['bold']}{COLORS['bg_yellow']} 投资决策 {COLORS['reset']}")
    action_color = get_signal_color(action)
    
    print(f"{COLORS['bold']}决策:{COLORS['reset']} {action_color}")
    print(f"{COLORS['bold']}数量:{COLORS['reset']} {quantity}")
    print(f"{COLORS['bold']}信心:{COLORS['reset']} {format_confidence(confidence)}")
    
    # 处理reasoning可能是列表的情况
    if isinstance(reasoning, list):
        print(f"{COLORS['bold']}理由:{COLORS['reset']}")
        for point in reasoning[:3]:  # 最多显示3点
            print(f"  • {point}")
    else:
        print(f"{COLORS['bold']}理由:{COLORS['reset']} {reasoning[:200]}..." if len(str(reasoning)) > 200 else f"{COLORS['bold']}理由:{COLORS['reset']} {reasoning}")
    
    print("-" * width)
    
    # 打印AI模型分析结果
    ai_model_data = extract_agent_data(state, "ai_model_analyst_agent")
    if ai_model_data:
        signal = safe_get(ai_model_data, "signal", default="未知")
        confidence = safe_get(ai_model_data, "confidence", default=0)
        
        print(f"\n{COLORS['bold']}{COLORS['bg_blue']} AI模型分析 {COLORS['reset']}")
        signal_color = get_signal_color(signal)
        
        print(f"{COLORS['bold']}信号:{COLORS['reset']} {signal_color}")
        print(f"{COLORS['bold']}置信度:{COLORS['reset']} {format_confidence(confidence)}")
        
        # 打印多模型信号（如果有）
        model_signals = safe_get(ai_model_data, "model_signals", default={})
        if model_signals:
            print(f"{COLORS['bold']}模型信号:{COLORS['reset']}")
            for model_name, model_data in model_signals.items():
                model_signal = safe_get(model_data, "signal", default="未知")
                model_conf = safe_get(model_data, "confidence", default=0)
                print(f"  • {model_name}: {get_signal_color(model_signal)} (置信度: {format_confidence(model_conf)})")
        
        # 打印投资组合配置建议（如果有）
        if safe_get(ai_model_data, "multi_asset") and "portfolio_allocation" in ai_model_data:
            allocation = safe_get(ai_model_data, "portfolio_allocation", "allocation", default={})
            if allocation:
                print(f"{COLORS['bold']}资产配置建议:{COLORS['reset']}")
                for asset, weight in allocation.items():
                    print(f"  • {asset}: {weight*100:.1f}%" if isinstance(weight, float) else f"  • {asset}: {weight}")
        
        print("-" * width)
    
    # 打印估值分析结果
    valuation_data = extract_agent_data(state, "valuation_agent")
    if valuation_data:
        signal = safe_get(valuation_data, "signal", default="未知")
        confidence = safe_get(valuation_data, "confidence", default=0)
        valuation_gap = safe_get(valuation_data, "valuation_gap", default=None)
        reasoning = safe_get(valuation_data, "reasoning", default={})
        
        print(f"\n{COLORS['bold']}{COLORS['bg_blue']} 估值分析 {COLORS['reset']}")
        signal_color = get_signal_color(signal)
        
        print(f"{COLORS['bold']}信号:{COLORS['reset']} {signal_color}")
        print(f"{COLORS['bold']}置信度:{COLORS['reset']} {format_confidence(confidence)}")
        
        if valuation_gap is not None:
            print(f"{COLORS['bold']}估值差距:{COLORS['reset']} {format_confidence(valuation_gap)}")
        
        # 打印DCF分析
        dcf_analysis = safe_get(reasoning, "dcf_analysis", default={})
        if dcf_analysis:
            print(f"{COLORS['bold']}DCF分析:{COLORS['reset']}")
            dcf_details = safe_get(dcf_analysis, "details", default="")
            if isinstance(dcf_details, str):
                details_parts = dcf_details.split(', ')
                for part in details_parts:
                    if ': ' in part:
                        key, value = part.split(': ', 1)
                        print(f"  • {key}: {value}")
        
        # 打印所有者收益分析
        owner_earnings = safe_get(reasoning, "owner_earnings_analysis", default={})
        if owner_earnings:
            print(f"{COLORS['bold']}所有者收益分析:{COLORS['reset']}")
            oe_details = safe_get(owner_earnings, "details", default="")
            if isinstance(oe_details, str):
                print(f"  {oe_details}")
        
        print("-" * width)
    
    # 打印基本面分析结果
    fundamentals_data = extract_agent_data(state, "fundamentals_agent")
    if fundamentals_data:
        signal = safe_get(fundamentals_data, "signal", default="未知")
        confidence = safe_get(fundamentals_data, "confidence", default=0)
        reasoning = safe_get(fundamentals_data, "reasoning", default={})
        
        print(f"\n{COLORS['bold']}{COLORS['bg_blue']} 基本面分析 {COLORS['reset']}")
        signal_color = get_signal_color(signal)
        
        print(f"{COLORS['bold']}信号:{COLORS['reset']} {signal_color}")
        print(f"{COLORS['bold']}置信度:{COLORS['reset']} {format_confidence(confidence)}")
        
        # 打印盈利能力和增长
        if isinstance(reasoning, dict):
            for category in ["profitability_signal", "growth_signal", "financial_health_signal", "price_ratios_signal"]:
                category_data = safe_get(reasoning, category, default={})
                if category_data:
                    category_name = {
                        "profitability_signal": "盈利能力",
                        "growth_signal": "增长情况",
                        "financial_health_signal": "财务健康",
                        "price_ratios_signal": "估值比率"
                    }.get(category, category)
                    
                    print(f"{COLORS['bold']}{category_name}:{COLORS['reset']} {safe_get(category_data, 'signal', default='未知')}")
                    details = safe_get(category_data, "details", default="")
                    if details:
                        print(f"  {details}")
        
        print("-" * width)
    
    # 打印技术分析结果
    technical_data = extract_agent_data(state, "technical_analyst_agent")
    if technical_data:
        signal = safe_get(technical_data, "signal", default="未知")
        confidence = safe_get(technical_data, "confidence", default=0)
        market_regime = safe_get(technical_data, "market_regime", default="未知")
        
        print(f"\n{COLORS['bold']}{COLORS['bg_blue']} 技术分析 {COLORS['reset']}")
        signal_color = get_signal_color(signal)
        
        print(f"{COLORS['bold']}信号:{COLORS['reset']} {signal_color}")
        print(f"{COLORS['bold']}置信度:{COLORS['reset']} {format_confidence(confidence)}")
        print(f"{COLORS['bold']}市场状态:{COLORS['reset']} {market_regime}")
        
        # 打印策略信号
        strategy_signals = safe_get(technical_data, "strategy_signals", default={})
        if strategy_signals:
            print(f"{COLORS['bold']}策略信号:{COLORS['reset']}")
            for strategy_name, strategy_data in strategy_signals.items():
                strategy_signal = safe_get(strategy_data, "signal", default="未知")
                strategy_conf = safe_get(strategy_data, "confidence", default=0)
                print(f"  • {strategy_name}: {get_signal_color(strategy_signal)} (置信度: {format_confidence(strategy_conf)})")
        
        print("-" * width)
    
    # 打印情感分析结果
    sentiment_data = extract_agent_data(state, "sentiment_agent")
    if sentiment_data:
        signal = safe_get(sentiment_data, "signal", default="未知")
        confidence = safe_get(sentiment_data, "confidence", default=0)
        reasoning = safe_get(sentiment_data, "reasoning", default="未提供理由")
        
        print(f"\n{COLORS['bold']}{COLORS['bg_blue']} 情感分析 {COLORS['reset']}")
        signal_color = get_signal_color(signal)
        
        print(f"{COLORS['bold']}信号:{COLORS['reset']} {signal_color}")
        print(f"{COLORS['bold']}置信度:{COLORS['reset']} {format_confidence(confidence)}")
        print(f"{COLORS['bold']}理由:{COLORS['reset']} {reasoning}")
        print("-" * width)
    
    # 打印研究员观点
    bull_data = extract_agent_data(state, "researcher_bull_agent")
    bear_data = extract_agent_data(state, "researcher_bear_agent")
    
    if bull_data or bear_data:
        print(f"\n{COLORS['bold']}{COLORS['bg_blue']} 研究员观点 {COLORS['reset']}")
        
        if bull_data:
            bull_conf = safe_get(bull_data, "confidence", default=0)
            bull_points = safe_get(bull_data, "thesis_points", default=[])
            
            print(f"{COLORS['bold']}多方观点 (置信度: {format_confidence(bull_conf)}):{COLORS['reset']}")
            for point in bull_points[:3]:  # 最多显示3点
                print(f"  • {point}")
        
        if bear_data:
            bear_conf = safe_get(bear_data, "confidence", default=0)
            bear_points = safe_get(bear_data, "thesis_points", default=[])
            
            print(f"{COLORS['bold']}空方观点 (置信度: {format_confidence(bear_conf)}):{COLORS['reset']}")
            for point in bear_points[:3]:  # 最多显示3点
                print(f"  • {point}")
        
        print("-" * width)
    
    # 打印辩论室结果
    debate_data = extract_agent_data(state, "debate_room_agent")
    if debate_data:
        signal = safe_get(debate_data, "signal", default="未知")
        confidence = safe_get(debate_data, "confidence", default=0)
        bull_conf = safe_get(debate_data, "bull_confidence", default=0)
        bear_conf = safe_get(debate_data, "bear_confidence", default=0)
        
        print(f"\n{COLORS['bold']}{COLORS['bg_blue']} 辩论室分析 {COLORS['reset']}")
        signal_color = get_signal_color(signal)
        
        print(f"{COLORS['bold']}最终观点:{COLORS['reset']} {signal_color}")
        print(f"{COLORS['bold']}置信度:{COLORS['reset']} {format_confidence(confidence)}")
        print(f"{COLORS['bold']}多方置信度:{COLORS['reset']} {format_confidence(bull_conf)}")
        print(f"{COLORS['bold']}空方置信度:{COLORS['reset']} {format_confidence(bear_conf)}")
        
        # 打印AI模型贡献（如果有）
        ai_contribution = safe_get(debate_data, "ai_model_contribution", default=None)
        if ai_contribution and safe_get(ai_contribution, "included"):
            ai_signal = safe_get(ai_contribution, "signal", default="未知")
            ai_confidence = safe_get(ai_contribution, "confidence", default=0)
            ai_weight = safe_get(ai_contribution, "weight", default=0)
            
            print(f"{COLORS['bold']}AI模型贡献:{COLORS['reset']} {get_signal_color(ai_signal)} (置信度: {format_confidence(ai_confidence)}, 权重: {ai_weight*100:.0f}%)")
        
        # 打印辩论摘要
        debate_summary = safe_get(debate_data, "debate_summary", default=[])
        if debate_summary:
            print(f"{COLORS['bold']}辩论摘要:{COLORS['reset']}")
            for i, point in enumerate(debate_summary[:6]):  # 最多显示6点
                print(f"  {point}")
        
        # 打印LLM分析
        llm_analysis = safe_get(debate_data, "llm_analysis", default="")
        if llm_analysis:
            llm_summary = llm_analysis[:150] + "..." if len(str(llm_analysis)) > 150 else llm_analysis
            print(f"{COLORS['bold']}AI专家分析:{COLORS['reset']}")
            print(f"  {llm_summary}")
        
        print("-" * width)
    
    # 打印风险管理结果
    risk_data = extract_agent_data(state, "risk_management_agent")
    if risk_data:
        max_size = safe_get(risk_data, "max_position_size", default=0)
        risk_score = safe_get(risk_data, "risk_score", default=0)
        trading_action = safe_get(risk_data, "trading_action", default="未知")
        risk_metrics = safe_get(risk_data, "risk_metrics", default={})
        
        print(f"\n{COLORS['bold']}{COLORS['bg_blue']} 风险管理 {COLORS['reset']}")
        print(f"{COLORS['bold']}风险评分:{COLORS['reset']} {risk_score}/10")
        print(f"{COLORS['bold']}交易建议:{COLORS['reset']} {get_signal_color(trading_action)}")
        print(f"{COLORS['bold']}最大仓位大小:{COLORS['reset']} {max_size}")
        
        if risk_metrics:
            print(f"{COLORS['bold']}风险指标:{COLORS['reset']}")
            volatility = safe_get(risk_metrics, "volatility", default=0)
            var = safe_get(risk_metrics, "value_at_risk_95", default=0)
            cvar = safe_get(risk_metrics, "conditional_var_95", default=0)
            max_dd = safe_get(risk_metrics, "max_drawdown", default=0)
            
            print(f"  • 波动率: {format_confidence(volatility)}")
            print(f"  • 风险价值(95%): {format_confidence(var)}")
            print(f"  • 条件风险价值: {format_confidence(cvar)}")
            print(f"  • 最大回撤: {format_confidence(max_dd)}")
        
        # 打印GARCH模型结果
        garch_model = safe_get(risk_data, "volatility_model", default={})
        if garch_model and "forecast" in garch_model:
            print(f"{COLORS['bold']}波动率预测:{COLORS['reset']} 未来波动率趋势{'上升' if safe_get(garch_model, 'forecast_annualized', 0) > safe_get(risk_metrics, 'volatility', 0) else '下降'}")
        
        print("-" * width)
    
    # 打印宏观分析结果
    macro_data = extract_agent_data(state, "macro_analyst_agent")
    if macro_data:
        signal = safe_get(macro_data, "signal", default="未知")
        confidence = safe_get(macro_data, "confidence", default=0)
        macro_env = safe_get(macro_data, "macro_environment", default="未知")
        impact = safe_get(macro_data, "impact_on_stock", default="未知")
        key_factors = safe_get(macro_data, "key_factors", default=[])
        reasoning = safe_get(macro_data, "reasoning", default="未提供理由")
        
        print(f"\n{COLORS['bold']}{COLORS['bg_blue']} 宏观分析 {COLORS['reset']}")
        signal_color = get_signal_color(signal)
        
        print(f"{COLORS['bold']}信号:{COLORS['reset']} {signal_color}")
        print(f"{COLORS['bold']}置信度:{COLORS['reset']} {format_confidence(confidence)}")
        
        env_color = get_signal_color(macro_env)
        impact_color = get_signal_color(impact)
        
        print(f"{COLORS['bold']}宏观环境:{COLORS['reset']} {env_color}")
        print(f"{COLORS['bold']}对股票影响:{COLORS['reset']} {impact_color}")
        
        if key_factors:
            print(f"{COLORS['bold']}关键因素:{COLORS['reset']}")
            for i, factor in enumerate(key_factors[:5]):  # 最多显示5个因素
                print(f"  {i+1}. {factor}")
        
        # 提取多资产分析结果（如果有）
        multi_asset = safe_get(macro_data, "multi_asset_analysis", default=None)
        if multi_asset:
            tickers = safe_get(multi_asset, "tickers", default=[])
            if tickers:
                print(f"{COLORS['bold']}多资产分析 ({len(tickers)} 只股票):{COLORS['reset']}")
                # 显示风险分散化提示
                diversification_tips = safe_get(multi_asset, "diversification_tips", default=[])
                for tip in diversification_tips[:2]:  # 最多显示2条提示
                    print(f"  • {tip}")
        
        print("-" * width)
    
    # 打印投资组合分析结果
    portfolio_analysis_data = extract_agent_data(state, "portfolio_analyzer_agent")
    if portfolio_analysis_data:
        tickers = safe_get(portfolio_analysis_data, "tickers", default=[])
        portfolio_analysis = safe_get(portfolio_analysis_data, "portfolio_analysis", default={})
        risk_analysis = safe_get(portfolio_analysis_data, "risk_analysis", default={})
        efficient_frontier = safe_get(portfolio_analysis_data, "efficient_frontier", default={})
        
        print(f"\n{COLORS['bold']}{COLORS['bg_blue']} 投资组合分析 ({len(tickers)} 只股票) {COLORS['reset']}")
        
        # 打印最优投资组合
        if "max_sharpe" in portfolio_analysis:
            max_sharpe = safe_get(portfolio_analysis, "max_sharpe", default={})
            sharpe = safe_get(max_sharpe, "sharpe_ratio", default=0)
            returns = safe_get(max_sharpe, "return", default=0)
            risk = safe_get(max_sharpe, "risk", default=0)
            
            print(f"{COLORS['bold']}最优投资组合 (最大夏普比率):{COLORS['reset']}")
            print(f"  • 夏普比率: {sharpe:.2f}")
            print(f"  • 预期收益率: {format_confidence(returns)}")
            print(f"  • 风险: {format_confidence(risk)}")
            
            # 打印权重
            weights = safe_get(max_sharpe, "weights", default={})
            if weights:
                print(f"  • 权重分配:")
                for ticker, weight in weights.items():
                    if weight > 0.05:  # 只显示权重超过5%的资产
                        print(f"    - {ticker}: {format_confidence(weight)}")
        
        # 打印风险分析
        if risk_analysis:
            var_95 = safe_get(risk_analysis, "var_95", default=0)
            max_dd = safe_get(risk_analysis, "max_drawdown", default=0)
            
            print(f"{COLORS['bold']}风险分析:{COLORS['reset']}")
            print(f"  • 风险价值(95%): {format_confidence(var_95)}")
            print(f"  • 最大回撤: {format_confidence(max_dd)}")
            
            # 显示最佳和最差资产
            best_asset = safe_get(risk_analysis, "best_asset", default={})
            worst_asset = safe_get(risk_analysis, "worst_asset", default={})
            
            if best_asset and worst_asset:
                print(f"  • 最佳资产: {safe_get(best_asset, 'ticker', default='未知')} (收益率: {format_confidence(safe_get(best_asset, 'return', default=0))})")
                print(f"  • 最差资产: {safe_get(worst_asset, 'ticker', default='未知')} (收益率: {format_confidence(safe_get(worst_asset, 'return', default=0))})")
        
        print("-" * width)
    
    # 打印结尾
    print("\n" + "=" * width)
    print(f"{COLORS['bold']}{COLORS['bg_green']}{'':^15}分析报告结束{'':^15}{COLORS['reset']}")
    print("=" * width + "\n")

def print_compact_summary(state: Dict[str, Any]) -> None:
    """
    打印简化版的投资分析汇总报告
    
    Args:
        state: 包含所有Agent分析结果的状态字典
    """
    if not state:
        logger.warning("尝试生成报告但状态为空")
        return
    
    ticker = state.get("ticker", "未知")
    
    # 提取最终决策
    portfolio_manager_data = extract_agent_data(state, "portfolio_management_agent")
    final_decision = state.get("final_decision", {})
    
    # 如果final_decision为空，尝试直接从portfolio_manager_data获取
    if not final_decision and portfolio_manager_data:
        final_decision = portfolio_manager_data
    
    action = safe_get(final_decision, "action", default="未知")
    quantity = safe_get(final_decision, "quantity", default=0)
    
    # 打印简化报告
    width = 80
    print("\n" + "=" * width)
    print(f"{COLORS['bold']}股票 {ticker} 投资决策简报{COLORS['reset']}")
    print("-" * width)
    
    # 打印决策摘要
    action_color = get_signal_color(action)
    print(f"{COLORS['bold']}决策:{COLORS['reset']} {action_color} {quantity} 股")
    
    # 收集所有Agent的信号
    signals_summary = []
    
    # 定义要检查的agent及其显示名称
    agents_to_check = [
        ("valuation_agent", "估值分析"),
        ("fundamentals_agent", "基本面"),
        ("technical_analyst_agent", "技术分析"),
        ("sentiment_agent", "情感分析"),
        ("debate_room_agent", "辩论结论"),
        ("ai_model_analyst_agent", "AI模型")
    ]
    
    # 提取每个agent的信号
    for agent_name, display_name in agents_to_check:
        agent_data = extract_agent_data(state, agent_name)
        if agent_data:
            signal = safe_get(agent_data, "signal", default="未知")
            confidence = safe_get(agent_data, "confidence", default=0)
            signals_summary.append(f"{display_name}: {get_signal_color(signal)} ({format_confidence(confidence)})")
    
    # 打印信号摘要
    print(f"{COLORS['bold']}分析师意见:{COLORS['reset']}")
    for i, signal in enumerate(signals_summary):
        print(f"  • {signal}")
    
    # 提取风险信息
    risk_data = extract_agent_data(state, "risk_management_agent")
    if risk_data:
        risk_score = safe_get(risk_data, "risk_score", default=0)
        print(f"{COLORS['bold']}风险评分:{COLORS['reset']} {risk_score}/10")
    
    print("=" * width + "\n")

def format_agent_data_for_web(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    格式化分析状态为Web友好格式
    
    Args:
        state: 包含所有Agent分析结果的状态字典
        
    Returns:
        用于Web展示的格式化数据字典
    """
    if not state:
        return {"error": "状态为空"}
    
    result = {
        "ticker": state.get("ticker", "未知"),
        "date_range": state.get("date_range", {}),
        "timestamp": datetime.now().isoformat(),
        "agents": {},
        "decision": {},
    }
    
    # 提取最终决策
    portfolio_manager_data = extract_agent_data(state, "portfolio_management_agent")
    final_decision = state.get("final_decision", {})
    
    # 如果final_decision为空，尝试直接从portfolio_manager_data获取
    if not final_decision and portfolio_manager_data:
        final_decision = portfolio_manager_data
    
    # 格式化最终决策
    result["decision"] = {
        "action": safe_get(final_decision, "action", default="未知"),
        "quantity": safe_get(final_decision, "quantity", default=0),
        "confidence": safe_get(final_decision, "confidence", default=0),
        "reasoning": safe_get(final_decision, "reasoning", default="未提供决策理由")
    }
    
    # 处理所有Agent的数据
    for agent_name, display_name in AGENT_NAMES.items():
        agent_data = extract_agent_data(state, agent_name)
        if agent_data:
            # 基本信息
            agent_info = {
                "name": display_name,
                "signal": safe_get(agent_data, "signal", default="未知"),
                "confidence": safe_get(agent_data, "confidence", default=0),
            }
            
            # 添加特定Agent的详细信息
            if agent_name == "valuation_agent":
                agent_info["valuation_gap"] = safe_get(agent_data, "valuation_gap", default=None)
                agent_info["capm_data"] = safe_get(agent_data, "capm_data", default={})
                
            elif agent_name == "risk_management_agent":
                agent_info["risk_score"] = safe_get(agent_data, "risk_score", default=0)
                agent_info["max_position_size"] = safe_get(agent_data, "max_position_size", default=0)
                agent_info["trading_action"] = safe_get(agent_data, "trading_action", default="未知")
                agent_info["risk_metrics"] = safe_get(agent_data, "risk_metrics", default={})
                
            elif agent_name == "ai_model_analyst_agent":
                agent_info["model_signals"] = safe_get(agent_data, "model_signals", default={})
                agent_info["multi_asset"] = safe_get(agent_data, "multi_asset", default=False)
                if agent_info["multi_asset"]:
                    agent_info["portfolio_allocation"] = safe_get(agent_data, "portfolio_allocation", default={})
                    agent_info["asset_signals"] = safe_get(agent_data, "asset_signals", default={})
            
            elif agent_name == "portfolio_analyzer_agent":
                agent_info["portfolio_analysis"] = safe_get(agent_data, "portfolio_analysis", default={})
                agent_info["risk_analysis"] = safe_get(agent_data, "risk_analysis", default={})
                
            # 将Agent数据添加到结果字典
            result["agents"][display_name] = agent_info
    
    return result