"""
Summary Report Module - 生成投资分析汇总报告

此模块负责将Agent分析结果转化为格式化的汇总报告。
"""

import json
import logging
from typing import Dict, Any, List
from datetime import datetime

# 日志设置
logger = logging.getLogger(__name__)

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
    
    # 提取最终决策
    final_decision = state.get("final_decision", {})
    action = final_decision.get("action", "未知")
    quantity = final_decision.get("quantity", 0)
    confidence = final_decision.get("confidence", 0)
    reasoning = final_decision.get("reasoning", "未提供决策理由")
    
    # 提取各Agent的分析结果
    analyses = state.get("analysis", {})
    
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
    action_color = {
        "buy": f"{COLORS['green']}买入{COLORS['reset']}",
        "sell": f"{COLORS['red']}卖出{COLORS['reset']}",
        "hold": f"{COLORS['yellow']}持有{COLORS['reset']}"
    }.get(action.lower(), action)
    
    print(f"{COLORS['bold']}决策:{COLORS['reset']} {action_color}")
    print(f"{COLORS['bold']}数量:{COLORS['reset']} {quantity}")
    print(f"{COLORS['bold']}信心:{COLORS['reset']} {confidence:.1%}" if isinstance(confidence, float) else f"{COLORS['bold']}信心:{COLORS['reset']} {confidence}")
    print(f"{COLORS['bold']}理由:{COLORS['reset']} {reasoning}")
    print("-" * width)
    
    # 打印估值分析结果
    if "valuation" in analyses:
        valuation = analyses.get("valuation", {})
        signal = valuation.get("signal", "未知")
        confidence = valuation.get("confidence", 0)
        reason = valuation.get("reasoning", {})
        
        print(f"\n{COLORS['bold']}{COLORS['bg_blue']} 估值分析 {COLORS['reset']}")
        signal_color = {
            "bullish": f"{COLORS['green']}看涨{COLORS['reset']}",
            "bearish": f"{COLORS['red']}看跌{COLORS['reset']}",
            "neutral": f"{COLORS['yellow']}中性{COLORS['reset']}"
        }.get(signal.lower(), signal)
        
        print(f"{COLORS['bold']}信号:{COLORS['reset']} {signal_color}")
        print(f"{COLORS['bold']}置信度:{COLORS['reset']} {confidence:.1%}" if isinstance(confidence, float) else f"{COLORS['bold']}置信度:{COLORS['reset']} {confidence}")
        
        # 打印DCF分析
        if isinstance(reason, dict) and "dcf_analysis" in reason:
            dcf = reason.get("dcf_analysis", {})
            print(f"{COLORS['bold']}DCF分析:{COLORS['reset']}")
            print(f"  内在价值: {dcf.get('details', '').split(', ')[0].split(': ')[1] if isinstance(dcf.get('details'), str) else '未知'}")
            print(f"  市值: {dcf.get('details', '').split(', ')[1].split(': ')[1] if isinstance(dcf.get('details'), str) else '未知'}")
            print(f"  差距: {dcf.get('details', '').split(', ')[2].split(': ')[1] if isinstance(dcf.get('details'), str) else '未知'}")
        
        print("-" * width)
    
    # 打印基本面分析结果
    if "fundamentals" in analyses:
        fundamentals = analyses.get("fundamentals", {})
        signal = fundamentals.get("signal", "未知")
        confidence = fundamentals.get("confidence", 0)
        reason = fundamentals.get("reasoning", {})
        
        print(f"\n{COLORS['bold']}{COLORS['bg_blue']} 基本面分析 {COLORS['reset']}")
        signal_color = {
            "bullish": f"{COLORS['green']}看涨{COLORS['reset']}",
            "bearish": f"{COLORS['red']}看跌{COLORS['reset']}",
            "neutral": f"{COLORS['yellow']}中性{COLORS['reset']}"
        }.get(signal.lower(), signal)
        
        print(f"{COLORS['bold']}信号:{COLORS['reset']} {signal_color}")
        print(f"{COLORS['bold']}置信度:{COLORS['reset']} {confidence:.1%}" if isinstance(confidence, float) else f"{COLORS['bold']}置信度:{COLORS['reset']} {confidence}")
        
        # 打印盈利能力和增长
        if isinstance(reason, dict):
            if "profitability_signal" in reason:
                prof = reason.get("profitability_signal", {})
                print(f"{COLORS['bold']}盈利能力:{COLORS['reset']} {prof.get('details', '未知')}")
            
            if "growth_signal" in reason:
                growth = reason.get("growth_signal", {})
                print(f"{COLORS['bold']}增长情况:{COLORS['reset']} {growth.get('details', '未知')}")
        
        print("-" * width)
    
    # 打印技术分析结果
    if "technical" in analyses:
        technical = analyses.get("technical", {})
        signal = technical.get("signal", "未知")
        confidence = technical.get("confidence", 0)
        
        print(f"\n{COLORS['bold']}{COLORS['bg_blue']} 技术分析 {COLORS['reset']}")
        signal_color = {
            "bullish": f"{COLORS['green']}看涨{COLORS['reset']}",
            "bearish": f"{COLORS['red']}看跌{COLORS['reset']}",
            "neutral": f"{COLORS['yellow']}中性{COLORS['reset']}"
        }.get(signal.lower(), signal)
        
        print(f"{COLORS['bold']}信号:{COLORS['reset']} {signal_color}")
        print(f"{COLORS['bold']}置信度:{COLORS['reset']} {confidence:.1%}" if isinstance(confidence, float) else f"{COLORS['bold']}置信度:{COLORS['reset']} {confidence}")
        
        # 打印技术指标
        if "strategy_signals" in technical:
            strategies = technical.get("strategy_signals", {})
            if "momentum" in strategies:
                momentum = strategies.get("momentum", {})
                metrics = momentum.get("metrics", {})
                print(f"{COLORS['bold']}动量指标:{COLORS['reset']}")
                for k, v in metrics.items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.2%}")
                    else:
                        print(f"  {k}: {v}")
            
            if "trend_following" in strategies:
                trend = strategies.get("trend_following", {})
                metrics = trend.get("metrics", {})
                print(f"{COLORS['bold']}趋势指标:{COLORS['reset']}")
                for k, v in metrics.items():
                    print(f"  {k}: {v}")
        
        print("-" * width)
    
    # 打印情感分析结果
    if "sentiment" in analyses:
        sentiment = analyses.get("sentiment", {})
        signal = sentiment.get("signal", "未知")
        confidence = sentiment.get("confidence", 0)
        reasoning = sentiment.get("reasoning", "未提供理由")
        
        print(f"\n{COLORS['bold']}{COLORS['bg_blue']} 情感分析 {COLORS['reset']}")
        signal_color = {
            "bullish": f"{COLORS['green']}看涨{COLORS['reset']}",
            "bearish": f"{COLORS['red']}看跌{COLORS['reset']}",
            "neutral": f"{COLORS['yellow']}中性{COLORS['reset']}"
        }.get(signal.lower(), signal)
        
        print(f"{COLORS['bold']}信号:{COLORS['reset']} {signal_color}")
        print(f"{COLORS['bold']}置信度:{COLORS['reset']} {confidence:.1%}" if isinstance(confidence, float) else f"{COLORS['bold']}置信度:{COLORS['reset']} {confidence}")
        print(f"{COLORS['bold']}理由:{COLORS['reset']} {reasoning}")
        print("-" * width)
    
    # 打印辩论室结果
    if "debate_room" in analyses:
        debate = analyses.get("debate_room", {})
        signal = debate.get("signal", "未知")
        confidence = debate.get("confidence", 0)
        bull_conf = debate.get("bull_confidence", 0)
        bear_conf = debate.get("bear_confidence", 0)
        
        print(f"\n{COLORS['bold']}{COLORS['bg_blue']} 辩论室分析 {COLORS['reset']}")
        signal_color = {
            "bullish": f"{COLORS['green']}看涨{COLORS['reset']}",
            "bearish": f"{COLORS['red']}看跌{COLORS['reset']}",
            "neutral": f"{COLORS['yellow']}中性{COLORS['reset']}"
        }.get(signal.lower(), signal)
        
        print(f"{COLORS['bold']}最终观点:{COLORS['reset']} {signal_color}")
        print(f"{COLORS['bold']}置信度:{COLORS['reset']} {confidence:.1%}" if isinstance(confidence, float) else f"{COLORS['bold']}置信度:{COLORS['reset']} {confidence}")
        print(f"{COLORS['bold']}多方置信度:{COLORS['reset']} {bull_conf:.1%}" if isinstance(bull_conf, float) else f"{COLORS['bold']}多方置信度:{COLORS['reset']} {bull_conf}")
        print(f"{COLORS['bold']}空方置信度:{COLORS['reset']} {bear_conf:.1%}" if isinstance(bear_conf, float) else f"{COLORS['bold']}空方置信度:{COLORS['reset']} {bear_conf}")
        
        # 打印辩论摘要
        if "debate_summary" in debate:
            summary = debate.get("debate_summary", [])
            if summary:
                print(f"{COLORS['bold']}辩论摘要:{COLORS['reset']}")
                for point in summary:
                    print(f"  {point}")
        
        # 打印LLM分析
        if "llm_analysis" in debate:
            llm_analysis = debate.get("llm_analysis", "")
            print(f"{COLORS['bold']}AI专家分析:{COLORS['reset']}")
            print(f"  {llm_analysis}")
        
        print("-" * width)
    
    # 打印风险管理结果
    if "risk_manager" in analyses:
        risk = analyses.get("risk_manager", {})
        max_size = risk.get("max_position_size", 0)
        risk_score = risk.get("risk_score", 0)
        risk_metrics = risk.get("risk_metrics", {})
        
        print(f"\n{COLORS['bold']}{COLORS['bg_blue']} 风险管理 {COLORS['reset']}")
        print(f"{COLORS['bold']}风险评分:{COLORS['reset']} {risk_score}/10")
        print(f"{COLORS['bold']}最大仓位大小:{COLORS['reset']} {max_size}")
        
        if risk_metrics:
            print(f"{COLORS['bold']}风险指标:{COLORS['reset']}")
            volatility = risk_metrics.get("volatility", 0)
            var = risk_metrics.get("value_at_risk_95", 0)
            max_dd = risk_metrics.get("max_drawdown", 0)
            
            print(f"  波动率: {volatility:.2%}" if isinstance(volatility, float) else f"  波动率: {volatility}")
            print(f"  风险价值(95%): {var:.2%}" if isinstance(var, float) else f"  风险价值(95%): {var}")
            print(f"  最大回撤: {max_dd:.2%}" if isinstance(max_dd, float) else f"  最大回撤: {max_dd}")
    
    # 打印结尾
    print("\n" + "=" * width)
    print(f"{COLORS['bold']}{COLORS['bg_green']}{'':^15}分析报告结束{'':^15}{COLORS['reset']}")
    print("=" * width + "\n")