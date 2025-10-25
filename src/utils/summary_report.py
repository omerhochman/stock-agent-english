import json
from typing import Dict, Any, Optional
from datetime import datetime

from src.utils.logging_config import setup_logger

# Logging setup
logger = setup_logger(__name__)

# ANSI color codes
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

# Agent name mapping
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
    Extract data for specified agent from state
    
    Args:
        state: Analysis state dictionary
        agent_name: Name of agent to extract
        
    Returns:
        Extracted agent data, returns None if not found
    """
    # Try to get from analysis dictionary
    analyses = state.get("analysis", {})
    if agent_name in analyses:
        return analyses[agent_name]
    
    # Try to extract from messages
    for msg in state.get("messages", []):
        if hasattr(msg, 'name') and msg.name == agent_name:
            try:
                # Try to parse message content
                if hasattr(msg, 'content'):
                    content = msg.content
                    if isinstance(content, str):
                        # If it's a JSON string, try to parse
                        if content.strip().startswith('{') and content.strip().endswith('}'):
                            return json.loads(content)
                        # If it might be JSON embedded in other text, try to extract
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
                logger.warning(f"Error parsing {agent_name} message content: {str(e)}")
    
    # Check if there's directly saved data in data dictionary
    data = state.get("data", {})
    if f"{agent_name.replace('_agent', '')}_analysis" in data:
        return data[f"{agent_name.replace('_agent', '')}_analysis"]
    elif f"{agent_name}_analysis" in data:
        return data[f"{agent_name}_analysis"]
    
    # Data not found
    return None

def safe_get(data: Dict[str, Any], *keys, default=None) -> Any:
    """
    Safely get value from nested dictionary
    
    Args:
        data: Dictionary to query
        *keys: Keys to query in order
        default: Default value to return if not found
        
    Returns:
        Queried value, returns default value if not found
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

def format_confidence(value, default_text="Unknown") -> str:
    """
    Format confidence as percentage
    
    Args:
        value: Confidence value
        default_text: Default text to use when formatting fails
        
    Returns:
        Formatted confidence string
    """
    if isinstance(value, (int, float)):
        # Check if value is already in percentage format (>1)
        if 0 <= value <= 1:
            return f"{value*100:.1f}%"
        elif 1 < value <= 100:
            return f"{value:.1f}%"
    elif isinstance(value, str):
        # Try to convert string to numeric value
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
    Return corresponding color based on signal type
    
    Args:
        signal: Signal type string
        
    Returns:
        Colored signal string
    """
    if not signal:
        return "Unknown"
    
    signal_lower = signal.lower()
    
    # Handle different types of signals
    if signal_lower in ["bullish", "buy", "positive"]:
        return f"{COLORS['green']}{signal}{COLORS['reset']}"
    elif signal_lower in ["bearish", "sell", "negative"]:
        return f"{COLORS['red']}{signal}{COLORS['reset']}"
    elif signal_lower in ["neutral", "hold"]:
        return f"{COLORS['yellow']}{signal}{COLORS['reset']}"
    
    return signal

def print_summary_report(state: Dict[str, Any]) -> None:
    """
    Print investment analysis summary report
    
    Args:
        state: State dictionary containing all Agent analysis results
    """
    if not state:
        logger.warning("Attempting to generate report but state is empty")
        return
    
    ticker = state.get("ticker", "Unknown")
    date_range = state.get("date_range", {})
    start_date = date_range.get("start", "Unknown")
    end_date = date_range.get("end", "Unknown")
    
    # Extract final decision (from portfolio_management_agent)
    portfolio_manager_data = extract_agent_data(state, "portfolio_management_agent")
    final_decision = state.get("final_decision", {})
    
    # If final_decision is empty, try to get directly from portfolio_manager_data
    if not final_decision and portfolio_manager_data:
        final_decision = portfolio_manager_data
    
    action = safe_get(final_decision, "action", default="Unknown")
    quantity = safe_get(final_decision, "quantity", default=0)
    confidence = safe_get(final_decision, "confidence", default=0)
    reasoning = safe_get(final_decision, "reasoning", default="No decision reasoning provided")
    
    # Print report title
    width = 100
    print("\n" + "=" * width)
    print(f"{COLORS['bold']}{COLORS['bg_blue']}{'':^10}Stock Code {ticker} Investment Analysis Summary Report{'':^10}{COLORS['reset']}")
    print("=" * width)
    
    # Print basic information
    print(f"{COLORS['bold']}Analysis Period:{COLORS['reset']} {start_date} to {end_date}")
    print(f"{COLORS['bold']}Analysis Date:{COLORS['reset']} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * width)
    
    # Print final decision
    print(f"\n{COLORS['bold']}{COLORS['bg_yellow']} Investment Decision {COLORS['reset']}")
    action_color = get_signal_color(action)
    
    print(f"{COLORS['bold']}Decision:{COLORS['reset']} {action_color}")
    print(f"{COLORS['bold']}Quantity:{COLORS['reset']} {quantity}")
    print(f"{COLORS['bold']}Confidence:{COLORS['reset']} {format_confidence(confidence)}")
    
    # Handle case where reasoning might be a list
    if isinstance(reasoning, list):
        print(f"{COLORS['bold']}Reasoning:{COLORS['reset']}")
        for point in reasoning[:3]:  # Show at most 3 points
            print(f"  • {point}")
    else:
        print(f"{COLORS['bold']}Reasoning:{COLORS['reset']} {reasoning[:200]}..." if len(str(reasoning)) > 200 else f"{COLORS['bold']}Reasoning:{COLORS['reset']} {reasoning}")
    
    print("-" * width)
    
    # Print AI model analysis results
    ai_model_data = extract_agent_data(state, "ai_model_analyst_agent")
    if ai_model_data:
        signal = safe_get(ai_model_data, "signal", default="Unknown")
        confidence = safe_get(ai_model_data, "confidence", default=0)
        
        print(f"\n{COLORS['bold']}{COLORS['bg_blue']} AI Model Analysis {COLORS['reset']}")
        signal_color = get_signal_color(signal)
        
        print(f"{COLORS['bold']}Signal:{COLORS['reset']} {signal_color}")
        print(f"{COLORS['bold']}Confidence:{COLORS['reset']} {format_confidence(confidence)}")
        
        # Print multi-model signals (if any)
        model_signals = safe_get(ai_model_data, "model_signals", default={})
        if model_signals:
            print(f"{COLORS['bold']}Model Signals:{COLORS['reset']}")
            for model_name, model_data in model_signals.items():
                model_signal = safe_get(model_data, "signal", default="Unknown")
                model_conf = safe_get(model_data, "confidence", default=0)
                print(f"  • {model_name}: {get_signal_color(model_signal)} (Confidence: {format_confidence(model_conf)})")
        
        # Print portfolio allocation recommendations (if any)
        if safe_get(ai_model_data, "multi_asset") and "portfolio_allocation" in ai_model_data:
            allocation = safe_get(ai_model_data, "portfolio_allocation", "allocation", default={})
            if allocation:
                print(f"{COLORS['bold']}Asset Allocation Recommendations:{COLORS['reset']}")
                for asset, weight in allocation.items():
                    print(f"  • {asset}: {weight*100:.1f}%" if isinstance(weight, float) else f"  • {asset}: {weight}")
        
        print("-" * width)
    
    # Print valuation analysis results
    valuation_data = extract_agent_data(state, "valuation_agent")
    if valuation_data:
        signal = safe_get(valuation_data, "signal", default="Unknown")
        confidence = safe_get(valuation_data, "confidence", default=0)
        valuation_gap = safe_get(valuation_data, "valuation_gap", default=None)
        reasoning = safe_get(valuation_data, "reasoning", default={})
        
        print(f"\n{COLORS['bold']}{COLORS['bg_blue']} Valuation Analysis {COLORS['reset']}")
        signal_color = get_signal_color(signal)
        
        print(f"{COLORS['bold']}Signal:{COLORS['reset']} {signal_color}")
        print(f"{COLORS['bold']}Confidence:{COLORS['reset']} {format_confidence(confidence)}")
        
        if valuation_gap is not None:
            print(f"{COLORS['bold']}Valuation Gap:{COLORS['reset']} {format_confidence(valuation_gap)}")
        
        # Print DCF analysis
        dcf_analysis = safe_get(reasoning, "dcf_analysis", default={})
        if dcf_analysis:
            print(f"{COLORS['bold']}DCF Analysis:{COLORS['reset']}")
            dcf_details = safe_get(dcf_analysis, "details", default="")
            if isinstance(dcf_details, str):
                details_parts = dcf_details.split(', ')
                for part in details_parts:
                    if ': ' in part:
                        key, value = part.split(': ', 1)
                        print(f"  • {key}: {value}")
        
        # Print owner earnings analysis
        owner_earnings = safe_get(reasoning, "owner_earnings_analysis", default={})
        if owner_earnings:
            print(f"{COLORS['bold']}Owner Earnings Analysis:{COLORS['reset']}")
            oe_details = safe_get(owner_earnings, "details", default="")
            if isinstance(oe_details, str):
                print(f"  {oe_details}")
        
        print("-" * width)
    
    # Print fundamental analysis results
    fundamentals_data = extract_agent_data(state, "fundamentals_agent")
    if fundamentals_data:
        signal = safe_get(fundamentals_data, "signal", default="Unknown")
        confidence = safe_get(fundamentals_data, "confidence", default=0)
        reasoning = safe_get(fundamentals_data, "reasoning", default={})
        
        print(f"\n{COLORS['bold']}{COLORS['bg_blue']} Fundamental Analysis {COLORS['reset']}")
        signal_color = get_signal_color(signal)
        
        print(f"{COLORS['bold']}Signal:{COLORS['reset']} {signal_color}")
        print(f"{COLORS['bold']}Confidence:{COLORS['reset']} {format_confidence(confidence)}")
        
        # Print profitability and growth
        if isinstance(reasoning, dict):
            for category in ["profitability_signal", "growth_signal", "financial_health_signal", "price_ratios_signal"]:
                category_data = safe_get(reasoning, category, default={})
                if category_data:
                    category_name = {
                        "profitability_signal": "Profitability",
                        "growth_signal": "Growth",
                        "financial_health_signal": "Financial Health",
                        "price_ratios_signal": "Valuation Ratios"
                    }.get(category, category)
                    
                    print(f"{COLORS['bold']}{category_name}:{COLORS['reset']} {safe_get(category_data, 'signal', default='Unknown')}")
                    details = safe_get(category_data, "details", default="")
                    if details:
                        print(f"  {details}")
        
        print("-" * width)
    
    # Print technical analysis results
    technical_data = extract_agent_data(state, "technical_analyst_agent")
    if technical_data:
        signal = safe_get(technical_data, "signal", default="Unknown")
        confidence = safe_get(technical_data, "confidence", default=0)
        market_regime = safe_get(technical_data, "market_regime", default="Unknown")
        
        print(f"\n{COLORS['bold']}{COLORS['bg_blue']} Technical Analysis {COLORS['reset']}")
        signal_color = get_signal_color(signal)
        
        print(f"{COLORS['bold']}Signal:{COLORS['reset']} {signal_color}")
        print(f"{COLORS['bold']}Confidence:{COLORS['reset']} {format_confidence(confidence)}")
        print(f"{COLORS['bold']}Market Regime:{COLORS['reset']} {market_regime}")
        
        # Print strategy signals
        strategy_signals = safe_get(technical_data, "strategy_signals", default={})
        if strategy_signals:
            print(f"{COLORS['bold']}Strategy Signals:{COLORS['reset']}")
            for strategy_name, strategy_data in strategy_signals.items():
                strategy_signal = safe_get(strategy_data, "signal", default="Unknown")
                strategy_conf = safe_get(strategy_data, "confidence", default=0)
                print(f"  • {strategy_name}: {get_signal_color(strategy_signal)} (Confidence: {format_confidence(strategy_conf)})")
        
        print("-" * width)
    
    # Print sentiment analysis results
    sentiment_data = extract_agent_data(state, "sentiment_agent")
    if sentiment_data:
        signal = safe_get(sentiment_data, "signal", default="Unknown")
        confidence = safe_get(sentiment_data, "confidence", default=0)
        reasoning = safe_get(sentiment_data, "reasoning", default="No reasoning provided")
        
        print(f"\n{COLORS['bold']}{COLORS['bg_blue']} Sentiment Analysis {COLORS['reset']}")
        signal_color = get_signal_color(signal)
        
        print(f"{COLORS['bold']}Signal:{COLORS['reset']} {signal_color}")
        print(f"{COLORS['bold']}Confidence:{COLORS['reset']} {format_confidence(confidence)}")
        print(f"{COLORS['bold']}Reasoning:{COLORS['reset']} {reasoning}")
        print("-" * width)
    
    # Print researcher perspectives
    bull_data = extract_agent_data(state, "researcher_bull_agent")
    bear_data = extract_agent_data(state, "researcher_bear_agent")
    
    if bull_data or bear_data:
        print(f"\n{COLORS['bold']}{COLORS['bg_blue']} Researcher Perspectives {COLORS['reset']}")
        
        if bull_data:
            bull_conf = safe_get(bull_data, "confidence", default=0)
            bull_points = safe_get(bull_data, "thesis_points", default=[])
            
            print(f"{COLORS['bold']}Bullish Perspective (Confidence: {format_confidence(bull_conf)}):{COLORS['reset']}")
            for point in bull_points[:3]:  # Show at most 3 points
                print(f"  • {point}")
        
        if bear_data:
            bear_conf = safe_get(bear_data, "confidence", default=0)
            bear_points = safe_get(bear_data, "thesis_points", default=[])
            
            print(f"{COLORS['bold']}Bearish Perspective (Confidence: {format_confidence(bear_conf)}):{COLORS['reset']}")
            for point in bear_points[:3]:  # Show at most 3 points
                print(f"  • {point}")
        
        print("-" * width)
    
    # Print debate room results
    debate_data = extract_agent_data(state, "debate_room_agent")
    if debate_data:
        signal = safe_get(debate_data, "signal", default="Unknown")
        confidence = safe_get(debate_data, "confidence", default=0)
        bull_conf = safe_get(debate_data, "bull_confidence", default=0)
        bear_conf = safe_get(debate_data, "bear_confidence", default=0)
        
        print(f"\n{COLORS['bold']}{COLORS['bg_blue']} Debate Room Analysis {COLORS['reset']}")
        signal_color = get_signal_color(signal)
        
        print(f"{COLORS['bold']}Final View:{COLORS['reset']} {signal_color}")
        print(f"{COLORS['bold']}Confidence:{COLORS['reset']} {format_confidence(confidence)}")
        print(f"{COLORS['bold']}Bull Confidence:{COLORS['reset']} {format_confidence(bull_conf)}")
        print(f"{COLORS['bold']}Bear Confidence:{COLORS['reset']} {format_confidence(bear_conf)}")
        
        # Print AI model contribution (if any)
        ai_contribution = safe_get(debate_data, "ai_model_contribution", default=None)
        if ai_contribution and safe_get(ai_contribution, "included"):
            ai_signal = safe_get(ai_contribution, "signal", default="Unknown")
            ai_confidence = safe_get(ai_contribution, "confidence", default=0)
            ai_weight = safe_get(ai_contribution, "weight", default=0)
            
            print(f"{COLORS['bold']}AI Model Contribution:{COLORS['reset']} {get_signal_color(ai_signal)} (Confidence: {format_confidence(ai_confidence)}, Weight: {ai_weight*100:.0f}%)")
        
        # Print debate summary
        debate_summary = safe_get(debate_data, "debate_summary", default=[])
        if debate_summary:
            print(f"{COLORS['bold']}Debate Summary:{COLORS['reset']}")
            for i, point in enumerate(debate_summary[:6]):  # Show at most 6 points
                print(f"  {point}")
        
        # Print LLM analysis
        llm_analysis = safe_get(debate_data, "llm_analysis", default="")
        if llm_analysis:
            llm_summary = llm_analysis[:150] + "..." if len(str(llm_analysis)) > 150 else llm_analysis
            print(f"{COLORS['bold']}AI Expert Analysis:{COLORS['reset']}")
            print(f"  {llm_summary}")
        
        print("-" * width)
    
    # Print risk management results
    risk_data = extract_agent_data(state, "risk_management_agent")
    if risk_data:
        max_size = safe_get(risk_data, "max_position_size", default=0)
        risk_score = safe_get(risk_data, "risk_score", default=0)
        trading_action = safe_get(risk_data, "trading_action", default="Unknown")
        risk_metrics = safe_get(risk_data, "risk_metrics", default={})
        
        print(f"\n{COLORS['bold']}{COLORS['bg_blue']} Risk Management {COLORS['reset']}")
        print(f"{COLORS['bold']}Risk Score:{COLORS['reset']} {risk_score}/10")
        print(f"{COLORS['bold']}Trading Recommendation:{COLORS['reset']} {get_signal_color(trading_action)}")
        print(f"{COLORS['bold']}Max Position Size:{COLORS['reset']} {max_size}")
        
        if risk_metrics:
            print(f"{COLORS['bold']}Risk Metrics:{COLORS['reset']}")
            volatility = safe_get(risk_metrics, "volatility", default=0)
            var = safe_get(risk_metrics, "value_at_risk_95", default=0)
            cvar = safe_get(risk_metrics, "conditional_var_95", default=0)
            max_dd = safe_get(risk_metrics, "max_drawdown", default=0)
            
            print(f"  • Volatility: {format_confidence(volatility)}")
            print(f"  • Value at Risk (95%): {format_confidence(var)}")
            print(f"  • Conditional VaR: {format_confidence(cvar)}")
            print(f"  • Max Drawdown: {format_confidence(max_dd)}")
        
        # Print GARCH model results
        garch_model = safe_get(risk_data, "volatility_model", default={})
        if garch_model and "forecast" in garch_model:
            print(f"{COLORS['bold']}Volatility Forecast:{COLORS['reset']} Future volatility trend {'rising' if safe_get(garch_model, 'forecast_annualized', 0) > safe_get(risk_metrics, 'volatility', 0) else 'falling'}")
        
        print("-" * width)
    
    # Print macro analysis results
    macro_data = extract_agent_data(state, "macro_analyst_agent")
    if macro_data:
        signal = safe_get(macro_data, "signal", default="Unknown")
        confidence = safe_get(macro_data, "confidence", default=0)
        macro_env = safe_get(macro_data, "macro_environment", default="Unknown")
        impact = safe_get(macro_data, "impact_on_stock", default="Unknown")
        key_factors = safe_get(macro_data, "key_factors", default=[])
        reasoning = safe_get(macro_data, "reasoning", default="No reasoning provided")
        
        print(f"\n{COLORS['bold']}{COLORS['bg_blue']} Macro Analysis {COLORS['reset']}")
        signal_color = get_signal_color(signal)
        
        print(f"{COLORS['bold']}Signal:{COLORS['reset']} {signal_color}")
        print(f"{COLORS['bold']}Confidence:{COLORS['reset']} {format_confidence(confidence)}")
        
        env_color = get_signal_color(macro_env)
        impact_color = get_signal_color(impact)
        
        print(f"{COLORS['bold']}Macro Environment:{COLORS['reset']} {env_color}")
        print(f"{COLORS['bold']}Impact on Stock:{COLORS['reset']} {impact_color}")
        
        if key_factors:
            print(f"{COLORS['bold']}Key Factors:{COLORS['reset']}")
            for i, factor in enumerate(key_factors[:5]):  # Show at most 5 factors
                print(f"  {i+1}. {factor}")
        
        # Extract multi-asset analysis results (if any)
        multi_asset = safe_get(macro_data, "multi_asset_analysis", default=None)
        if multi_asset:
            tickers = safe_get(multi_asset, "tickers", default=[])
            if tickers:
                print(f"{COLORS['bold']}Multi-Asset Analysis ({len(tickers)} stocks):{COLORS['reset']}")
                # Show risk diversification tips
                diversification_tips = safe_get(multi_asset, "diversification_tips", default=[])
                for tip in diversification_tips[:2]:  # Show at most 2 tips
                    print(f"  • {tip}")
        
        print("-" * width)
    
    # Print portfolio analysis results
    portfolio_analysis_data = extract_agent_data(state, "portfolio_analyzer_agent")
    if portfolio_analysis_data:
        tickers = safe_get(portfolio_analysis_data, "tickers", default=[])
        portfolio_analysis = safe_get(portfolio_analysis_data, "portfolio_analysis", default={})
        risk_analysis = safe_get(portfolio_analysis_data, "risk_analysis", default={})
        efficient_frontier = safe_get(portfolio_analysis_data, "efficient_frontier", default={})
        
        print(f"\n{COLORS['bold']}{COLORS['bg_blue']} Portfolio Analysis ({len(tickers)} stocks) {COLORS['reset']}")
        
        # Print optimal portfolio
        if "max_sharpe" in portfolio_analysis:
            max_sharpe = safe_get(portfolio_analysis, "max_sharpe", default={})
            sharpe = safe_get(max_sharpe, "sharpe_ratio", default=0)
            returns = safe_get(max_sharpe, "return", default=0)
            risk = safe_get(max_sharpe, "risk", default=0)
            
            print(f"{COLORS['bold']}Optimal Portfolio (Max Sharpe Ratio):{COLORS['reset']}")
            print(f"  • Sharpe Ratio: {sharpe:.2f}")
            print(f"  • Expected Return: {format_confidence(returns)}")
            print(f"  • Risk: {format_confidence(risk)}")
            
            # Print weights
            weights = safe_get(max_sharpe, "weights", default={})
            if weights:
                print(f"  • Weight Allocation:")
                for ticker, weight in weights.items():
                    if weight > 0.05:  # Only show assets with weight > 5%
                        print(f"    - {ticker}: {format_confidence(weight)}")
        
        # Print risk analysis
        if risk_analysis:
            var_95 = safe_get(risk_analysis, "var_95", default=0)
            max_dd = safe_get(risk_analysis, "max_drawdown", default=0)
            
            print(f"{COLORS['bold']}Risk Analysis:{COLORS['reset']}")
            print(f"  • Value at Risk (95%): {format_confidence(var_95)}")
            print(f"  • Max Drawdown: {format_confidence(max_dd)}")
            
            # Show best and worst assets
            best_asset = safe_get(risk_analysis, "best_asset", default={})
            worst_asset = safe_get(risk_analysis, "worst_asset", default={})
            
            if best_asset and worst_asset:
                print(f"  • Best Asset: {safe_get(best_asset, 'ticker', default='Unknown')} (Return: {format_confidence(safe_get(best_asset, 'return', default=0))})")
                print(f"  • Worst Asset: {safe_get(worst_asset, 'ticker', default='Unknown')} (Return: {format_confidence(safe_get(worst_asset, 'return', default=0))})")
        
        print("-" * width)
    
    # Print ending
    print("\n" + "=" * width)
    print(f"{COLORS['bold']}{COLORS['bg_green']}{'':^15}Analysis Report Complete{'':^15}{COLORS['reset']}")
    print("=" * width + "\n")

def print_compact_summary(state: Dict[str, Any]) -> None:
    """
    Print simplified investment analysis summary report
    
    Args:
        state: State dictionary containing all Agent analysis results
    """
    if not state:
        logger.warning("Attempting to generate report but state is empty")
        return
    
    ticker = state.get("ticker", "Unknown")
    
    # Extract final decision
    portfolio_manager_data = extract_agent_data(state, "portfolio_management_agent")
    final_decision = state.get("final_decision", {})
    
    # If final_decision is empty, try to get directly from portfolio_manager_data
    if not final_decision and portfolio_manager_data:
        final_decision = portfolio_manager_data
    
    action = safe_get(final_decision, "action", default="Unknown")
    quantity = safe_get(final_decision, "quantity", default=0)
    
    # Print simplified report
    width = 80
    print("\n" + "=" * width)
    print(f"{COLORS['bold']}Stock {ticker} Investment Decision Brief{COLORS['reset']}")
    print("-" * width)
    
    # Print decision summary
    action_color = get_signal_color(action)
    print(f"{COLORS['bold']}Decision:{COLORS['reset']} {action_color} {quantity} shares")
    
    # Collect signals from all Agents
    signals_summary = []
    
    # Define agents to check and their display names
    agents_to_check = [
        ("valuation_agent", "Valuation Analysis"),
        ("fundamentals_agent", "Fundamentals"),
        ("technical_analyst_agent", "Technical Analysis"),
        ("sentiment_agent", "Sentiment Analysis"),
        ("debate_room_agent", "Debate Conclusion"),
        ("ai_model_analyst_agent", "AI Model")
    ]
    
    # Extract signals from each agent
    for agent_name, display_name in agents_to_check:
        agent_data = extract_agent_data(state, agent_name)
        if agent_data:
            signal = safe_get(agent_data, "signal", default="Unknown")
            confidence = safe_get(agent_data, "confidence", default=0)
            signals_summary.append(f"{display_name}: {get_signal_color(signal)} ({format_confidence(confidence)})")
    
    # Print signal summary
    print(f"{COLORS['bold']}Analyst Opinions:{COLORS['reset']}")
    for i, signal in enumerate(signals_summary):
        print(f"  • {signal}")
    
    # Extract risk information
    risk_data = extract_agent_data(state, "risk_management_agent")
    if risk_data:
        risk_score = safe_get(risk_data, "risk_score", default=0)
        print(f"{COLORS['bold']}Risk Score:{COLORS['reset']} {risk_score}/10")
    
    print("=" * width + "\n")

def format_agent_data_for_web(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format analysis state for web-friendly format
    
    Args:
        state: State dictionary containing all Agent analysis results
        
    Returns:
        Formatted data dictionary for web display
    """
    if not state:
        return {"error": "State is empty"}
    
    result = {
        "ticker": state.get("ticker", "Unknown"),
        "date_range": state.get("date_range", {}),
        "timestamp": datetime.now().isoformat(),
        "agents": {},
        "decision": {},
    }
    
    # Extract final decision
    portfolio_manager_data = extract_agent_data(state, "portfolio_management_agent")
    final_decision = state.get("final_decision", {})
    
    # If final_decision is empty, try to get directly from portfolio_manager_data
    if not final_decision and portfolio_manager_data:
        final_decision = portfolio_manager_data
    
    # Format final decision
    result["decision"] = {
        "action": safe_get(final_decision, "action", default="Unknown"),
        "quantity": safe_get(final_decision, "quantity", default=0),
        "confidence": safe_get(final_decision, "confidence", default=0),
        "reasoning": safe_get(final_decision, "reasoning", default="No decision reasoning provided")
    }
    
    # Process all Agent data
    for agent_name, display_name in AGENT_NAMES.items():
        agent_data = extract_agent_data(state, agent_name)
        if agent_data:
            # Basic information
            agent_info = {
                "name": display_name,
                "signal": safe_get(agent_data, "signal", default="Unknown"),
                "confidence": safe_get(agent_data, "confidence", default=0),
            }
            
            # Add specific Agent details
            if agent_name == "valuation_agent":
                agent_info["valuation_gap"] = safe_get(agent_data, "valuation_gap", default=None)
                agent_info["capm_data"] = safe_get(agent_data, "capm_data", default={})
                
            elif agent_name == "risk_management_agent":
                agent_info["risk_score"] = safe_get(agent_data, "risk_score", default=0)
                agent_info["max_position_size"] = safe_get(agent_data, "max_position_size", default=0)
                agent_info["trading_action"] = safe_get(agent_data, "trading_action", default="Unknown")
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
                
            # Add Agent data to result dictionary
            result["agents"][display_name] = agent_info
    
    return result