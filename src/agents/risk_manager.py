import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage

from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.agents.regime_detector import AdvancedRegimeDetector
from src.tools.api import prices_to_df
from src.utils.api_utils import agent_endpoint
from src.utils.logging_config import setup_logger
from src.calc.tail_risk_measures import calculate_historical_var, calculate_conditional_var
from src.calc.volatility_models import fit_garch, forecast_garch_volatility

import json
import ast

##### Risk Management Agent #####

logger = setup_logger('risk_management_agent')

@agent_endpoint("risk_management", "Risk management expert, assessing investment risks and providing risk-adjusted trading recommendations")
def risk_management_agent(state: AgentState):
    """
    Regime-aware risk management system based on 2024-2025 research
    Risk control technology integrating FINSABER, FLAG-Trader, RLMF and other frameworks
    """
    show_workflow_status("Risk Manager")
    show_reasoning = state["metadata"]["show_reasoning"]
    portfolio = state["data"]["portfolio"]
    data = state["data"]

    # Initialize regime detector
    regime_detector = AdvancedRegimeDetector()
    
    # Get asset list
    tickers = data.get("tickers", [])
    if isinstance(tickers, str):
        tickers = [ticker.strip() for ticker in tickers.split(',')]
    
    # If no tickers provided but ticker is provided, use single ticker
    if not tickers and data.get("ticker"):
        tickers = [data["ticker"]]
    
    # Price data for main asset
    prices_df = prices_to_df(data["prices"])
    
    # Perform regime analysis
    regime_features = regime_detector.extract_regime_features(prices_df)
    regime_model_results = regime_detector.fit_regime_model(regime_features)
    current_regime = regime_detector.predict_current_regime(regime_features)
    
    logger.info(f"Risk manager detected market regime: {current_regime.get('regime_name', 'unknown')} (confidence: {current_regime.get('confidence', 0):.2f})")
    
    # Multi-asset portfolio risk analysis
    portfolio_risk_analysis = {}
    
    # If there's multi-asset data, calculate portfolio risk
    all_stock_data = data.get("all_stock_data", {})
    if len(tickers) > 1 and all_stock_data:
        # Extract returns for all stocks
        returns_dict = {}
        for ticker in tickers:
            if ticker in all_stock_data:
                stock_prices = prices_to_df(all_stock_data[ticker]["prices"])
                if not stock_prices.empty:
                    returns_dict[ticker] = stock_prices['close'].pct_change().dropna()
        
        # If there's enough data, calculate portfolio risk
        if len(returns_dict) > 1:
            returns_df = pd.DataFrame(returns_dict)
            
            # Calculate correlation matrix
            correlation_matrix = returns_df.corr()
            
            # Calculate equal-weight portfolio returns
            portfolio_return = returns_df.mean(axis=1)  # Equal-weight portfolio
            
            # Use portfolio returns to calculate risk metrics
            daily_vol = portfolio_return.ewm(span=20, adjust=False).std().iloc[-1]
            volatility = daily_vol * (252 ** 0.5)
            
            # Calculate VaR and CVaR
            var_95 = calculate_historical_var(portfolio_return, confidence_level=0.95)
            cvar_95 = calculate_conditional_var(portfolio_return, confidence_level=0.95)
            
            # Calculate maximum drawdown
            cum_returns = (1 + portfolio_return).cumprod()
            running_max = cum_returns.cummax()
            drawdown = (cum_returns / running_max - 1)
            max_drawdown = drawdown.min()
            
            # Calculate skewness and kurtosis
            skewness = portfolio_return.skew()
            kurtosis = portfolio_return.kurt()
            
            # Save multi-asset portfolio risk analysis results
            portfolio_risk_analysis = {
                "tickers": tickers,
                "volatility": float(volatility),
                "var_95": float(var_95),
                "cvar_95": float(cvar_95),
                "max_drawdown": float(max_drawdown),
                "skewness": float(skewness),
                "kurtosis": float(kurtosis),
                "correlation_matrix": correlation_matrix.to_dict()
            }
            
            # Regime-based diversification suggestions
            diversification_tips = _generate_regime_aware_diversification_tips(
                correlation_matrix, current_regime, high_corr_pairs=[]
            )
            
            # Correlation-based suggestions
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    if correlation_matrix.iloc[i, j] > 0.7:  # High correlation threshold
                        high_corr_pairs.append(f"{correlation_matrix.columns[i]}-{correlation_matrix.columns[j]}")
            
            if high_corr_pairs:
                diversification_tips.append(f"Found high correlation asset pairs: {', '.join(high_corr_pairs)}, consider reducing allocation of one asset to lower concentration risk")
            
            avg_corr = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
            if avg_corr > 0.6:
                diversification_tips.append(f"Portfolio average correlation is high ({avg_corr:.2f}), may need to add other asset classes to improve diversification")
            
            portfolio_risk_analysis["diversification_tips"] = diversification_tips

    # Get debate room assessment
    debate_message = next(
        msg for msg in state["messages"] if msg.name == "debate_room_agent")

    try:
        debate_results = json.loads(debate_message.content)
    except Exception as e:
        debate_results = ast.literal_eval(debate_message.content)

    # 1. Calculate basic risk metrics
    returns = prices_df['close'].pct_change().dropna()
    
    # Use EWMA to optimize volatility calculation
    daily_vol = returns.ewm(span=20, adjust=False).std().iloc[-1]  # EWMA volatility
    volatility = daily_vol * (252 ** 0.5)  # Annualized
    
    # Calculate historical distribution of volatility
    rolling_std = returns.rolling(window=120).std() * (252 ** 0.5)
    volatility_mean = rolling_std.mean()
    volatility_std = rolling_std.std()
    volatility_percentile = (volatility - volatility_mean) / volatility_std if volatility_std != 0 else 0

    # 2. Advanced risk metrics calculation - using functions from calc module
    # 2.1 Historical VaR (95% confidence level) - using calc/tail_risk_measures module
    var_95 = calculate_historical_var(returns, confidence_level=0.95)
    
    # 2.2 Conditional VaR/Expected Shortfall
    cvar_95 = calculate_conditional_var(returns, confidence_level=0.95)
    
    # 2.3 Maximum drawdown calculation
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max - 1)
    max_drawdown = drawdown.min()
    
    # 2.4 Skewness and kurtosis (detect non-normal distribution)
    skewness = returns.skew()
    kurtosis = returns.kurt()
    
    # 2.5 Sortino ratio (only consider downside risk)
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    mean_return = returns.mean() * 252
    sortino_ratio = mean_return / downside_deviation if downside_deviation != 0 else 0

    # 3. Regime-aware risk assessment (based on 2024-2025 research)
    regime_risk_score = _calculate_regime_risk_score(
        current_regime, volatility_percentile, var_95, max_drawdown, skewness, kurtosis
    )

    # 4. Dynamic position sizing calculation - based on FLAG-Trader 2025 framework
    # Consider total portfolio value, not just cash
    current_stock_value = portfolio['stock'] * prices_df['close'].iloc[-1]
    total_portfolio_value = portfolio['cash'] + current_stock_value

    # Regime-aware Kelly criterion optimization
    position_sizing_result = _calculate_regime_aware_position_sizing(
        debate_results, current_regime, returns, total_portfolio_value
    )

    # 5. Stress testing (more comprehensive scenarios)
    stress_test_scenarios = {
        "market_crash": -0.20,            # Market crash
        "moderate_decline": -0.10,        # Moderate decline
        "slight_decline": -0.05,          # Slight decline
        "volatility_spike": -0.15,        # Volatility spike
        "liquidity_crisis": -0.25,        # Liquidity crisis
        "sector_rotation": -0.12,         # Sector rotation
    }

    stress_test_results = {}
    current_position_value = current_stock_value
    if current_position_value > 0:
        for scenario, decline in stress_test_scenarios.items():
            potential_loss = current_position_value * decline
            portfolio_impact = potential_loss / total_portfolio_value if total_portfolio_value != 0 else float('nan')
            
            # Perform VaR analysis for each scenario
            scenario_var = potential_loss * (var_95 / 0.05)  # Adjust VaR to reflect scenario severity
            stress_test_results[scenario] = {
                "potential_loss": float(potential_loss),
                "portfolio_impact": float(portfolio_impact),
                "scenario_var": float(scenario_var)
            }
    else:
        # Use empty dictionary or None when no position
        stress_test_results = {"no_position": True}

    # 6. Risk-adjusted signal analysis
    # Consider debate room confidence
    bull_confidence = debate_results.get("bull_confidence", 0)
    bear_confidence = debate_results.get("bear_confidence", 0)
    debate_confidence = debate_results.get("confidence", 0)

    # Evaluate certainty of debate results
    confidence_diff = abs(bull_confidence - bear_confidence)
    if confidence_diff < 0.1:  # Close debate
        regime_risk_score += 1
    if debate_confidence < 0.3:  # Overall low confidence
        regime_risk_score += 1

    # Cap risk score at 10
    risk_score = min(round(regime_risk_score), 10)

    # 7. Generate trading action
    debate_signal = debate_results.get("signal", "neutral")

    # Get macroeconomic environment information
    macro_message = next(
        (msg for msg in state["messages"] if msg.name == "macro_analyst_agent"), None)
    
    macro_environment_positive = False
    if macro_message:
        try:
            macro_data = json.loads(macro_message.content)
            # Check if macroeconomic analysis is positive
            macro_impact = macro_data.get("impact_on_stock", "neutral")
            macro_environment = macro_data.get("macro_environment", "neutral")
            
            # If both macro environment and stock impact are positive, reduce risk score
            if macro_impact == "positive" and macro_environment == "positive":
                macro_environment_positive = True
                # Reduce risk score but ensure it's not less than 0
                risk_score = max(risk_score - 2, 0)
                logger.info(f"Reduced risk score to {risk_score} based on positive macro environment")
        except Exception as e:
            logger.warning(f"Failed to parse macro analysis data: {e}")
            
    # Add assessment of recent market performance
    recent_period = min(20, len(returns))
    if recent_period > 5:  # Ensure sufficient data
        recent_returns = returns[-recent_period:]
        recent_positive_days = sum(1 for r in recent_returns if r > 0)
        recent_positive_ratio = recent_positive_days / len(recent_returns)
        
        # If most recent trading days have positive returns, reduce risk score
        if recent_positive_ratio > 0.65:  # More than 65% of days have positive returns
            risk_score = max(risk_score - 1, 0)  # Reduce by 1 more point
            logger.info(f"Reduced risk score to {risk_score} based on recent positive market performance")

    logger.info(f"Risk score: {risk_score}")
    logger.info(f"Debate signal: {debate_signal}")
    logger.info(f"Debate confidence: {debate_confidence}")
    # Decision rules based on risk score and debate signal
    if risk_score >= 9:  
        trading_action = "hold"  
        # Reason: Extremely high risk (9-10/10), market is extremely dangerous,
        # Regardless of debate results, immediately stop all trading operations and maintain watch,
        # Avoid major losses in high-risk environment
        
    elif risk_score >= 7:  
        if debate_signal == "bearish":
            trading_action = "sell"  
            # Reason: High risk (7-8/10) and bearish signal, double warning,
            # Should sell in time to avoid potential losses and protect portfolio
        else:
            trading_action = "reduce"  
            # Reason: High risk environment but not bearish, market may have uncertainty,
            # Reduce risk exposure through position reduction, but not complete liquidation,
            # Keep some positions in case of market reversal
            
    elif risk_score >= 6:  # Adjusted: was 5, now raised to 6
        if debate_signal == "bearish" and debate_confidence >= 0.4:  # Adjusted: was 0.3, now raised to 0.4
            trading_action = "sell"
            # Reason: Medium-high risk (6+/10) and strong bearish signal,
            # Need higher confidence to execute sell
        elif debate_signal == "bullish" and debate_confidence >= 0.3:  # Adjusted: was 0.4, now lowered to 0.3
            trading_action = "buy"
            # Reason: Medium-high risk but strong bullish signal,
            # Lower confidence requirement for buying, increase trading opportunities
        else:
            trading_action = "buy"  # Adjusted: changed to buy instead of reduce, more aggressive
            # Reason: Under controllable risk, encourage moderate trading rather than excessive conservatism
            
    elif risk_score >= 4:  # Adjusted: was 3, now 4
        if debate_signal == "bearish" and debate_confidence >= 0.3:  # Adjusted: was 0.25, raised to 0.3
            trading_action = "sell"
            # Reason: Medium risk (4-5/10) with bearish signal,
            # Raise confidence requirement to avoid premature selling
        elif debate_signal == "bearish" and debate_confidence < 0.3:
            trading_action = "buy"  # Adjusted: changed to buy instead of reduce, more aggressive
            # Reason: Controllable risk and weak bearish signal, maintain positive attitude
        elif debate_signal == "bullish" and debate_confidence >= 0.25:  # Adjusted: was 0.35, lowered to 0.25
            trading_action = "buy"
            # Reason: Medium risk with bullish signal, lower threshold to increase trading opportunities
        else:
            trading_action = "buy"  # Adjusted: changed to buy instead of hold, more aggressive
            # Reason: Under medium risk, default tendency is to build positions rather than watch
            
    else:  # risk_score < 4: Low risk range, significantly adjusted to be more aggressive
        if debate_signal == "bullish" and debate_confidence >= 0.2:  # Adjusted: was 0.3, lowered to 0.2
            trading_action = "buy"
            # Reason: Low risk environment with bullish signal, significantly lower threshold
            
        elif debate_signal == "bearish" and debate_confidence >= 0.4:  # Adjusted: was 0.35, raised to 0.4
            trading_action = "sell"
            # Reason: In low risk environment, need stronger bearish signal to sell
            
        elif debate_signal == "neutral" and debate_confidence >= 0.25:  # Adjusted: was 0.3, lowered to 0.25
            # Neutral signal but high confidence, need to combine with market risk score for detailed decision
            
            if regime_risk_score <= 3:  # Adjusted: was 2, now 3
                trading_action = "buy"
                # Reason: Expand low risk range, more aggressive position building
                
            elif regime_risk_score >= 7:  # Adjusted: was 6, now 7
                trading_action = "sell"
                # Reason: Raise risk threshold, reduce premature position reduction
                
            else:  # Market risk moderate (4-6)
                trading_action = "buy"  # Adjusted: changed to buy instead of hold
                # Reason: Default tendency is to build positions rather than watch
                
        elif debate_signal == "bullish" and debate_confidence < 0.2:  # Adjusted: was 0.3, now 0.2
            trading_action = "buy"  # Adjusted: buy even with low confidence
            # Reason: In low risk environment, even weak signals can try to build positions
            
        elif debate_signal == "bearish" and debate_confidence < 0.4:  # Adjusted: was 0.35, now 0.4
            trading_action = "buy"  # Adjusted: changed to buy instead of hold
            # Reason: When bearish signal is not strong, can still build positions in low risk environment
            
        elif debate_signal == "neutral" and debate_confidence < 0.25:  # Adjusted: was 0.3, now 0.25
            trading_action = "buy"  # Adjusted: changed to buy instead of hold
            # Reason: In low risk environment, can moderately build positions even with unclear signals
            
        else:
            # Catch all unexpected situations
            trading_action = "buy"  # Adjusted: changed to buy instead of hold, more aggressive
            # Reason: In low risk environment, default tendency is to build positions rather than watch

    # 8. GARCH model fitting and forecasting
    garch_results = {}
    try:
        # Get sufficient data length for GARCH fitting
        if len(returns) >= 100:  # Ensure sufficient data
            garch_params, log_likelihood = fit_garch(returns.values)
            volatility_forecast = forecast_garch_volatility(returns.values, garch_params, 
                                                         forecast_horizon=10)
            
            # Add results to analysis
            regime_risk_score += 1 if garch_params['persistence'] > 0.95 else 0  # High persistence indicates higher risk
            
            # Save GARCH results
            garch_results = {
                'model_type': 'GARCH(1,1)',
                'parameters': {
                    'omega': float(garch_params['omega']),
                    'alpha': float(garch_params['alpha']),
                    'beta': float(garch_params['beta']),
                    'persistence': float(garch_params['persistence'])
                },
                'log_likelihood': float(log_likelihood),
                'forecast': [float(v) for v in volatility_forecast],
                'forecast_annualized': [float(v * np.sqrt(252)) for v in volatility_forecast]
            }
    except Exception as e:
        logger.warning(f"GARCH model fitting failed: {e}")
        garch_results = {"error": str(e)}

    # 9. Build output message
    message_content = {
        "max_position_size": float(position_sizing_result["max_position_size"]),
        "risk_score": risk_score,
        "trading_action": trading_action,
        "risk_metrics": {
            "volatility": float(volatility),
            "value_at_risk_95": float(var_95),
            "conditional_var_95": float(cvar_95),  
            "max_drawdown": float(max_drawdown),
            "skewness": float(skewness),           
            "kurtosis": float(kurtosis),           
            "sortino_ratio": float(sortino_ratio), 
            "regime_risk_score": regime_risk_score,
            "stress_test_results": stress_test_results,
            "macro_environment_assessment": { 
                "global_risks": regime_risk_score > 5,
                "liquidity_concerns": regime_risk_score > 7,
                "volatility_regime": "high" if volatility > 0.3 else "medium" if volatility > 0.2 else "low"
            }
        },
        "position_sizing": {
            "kelly_fraction": float(position_sizing_result["kelly_fraction"]),
            "win_rate": float(position_sizing_result["win_rate"]),
            "win_loss_ratio": float(position_sizing_result["win_loss_ratio"]),
            "risk_adjustment": float(position_sizing_result["risk_adjustment"]),
            "total_portfolio_value": float(total_portfolio_value)
        },
        "debate_analysis": {
            "bull_confidence": bull_confidence,
            "bear_confidence": bear_confidence,
            "debate_confidence": debate_confidence,
            "debate_signal": debate_signal
        },
        "volatility_model": garch_results,
        "reasoning": f"Risk Score {risk_score}/10: Market Risk={regime_risk_score}, "
                     f"Volatility={volatility:.2%}, VaR={var_95:.2%}, CVaR={cvar_95:.2%}, "
                     f"Max Drawdown={max_drawdown:.2%}, Skewness={skewness:.2f}, "
                     f"Debate Signal={debate_signal}, Kelly Recommended Ratio={position_sizing_result['kelly_fraction']:.2f}"
    }
    
    # If there are multi-asset analysis results, add to output
    if portfolio_risk_analysis:
        message_content["portfolio_risk_analysis"] = portfolio_risk_analysis
        
        # Add multi-asset risk tips to reasoning
        if "diversification_tips" in portfolio_risk_analysis and portfolio_risk_analysis["diversification_tips"]:
            message_content["reasoning"] += "\nMulti-asset Risk Analysis: " + " ".join(portfolio_risk_analysis["diversification_tips"])

    # Create risk management message
    message = HumanMessage(
        content=json.dumps(message_content),
        name="risk_management_agent",
    )

    if show_reasoning:
        show_agent_reasoning(message_content, "Risk Management Agent")
        # Save reasoning information to metadata for API use
        state["metadata"]["agent_reasoning"] = message_content

    show_workflow_status("Risk Manager", "completed")
    return {
        "messages": state["messages"] + [message],
        "data": {
            **data,
            "risk_analysis": message_content
        },
        "metadata": state["metadata"],
    }


def _generate_regime_aware_diversification_tips(correlation_matrix, current_regime, high_corr_pairs):
    """Generate diversification suggestions based on market regime"""
    tips = []
    regime_name = current_regime.get('regime_name', 'unknown')
    
    if regime_name == "crisis_regime":
        tips.append("In crisis regime, traditional diversification effects weaken, consider increasing safe-haven assets (bonds, gold) allocation")
        tips.append("During crisis periods, asset correlations tend toward 1, suggest reducing overall risk exposure")
    elif regime_name == "high_volatility_mean_reverting":
        tips.append("In high volatility oscillating markets, can appropriately increase hedging strategies and volatility trading")
        tips.append("Oscillating markets suit range trading, suggest setting stricter stop-loss and take-profit")
    elif regime_name == "low_volatility_trending":
        tips.append("Low volatility trending markets suit increasing momentum strategy allocation")
        tips.append("In trending markets, can appropriately increase single-direction exposure")
    
    return tips


def _calculate_regime_risk_score(current_regime, volatility_percentile, var_95, max_drawdown, skewness, kurtosis):
    """Calculate risk score based on market regime"""
    regime_name = current_regime.get('regime_name', 'unknown')
    regime_confidence = current_regime.get('confidence', 0.5)
    
    # Base risk score
    risk_score = 0
    
    # Regime-specific risk adjustments
    if regime_name == "crisis_regime":
        risk_score += 3  # Crisis regime has high base risk
        if regime_confidence > 0.7:
            risk_score += 1  # High confidence crisis prediction
    elif regime_name == "high_volatility_mean_reverting":
        risk_score += 2  # High volatility regime medium risk
    elif regime_name == "low_volatility_trending":
        risk_score += 1  # Low volatility trending regime lower risk
    
    # Volatility score (based on percentile)
    if volatility_percentile > 1.5:     # Above 1.5 standard deviations
        risk_score += 2
    elif volatility_percentile > 1.0:   # Above 1 standard deviation
        risk_score += 1

    # VaR score (based on historical distribution)
    if var_95 > 0.03:
        risk_score += 2
    elif var_95 > 0.02:
        risk_score += 1

    # Maximum drawdown score
    if abs(max_drawdown) > 0.20:  # Severe drawdown
        risk_score += 2
    elif abs(max_drawdown) > 0.10:
        risk_score += 1
        
    # Distribution abnormality score (significantly non-normal)
    if abs(skewness) > 1.0 or kurtosis > 5.0:
        risk_score += 1
    
    return risk_score


def _calculate_regime_aware_position_sizing(debate_results, current_regime, returns, total_portfolio_value):
    """Dynamic position sizing calculation based on regime"""
    regime_name = current_regime.get('regime_name', 'unknown')
    regime_confidence = current_regime.get('confidence', 0.5)
    
    # Adjust win rate from debate results
    bull_confidence = debate_results.get("bull_confidence", 0.5)
    bear_confidence = debate_results.get("bear_confidence", 0.5)
    if bull_confidence > bear_confidence:
        win_rate = bull_confidence
    else:
        win_rate = 1 - bear_confidence
    
    # Regime-specific win rate adjustments
    if regime_name == "crisis_regime":
        win_rate *= 0.7  # Reduce win rate expectations during crisis
    elif regime_name == "low_volatility_trending":
        win_rate *= 1.1  # Increase win rate expectations in trending markets
        win_rate = min(win_rate, 0.8)  # Upper limit
    
    # Optimize return expectations with volatility model
    try:
        # Use GARCH model to predict future volatility
        if len(returns) >= 100:  # Ensure sufficient data
            garch_params, _ = fit_garch(returns.values)
            volatility_forecast = forecast_garch_volatility(returns.values, garch_params, 
                                                         forecast_horizon=10)
            
            # Adjust win rate based on volatility forecast
            avg_forecast = np.mean(volatility_forecast)
            current_vol = returns.std()
            volatility_trend = avg_forecast / current_vol - 1 if current_vol != 0 else 0
            
            # Adjust win rate
            if volatility_trend > 0.1:  # Volatility expected to rise significantly
                win_rate = max(0.3, win_rate - 0.1)  # Minimum set to 0.3
            elif volatility_trend < -0.1:  # Volatility expected to fall significantly
                win_rate = min(0.8, win_rate + 0.05)  # Maximum set to 0.8
    except Exception as e:
        logger.warning(f"GARCH model calculation failed: {e}")
    
    # Calculate average winning returns and loss ratio
    avg_gain = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0.005
    avg_loss = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0.005
    win_loss_ratio = avg_gain / avg_loss if avg_loss != 0 else 1.5
    
    # Apply improved Kelly formula
    kelly_fraction = win_rate - ((1 - win_rate) / win_loss_ratio)
    kelly_fraction = max(0.05, kelly_fraction)
    
    # Regime-specific conservative factor
    if regime_name == "crisis_regime":
        conservative_factor = 0.3  # More conservative during crisis
    elif regime_name == "high_volatility_mean_reverting":
        conservative_factor = 0.4  # Conservative during high volatility
    else:
        conservative_factor = 0.5  # Normal situation
    
    # Risk adjustment based on regime confidence
    risk_adjustment = regime_confidence if regime_confidence > 0.6 else 0.5
    
    # Final position size
    max_position_size = total_portfolio_value * kelly_fraction * conservative_factor * risk_adjustment
    max_position_size = max(max_position_size, total_portfolio_value * 0.02)  # Set minimum position
    max_position_size = min(max_position_size, total_portfolio_value * 0.4)   # Set maximum position
    
    return {
        "max_position_size": max_position_size,
        "kelly_fraction": kelly_fraction,
        "win_rate": win_rate,
        "win_loss_ratio": win_loss_ratio,
        "risk_adjustment": risk_adjustment,
        "conservative_factor": conservative_factor,
        "regime_adjustment": f"Applied {regime_name} specific adjustments"
    }