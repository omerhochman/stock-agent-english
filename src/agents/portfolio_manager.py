from langchain_core.messages import HumanMessage
import json
import numpy as np
import pandas as pd

from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.tools.openrouter_config import get_chat_completion
from src.utils.api_utils import agent_endpoint, log_llm_interaction
from src.tools.api import prices_to_df
from src.tools.factor_data_api import get_multiple_index_data, get_risk_free_rate
from src.calc.portfolio_optimization import portfolio_volatility
from src.utils.logging_config import setup_logger
from src.utils.formatting_utils import format_market_data_summary

# Setup logger
logger = setup_logger('portfolio_management_agent')

@agent_endpoint("portfolio_management", "Responsible for portfolio management and final trading decisions")
def portfolio_management_agent(state: AgentState):
    """Responsible for portfolio management and trading decisions"""
    show_workflow_status("Portfolio Manager")
    show_reasoning = state["metadata"]["show_reasoning"]
    portfolio = state["data"]["portfolio"]
    data = state["data"]
    
    # Get asset list
    tickers = data.get("tickers", [])
    if isinstance(tickers, str):
        tickers = [ticker.strip() for ticker in tickers.split(',')]
    
    # If no tickers provided but ticker is provided, use single ticker
    if not tickers and data.get("ticker"):
        tickers = [data["ticker"]]
        
    # Use main asset for analysis (maintain backward compatibility)
    ticker = tickers[0] if tickers else ""
    prices = data.get("prices", [])
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    
    # If there are multi-asset analysis results, incorporate them into decisions
    portfolio_analysis = data.get("portfolio_analysis", {})
    
    # Get latest price
    current_price = prices[-1]['close'] if prices else 0
    
    # Get signals from various analysts
    technical_message = next(
        msg for msg in state["messages"] if msg.name == "technical_analyst_agent")
    fundamentals_message = next(
        msg for msg in state["messages"] if msg.name == "fundamentals_agent")
    sentiment_message = next(
        msg for msg in state["messages"] if msg.name == "sentiment_agent")
    valuation_message = next(
        msg for msg in state["messages"] if msg.name == "valuation_agent")
    risk_message = next(
        msg for msg in state["messages"] if msg.name == "risk_management_agent")
    macro_message = next(
        msg for msg in state["messages"] if msg.name == "macro_analyst_agent")
    ai_model_message = next(
        (msg for msg in state["messages"] if msg.name == "ai_model_analyst_agent"), None)

    system_message = {
        "role": "system",
        "content": """You are a portfolio manager making final trading decisions.
            Your job is to make a trading decision based on the team's analysis while strictly adhering
            to risk management constraints.

            RISK MANAGEMENT CONSTRAINTS:
            - You MUST NOT exceed the max_position_size specified by the risk manager
            - You MUST follow the trading_action (buy/sell/hold) recommended by risk management
            - These are hard constraints that cannot be overridden by other signals

            When weighing the different signals for direction and timing:
            0. AI Models Analysis (15% weight)
               - Machine learning predictions based on historical patterns
               - Reinforcement learning optimal policies
               - Automatically discovered trading factors
            1. Valuation Analysis (35% weight)
               - Primary driver of fair value assessment
               - Determines if price offers good entry/exit point
            
            2. Fundamental Analysis (30% weight)
               - Business quality and growth assessment
               - Determines conviction in long-term potential
            
            3. Technical Analysis (25% weight)
               - Secondary confirmation
               - Helps with entry/exit timing
               
            4. Macro Analysis (15% weight)
               - Provides broader economic context
               - Helps assess external risks and opportunities
            
            5. Sentiment Analysis (10% weight)
               - Final consideration
               - Can influence sizing within risk limits
            
            The decision process should be:
            0. First check risk management constraints
            1. Consider AI model predictions and signals
            2. Then evaluate valuation signal
            3. Then evaluate fundamentals signal
            4. Consider macro environment analysis
            5. Use technical analysis for timing
            6. Consider sentiment for final adjustment
            
            Provide the following in your output:
            - "action": "buy" | "sell" | "hold",
            - "quantity": <positive integer>
            - "confidence": <float between 0 and 1>
            - "agent_signals": <list of agent signals including agent name, signal (bullish | bearish | neutral), and their confidence>
            - "reasoning": <concise explanation of the decision including how you weighted the signals>

            Trading Rules:
            - Never exceed risk management position limits
            - Only buy if you have available cash
            - Only sell if you have shares to sell
            - Quantity must be ≤ current position for sells
            - Quantity must be ≤ max_position_size from risk management"""
    }

    # Create user message
    user_message = {
        "role": "user",
        "content": f"""Based on the team's analysis below, make your trading decision.

            Technical Analysis Trading Signal: {technical_message.content}
            Fundamental Analysis Trading Signal: {fundamentals_message.content}
            Sentiment Analysis Trading Signal: {sentiment_message.content}
            Valuation Analysis Trading Signal: {valuation_message.content}
            Risk Management Trading Signal: {risk_message.content}
            Macro Analysis Trading Signal: {macro_message.content}
            
            {f'Portfolio Analysis: {json.dumps(portfolio_analysis) if portfolio_analysis else "No portfolio analysis available"}'}

            Here is the current portfolio:
            Portfolio:
            Cash: {portfolio['cash']:.2f}
            Current Position: {portfolio['stock']} shares
            
            {'Multiple assets are being analyzed: ' + ', '.join(tickers) if len(tickers) > 1 else ''}
            """
    }

    if ai_model_message:
        try:
            ai_model_results = json.loads(ai_model_message.content)
            
            # Add AI model analysis results
            user_message["content"] += f"\nAI Model Analysis Signal: {ai_model_message.content}\n"
            
            # If it's multi-asset analysis, add multi-asset allocation recommendations
            if ai_model_results.get("multi_asset", False) and "portfolio_allocation" in ai_model_results:
                allocation = ai_model_results["portfolio_allocation"].get("allocation", {})
                if allocation:
                    user_message["content"] += f"\nAI Model Recommended Asset Allocation:\n"
                    for ticker, weight in allocation.items():
                        user_message["content"] += f"- {ticker}: {weight*100:.2f}%\n"
        except Exception as e:
            logger.error(f"Error parsing AI model analysis results: {e}")

    user_message["content"] += """

            Only include the action, quantity, reasoning, confidence, and agent_signals in your output as JSON.  Do not include any JSON markdown.

            Remember, the action must be either buy, sell, or hold.
            You can only buy if you have available cash.
            You can only sell if you have shares in the portfolio to sell."""

    # Record LLM request
    request_data = {
        "system_message": system_message,
        "user_message": user_message
    }

    # Get LLM analysis results
    result = get_chat_completion([system_message, user_message])

    # Record LLM interaction
    state["metadata"]["current_agent_name"] = "portfolio_management"
    log_llm_interaction(state)(
        lambda: result
    )()

    # Parse LLM response
    try:
        llm_decision = json.loads(result)
    except Exception as e:
        logger.error(f"Failed to parse LLM response: {e}")
        # If API call fails or parsing error, use default conservative decision
        llm_decision = {
            "action": "hold",
            "quantity": 0,
            "confidence": 0.7,
            "agent_signals": [
                {"agent_name": "technical_analysis", "signal": "neutral", "confidence": 0.0},
                {"agent_name": "fundamental_analysis", "signal": "bullish", "confidence": 1.0},
                {"agent_name": "sentiment_analysis", "signal": "bullish", "confidence": 0.6},
                {"agent_name": "valuation_analysis", "signal": "bearish", "confidence": 0.67},
                {"agent_name": "risk_management", "signal": "hold", "confidence": 1.0}
            ],
            "reasoning": "API error occurred. Following risk management signal to hold."
        }

    # Use more advanced portfolio optimization methods
    optimized_decision = optimize_portfolio_decision_advanced(
        llm_decision=llm_decision,
        portfolio=portfolio,
        current_price=current_price,
        risk_message=risk_message.content,
        ticker=ticker,
        prices_df=prices_to_df(prices),
        start_date=start_date,
        end_date=end_date,
        data=data,
        ai_model_message=ai_model_message.content if ai_model_message else None,
    )

    # Create portfolio management message
    # Wrap in decision format to ensure backtesting system can parse correctly
    decision_content = {
        "decision": {
            "action": optimized_decision.get("action", "hold"),
            "quantity": optimized_decision.get("quantity", 0),
            "confidence": optimized_decision.get("confidence", 0.5)
        },
        "full_analysis": optimized_decision  # Keep complete analysis results
    }
    
    message = HumanMessage(
        content=json.dumps(decision_content),
        name="portfolio_management",
    )

    # Save reasoning information
    if show_reasoning:
        show_agent_reasoning(optimized_decision, "Portfolio Management Agent")
        state["metadata"]["agent_reasoning"] = optimized_decision

    show_workflow_status("Portfolio Manager", "completed")
    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
        "metadata": state["metadata"],
    }


def optimize_portfolio_decision_advanced(llm_decision, portfolio, current_price, risk_message, ticker, prices_df, start_date, end_date, data=None, ai_model_message=None):
    """
    Optimize portfolio decisions using modern portfolio theory and advanced risk management principles
    
    Args:
        llm_decision: Initial decision generated by LLM
        portfolio: Current portfolio state
        current_price: Current stock price
        risk_message: Risk management agent information
        ticker: Stock code
        prices_df: Historical price data DataFrame
        start_date: Start date
        end_date: End date
        data: Complete data state, including multi-asset information
        ai_model_message: AI model analysis result message content
        
    Returns:
        dict: Optimized investment decision
    """
    # Parse basic decision
    action = llm_decision.get("action", "hold")
    original_quantity = llm_decision.get("quantity", 0)
    confidence = llm_decision.get("confidence", 0.5)
    reasoning = llm_decision.get("reasoning", "")
    agent_signals = llm_decision.get("agent_signals", [])
    
    # Parse risk management information
    try:
        risk_data = json.loads(risk_message)
        max_position_size = risk_data.get("max_position_size", 0)
        risk_score = risk_data.get("risk_score", 5)
        trading_action = risk_data.get("trading_action", "hold")
        # Extract GARCH model results (if any)
        volatility_model = risk_data.get("volatility_model", {})
    except Exception as e:
        logger.error(f"Failed to parse risk management information: {e}")
        # Use conservative default values when parsing fails
        max_position_size = portfolio['cash'] / current_price * 0.25 if current_price > 0 else 0
        risk_score = 5
        trading_action = "hold"
        volatility_model = {}
    
    # Current portfolio value
    total_portfolio_value = portfolio['cash'] + (portfolio['stock'] * current_price)
    
    # 1. Get actual market data
    # Get stock and market index data
    market_data = {}
    market_volatility = 0.15  # Default value, only used when actual data cannot be obtained
    
    try:
        # Get major market index data (CSI 300, Shenzhen Component Index, ChiNext Index)
        indices = ["000300", "399001", "399006"]  # CSI 300, Shenzhen Component Index, ChiNext Index
        market_indices = get_multiple_index_data(indices, start_date, end_date)
        
        # Create market index return series
        market_returns_dict = {}
        for idx_name, idx_data in market_indices.items():
            if isinstance(idx_data, pd.DataFrame) and 'close' in idx_data.columns:
                idx_df = idx_data.copy()
                idx_df['return'] = idx_df['close'].pct_change()
                market_returns_dict[idx_name] = idx_df['return'].dropna()
        
        # Use CSI 300 as main market benchmark
        if '000300' in market_returns_dict:
            market_returns = market_returns_dict['000300']
            # Calculate actual market volatility
            market_volatility = market_returns.std() * np.sqrt(252)
            market_data['market_returns'] = market_returns
            market_data['market_volatility'] = market_volatility
            logger.info(f"Successfully obtained market data - Market volatility: {market_volatility:.2%}")
        else:
            logger.warning("Unable to obtain CSI 300 index data, using default market volatility")
    except Exception as e:
        logger.error(f"Failed to obtain market data: {e}")
    
    # 2. Get actual risk-free rate data
    try:
        # Use factor_data_api to get risk-free rate data
        risk_free_data = get_risk_free_rate(start_date=start_date, end_date=end_date)
        
        # Calculate average risk-free rate
        if not risk_free_data.empty:
            risk_free_rate = risk_free_data.mean()
            logger.info(f"Successfully obtained risk-free rate data: {risk_free_rate:.2%}")
        else:
            # If data is empty, use default value
            risk_free_rate = 0.03
            logger.warning(f"Risk-free rate data is empty, using default value: {risk_free_rate:.2%}")
    except Exception as e:
        # If acquisition fails, use reasonable default value
        risk_free_rate = 0.03
        logger.error(f"Failed to obtain risk-free rate: {e}, using default value: {risk_free_rate:.2%}")
    
    # 3. Calculate stock returns and volatility
    stock_returns = None
    stock_volatility = None
    if not prices_df.empty:
        try:
            stock_returns = prices_df['close'].pct_change().dropna()
            # Use portfolio_volatility function to calculate actual volatility
            stock_volatility = portfolio_volatility(np.ones(1), np.array([[stock_returns.var()]])) * np.sqrt(252)
            market_data['stock_returns'] = stock_returns
            market_data['stock_volatility'] = stock_volatility
            logger.info(f"Successfully calculated stock volatility: {stock_volatility:.2%}")
        except Exception as e:
            logger.error(f"Failed to calculate stock volatility: {e}")
            stock_volatility = 0.3  # Default value
    
    # 4. Calculate relative volatility (Beta) and Beta-adjusted portfolio optimization
    beta = 1.0  # Default value
    if stock_returns is not None and 'market_returns' in market_data:
        market_returns = market_data['market_returns']
        common_dates = stock_returns.index.intersection(market_returns.index)
        
        if len(common_dates) > 30:  # Ensure sufficient data points
            # Calculate covariance
            covariance = stock_returns[common_dates].cov(market_returns[common_dates])
            # Calculate market variance
            market_variance = market_returns[common_dates].var()
            # Calculate Beta
            if market_variance > 0:
                beta = covariance / market_variance
                market_data['beta'] = beta
                logger.info(f"Successfully calculated stock Beta: {beta:.2f}")
    
    # 5. Parse AI model analysis results
    ai_model_data = None
    if ai_model_message:
        try:
            ai_model_data = json.loads(ai_model_message)
            logger.info(f"Successfully parsed AI model analysis data: {ai_model_data.get('signal', 'neutral')}")
        except Exception as e:
            logger.error(f"Failed to parse AI model analysis data: {e}")
    
    # 6. Apply optimization functions from calculation module to enhance decisions
    portfolio_optimization_results = {}
    try:
        if stock_returns is not None and len(stock_returns) > 30:
            # Use EWMA method to estimate covariance matrix
            stock_return_mean = stock_returns.mean()
            stock_return_std = stock_returns.std()
            
            # Calculate risk-adjusted expected return (using technical and fundamental signals)
            expected_return_multiplier = 1.0
            tech_signal = next((s for s in agent_signals if s.get("agent_name", "").lower() == "technical_analysis"), None)
            fund_signal = next((s for s in agent_signals if s.get("agent_name", "").lower() == "fundamental_analysis"), None)
            
            if tech_signal and fund_signal:
                tech_value = {"bullish": 1, "neutral": 0, "bearish": -1}.get(tech_signal.get("signal", "neutral").lower(), 0)
                fund_value = {"bullish": 1, "neutral": 0, "bearish": -1}.get(fund_signal.get("signal", "neutral").lower(), 0)
                signal_value = (tech_value * 0.4 + fund_value * 0.6)  # Fundamental analysis has higher weight
                expected_return_multiplier += signal_value * 0.5  # Adjust baseline expected return
            
            # 7. If AI model data exists, adjust return expectations
            if ai_model_data:
                ai_signal = ai_model_data.get("signal", "neutral")
                ai_confidence = ai_model_data.get("confidence", 0.5)
                ai_value = {"bullish": 1, "neutral": 0, "bearish": -1}.get(ai_signal.lower(), 0)
                
                # Use AI model signal to further adjust expected return multiplier
                ai_adjustment = ai_value * ai_confidence * 0.2  # AI model contributes 20% adjustment
                expected_return_multiplier += ai_adjustment
                logger.info(f"AI model contribution adjustment: {ai_adjustment:.2f}, adjusted expected return multiplier: {expected_return_multiplier:.2f}")
            
            # Use GARCH forecast to adjust volatility expectations
            volatility_adjustment = 1.0
            if volatility_model and "forecast" in volatility_model:
                try:
                    garch_forecast = volatility_model["forecast"]
                    if isinstance(garch_forecast, list) and len(garch_forecast) > 0:
                        avg_forecast = sum(garch_forecast) / len(garch_forecast)
                        current_vol = stock_return_std
                        # If predicted volatility rises, reduce position; if it falls, increase position
                        volatility_adjustment = current_vol / avg_forecast if avg_forecast > 0 else 1.0
                        volatility_adjustment = min(max(volatility_adjustment, 0.7), 1.3)  # Limit adjustment range
                except Exception as e:
                    logger.error(f"Volatility adjustment calculation error: {e}")
            
            # Adjusted expected return and volatility
            adjusted_expected_return = stock_return_mean * expected_return_multiplier * 252  # Annualized
            adjusted_volatility = stock_return_std * np.sqrt(252) / volatility_adjustment  # Annualized and adjusted
            
            # Use actual risk-free rate to calculate Sharpe ratio
            sharpe_ratio = (adjusted_expected_return - risk_free_rate) / adjusted_volatility if adjusted_volatility > 0 else 0
            
            # Use actual calculated Beta for optimization adjustment
            beta_adjusted_return = risk_free_rate + beta * (adjusted_expected_return - risk_free_rate)
            
            # Create portfolio for optimization
            portfolio_optimization_results = {}
            try:
                if stock_returns is not None and len(stock_returns) > 30:
                    # Check if there are multiple asset data
                    all_stock_data = data.get("all_stock_data", {})
                    tickers = data.get("tickers", [ticker])
                    
                    if len(tickers) > 1 and all_stock_data:
                        # Extract returns for all stocks
                        returns_dict = {}
                        for symbol in tickers:
                            if symbol in all_stock_data:
                                stock_prices = prices_to_df(all_stock_data[symbol]["prices"])
                                if not stock_prices.empty:
                                    returns_dict[symbol] = stock_prices['close'].pct_change().dropna()
                        
                        # If there is sufficient data, calculate portfolio optimization
                        if len(returns_dict) > 1:
                            returns_df = pd.DataFrame(returns_dict)
                            
                            # Calculate average returns and covariance matrix
                            mean_returns = returns_df.mean() * 252  # Annualized
                            cov_matrix = returns_df.cov() * 252  # Annualized
                            
                            # Convert Series to numpy arrays
                            asset_returns = mean_returns.values
                            asset_cov = cov_matrix.values
                            
                            # Calculate optimal weights (using portfolio optimization function)
                            try:
                                from src.calc.portfolio_optimization import optimize_portfolio
                                
                                # Create expected returns Series and covariance matrix DataFrame
                                expected_returns = pd.Series(mean_returns, index=tickers)
                                covariance_matrix = pd.DataFrame(cov_matrix, index=tickers, columns=tickers)
                                
                                # Maximum Sharpe ratio portfolio
                                max_sharpe_portfolio = optimize_portfolio(
                                    expected_returns=expected_returns,
                                    cov_matrix=covariance_matrix,
                                    risk_free_rate=risk_free_rate,
                                    objective='sharpe'
                                )
                                
                                # Minimum risk portfolio
                                min_risk_portfolio = optimize_portfolio(
                                    expected_returns=expected_returns,
                                    cov_matrix=covariance_matrix,
                                    risk_free_rate=risk_free_rate,
                                    objective='min_risk'
                                )
                                
                                # 8. If there are AI model portfolio allocation recommendations, consider them
                                ai_portfolio_weights = None
                                if ai_model_data and ai_model_data.get("multi_asset", False):
                                    portfolio_allocation = ai_model_data.get("portfolio_allocation", {})
                                    allocation = portfolio_allocation.get("allocation", {})
                                    
                                    if allocation:
                                        ai_portfolio_weights = pd.Series(allocation)
                                        # Ensure weights are normalized to 1
                                        if ai_portfolio_weights.sum() > 0:
                                            ai_portfolio_weights = ai_portfolio_weights / ai_portfolio_weights.sum()
                                            
                                            # Create AI model recommended portfolio
                                            ai_return = (expected_returns * ai_portfolio_weights).sum()
                                            ai_risk = np.sqrt(np.dot(ai_portfolio_weights, np.dot(covariance_matrix, ai_portfolio_weights)))
                                            ai_sharpe = (ai_return - risk_free_rate) / ai_risk if ai_risk > 0 else 0
                                            
                                            # Add AI model portfolio to results
                                            portfolio_optimization_results["ai_model"] = {
                                                "weights": ai_portfolio_weights.to_dict(),
                                                "expected_return": float(ai_return),
                                                "expected_volatility": float(ai_risk),
                                                "sharpe_ratio": float(ai_sharpe)
                                            }
                                
                                # Save optimization results
                                portfolio_optimization_results = {
                                    "multi_asset": True,
                                    "tickers": tickers,
                                    "max_sharpe": {
                                        "weights": max_sharpe_portfolio['weights'].to_dict(),
                                        "expected_return": float(max_sharpe_portfolio['return']),
                                        "expected_volatility": float(max_sharpe_portfolio['risk']),
                                        "sharpe_ratio": float(max_sharpe_portfolio['sharpe_ratio'])
                                    },
                                    "min_risk": {
                                        "weights": min_risk_portfolio['weights'].to_dict(),
                                        "expected_return": float(min_risk_portfolio['return']),
                                        "expected_volatility": float(min_risk_portfolio['risk']),
                                        "sharpe_ratio": float(min_risk_portfolio['sharpe_ratio'])
                                    }
                                }
                                
                                # Use maximum Sharpe ratio portfolio data
                                adjusted_expected_return = max_sharpe_portfolio['return']
                                adjusted_volatility = max_sharpe_portfolio['risk']
                                sharpe_ratio = max_sharpe_portfolio['sharpe_ratio']
                                
                                # Keep these fields for compatibility with single asset logic
                                portfolio_optimization_results.update({
                                    "expected_annual_return": float(adjusted_expected_return),
                                    "expected_annual_volatility": float(adjusted_volatility),
                                    "sharpe_ratio": float(sharpe_ratio),
                                    "volatility_adjustment": float(volatility_adjustment),
                                    "return_multiplier": float(expected_return_multiplier),
                                    "beta": float(beta),
                                    "market_volatility": float(market_volatility),
                                    "risk_free_rate": float(risk_free_rate)
                                })
                                
                            except Exception as e:
                                logger.error(f"Multi-asset portfolio optimization failed: {e}")
                        else:
                            # Single asset optimization logic
                            weights = np.array([1.0])  # 100% invested in this single asset
                            asset_returns = np.array([beta_adjusted_return])
                            asset_cov = np.array([[adjusted_volatility**2]])
                            
                            # Save optimization results
                            portfolio_optimization_results = {
                                "multi_asset": False,
                                "expected_annual_return": float(adjusted_expected_return),
                                "expected_annual_volatility": float(adjusted_volatility),
                                "beta_adjusted_return": float(beta_adjusted_return),
                                "sharpe_ratio": float(sharpe_ratio),
                                "volatility_adjustment": float(volatility_adjustment),
                                "return_multiplier": float(expected_return_multiplier),
                                "beta": float(beta),
                                "market_volatility": float(market_volatility),
                                "risk_free_rate": float(risk_free_rate)
                            }
                    else:
                        # Single asset optimization logic
                        weights = np.array([1.0])  # 100% invested in this single asset
                        asset_returns = np.array([beta_adjusted_return])
                        asset_cov = np.array([[adjusted_volatility**2]])
                        
                        # Save optimization results
                        portfolio_optimization_results = {
                            "multi_asset": False,
                            "expected_annual_return": float(adjusted_expected_return),
                            "expected_annual_volatility": float(adjusted_volatility),
                            "beta_adjusted_return": float(beta_adjusted_return),
                            "sharpe_ratio": float(sharpe_ratio),
                            "volatility_adjustment": float(volatility_adjustment),
                            "return_multiplier": float(expected_return_multiplier),
                            "beta": float(beta),
                            "market_volatility": float(market_volatility),
                            "risk_free_rate": float(risk_free_rate)
                        }
            except Exception as e:
                logger.error(f"Portfolio optimization calculation failed: {e}")
                portfolio_optimization_results = {"error": str(e)}
    except Exception as e:
        logger.error(f"Portfolio optimization calculation failed: {e}")
        portfolio_optimization_results = {"error": str(e)}
        
    # 9. Apply position management rules - improved Kelly criterion
    # Use portfolio optimization results to adjust Kelly fraction
    kelly_fraction = confidence * 2 - 1  # Convert confidence to Kelly fraction (-1 to 1)
    kelly_fraction = max(0, kelly_fraction)  # Ensure non-negative
    
    # Adjust Kelly fraction based on portfolio optimization results
    if "sharpe_ratio" in portfolio_optimization_results:
        sharpe_ratio = portfolio_optimization_results["sharpe_ratio"]
        # Use Sharpe ratio to adjust Kelly fraction
        if sharpe_ratio > 1.0:  # Very good risk-adjusted return
            kelly_fraction = min(kelly_fraction * 1.2, 1.0)  # Increase position but not exceed 1
        elif sharpe_ratio < 0:  # Negative risk-adjusted return
            kelly_fraction = kelly_fraction * 0.5  # Halve position
    
    # Use Beta and volatility to further adjust position
    if "beta" in portfolio_optimization_results:
        beta_value = portfolio_optimization_results["beta"]
        # Beta above 1 means stock is more volatile than market, need to reduce position
        if beta_value > 1.2:
            kelly_fraction = kelly_fraction * (1 / beta_value)  # Reduce position proportionally
    
    # 10. If there is AI model data, further adjust position based on AI model confidence
    if ai_model_data:
        ai_confidence = ai_model_data.get("confidence", 0.5)
        ai_signal = ai_model_data.get("signal", "neutral")
        
        # Adjust position based on AI model signal and confidence
        if action == "buy" and ai_signal == "bullish":
            # AI is also bullish, increase position based on confidence
            kelly_fraction = min(kelly_fraction * (1 + ai_confidence * 0.2), 1.0)
        elif action == "sell" and ai_signal == "bearish":
            # AI is also bearish, increase position based on confidence
            kelly_fraction = min(kelly_fraction * (1 + ai_confidence * 0.2), 1.0)
        elif (action == "buy" and ai_signal == "bearish") or (action == "sell" and ai_signal == "bullish"):
            # AI signal is opposite to decision, reduce position
            kelly_fraction = kelly_fraction * (1 - ai_confidence * 0.3)
    
    # Risk adjustment
    risk_factor = 1 - (risk_score / 10)  # Higher risk score, lower risk factor
    
    # 11. Calculate suggested position
    # Kelly suggested position = portfolio value * Kelly fraction * risk conservative factor
    conservative_factor = 0.5  # Half Kelly, more conservative
    suggested_position_value = total_portfolio_value * kelly_fraction * conservative_factor * risk_factor
    
    # Ensure not exceeding maximum position specified by risk management
    suggested_position_value = min(suggested_position_value, max_position_size)
    
    # Get macro analysis results
    macro_signal = next((s for s in agent_signals if s.get("agent_name", "").lower() == "macro_analysis"), None)
    
    # Adjust capital allocation based on macro analysis
    macro_adjustment = 1.0  # Default no adjustment
    if macro_signal:
        # Increase capital allocation when macro environment is positive
        if macro_signal.get("signal", "").lower() == "bullish":
            macro_adjustment = 1.2
        # Reduce capital allocation when macro environment is negative
        elif macro_signal.get("signal", "").lower() == "bearish":
            macro_adjustment = 0.8
        
        # Apply macro adjustment to suggested position
        suggested_position_value = suggested_position_value * macro_adjustment
    
    # Convert to number of shares
    suggested_quantity = int(suggested_position_value / current_price) if current_price > 0 else 0
    
    # 12. Apply stop-loss and take-profit logic
    stop_loss_level = 0.05  # 5% stop loss
    take_profit_level = 0.15  # 15% take profit
    
    # Try to calculate current position profit/loss from price data
    position_profit_pct = 0  # Default assumption no profit/loss
    if not prices_df.empty and portfolio['stock'] > 0:
        # Simple assumption: use recent 20-day average price as position cost
        recent_period = min(20, len(prices_df))
        avg_price = prices_df['close'].iloc[-recent_period:].mean() if recent_period > 0 else current_price
        if avg_price > 0:
            position_profit_pct = (current_price - avg_price) / avg_price
    
    if position_profit_pct <= -stop_loss_level and portfolio['stock'] > 0:
        # Trigger stop loss
        action = "sell"
        new_quantity = portfolio['stock']  # Sell all
        reasoning = f"{reasoning}\nAdditionally, stop-loss mechanism triggered (loss exceeds {stop_loss_level*100:.0f}%), protecting portfolio."
    elif position_profit_pct >= take_profit_level and portfolio['stock'] > 0:
        # Trigger take profit - partial profit taking
        action = "sell"
        new_quantity = max(int(portfolio['stock'] * 0.5), 1)  # Sell half position
        reasoning = f"{reasoning}\nPartial profit taking strategy: profit reached {take_profit_level*100:.0f}%, reduce position to lock in gains."
    else:
        # Under normal circumstances, follow signals and risk management recommendations
        
        # First, ensure following risk management recommended trading direction
        if trading_action == "reduce" and action != "sell" and portfolio['stock'] > 0:
            action = "sell"
            new_quantity = max(int(portfolio['stock'] * 0.3), 1)  # Reduce position by 30%
            reasoning = f"{reasoning}\nReduce position based on risk management recommendation, lowering risk exposure."
        elif trading_action == "hold" and action == "buy":
            action = "hold"
            new_quantity = 0
            reasoning = f"{reasoning}\nRisk management indicates hold and wait, pause buying operations."
        else:
            new_quantity = suggested_quantity
    
    # 13. Apply portfolio constraints
    if action == "buy":
        # Constraints when buying
        max_affordable = int(portfolio['cash'] / current_price) if current_price > 0 else 0
        new_quantity = min(new_quantity, max_affordable)
        
        # Set minimum transaction size to avoid too small orders
        min_transaction = 100  # Minimum transaction 100 yuan
        min_shares = max(1, int(min_transaction / current_price)) if current_price > 0 else 1
        
        if new_quantity < min_shares:
            # Transaction size too small, change to hold
            action = "hold"
            new_quantity = 0
            reasoning = f"{reasoning}\nSuggested buy quantity too small (below minimum transaction size), maintain cash holding."
            
    elif action == "sell":
        # Constraints when selling
        new_quantity = min(new_quantity, portfolio['stock'])
        
        # If remaining position is too small, sell all
        if portfolio['stock'] - new_quantity < 10 and portfolio['stock'] > 0:
            new_quantity = portfolio['stock']
            reasoning = f"{reasoning}\nRemaining position too small, choose to sell all to optimize position."
            
    # 14. Smooth trading - avoid frequent small trades
    last_action = "hold"  # Assume last action was hold
    last_price = current_price  # Assume last price equals current price
    
    # If there is historical price data, try to get previous trading day price
    if not prices_df.empty and len(prices_df) > 1:
        last_price = prices_df['close'].iloc[-2]
    
    # If last trade and this suggestion are the same, but price change is less than threshold, reduce trade size or hold
    price_change_threshold = 0.03  # 3% price change threshold
    actual_price_change = abs(current_price - last_price) / last_price if last_price > 0 else 0
    
    if action == last_action and actual_price_change < price_change_threshold:
        if action == "buy":
            new_quantity = int(new_quantity * 0.5)  # Halve buy quantity
            if new_quantity == 0:
                action = "hold"
        elif action == "sell":
            new_quantity = int(new_quantity * 0.5)  # Halve sell quantity
            if new_quantity == 0:
                action = "hold"
    
    # Format market data
    formatted_market_data = format_market_data_summary(market_data)
    
    # 15. Add AI model analysis impact to decision reasoning
    if ai_model_data:
        ai_signal = ai_model_data.get("signal", "neutral")
        ai_confidence = ai_model_data.get("confidence", 0.5)
        
        if ai_signal == action.replace("hold", "neutral").replace("buy", "bullish").replace("sell", "bearish"):
            reasoning += f"\n\nAI model analysis supports {action} decision with confidence {ai_confidence:.2f}, enhancing decision confidence."
        else:
            reasoning += f"\n\nAI model analysis gives {ai_signal} signal, although different from decision direction, it has been considered and position appropriately adjusted."
        
        # If it's multi-asset analysis, add portfolio allocation recommendations
        if ai_model_data.get("multi_asset", False) and "portfolio_allocation" in ai_model_data:
            allocation = ai_model_data["portfolio_allocation"].get("allocation", {})
            if allocation:
                allocation_text = ", ".join([f"{ticker}: {weight*100:.1f}%" for ticker, weight in allocation.items()])
                reasoning += f"\n\nAI model recommended asset allocation: {allocation_text}"

    # Final decision integration
    optimized_decision = {
        "action": action,
        "quantity": new_quantity,
        "confidence": confidence,
        "agent_signals": agent_signals,
        "reasoning": reasoning,
        "portfolio_optimization": {
            "risk_score": risk_score,
            "kelly_fraction": kelly_fraction,
            "risk_factor": risk_factor,
            "max_position_size": max_position_size,
            "suggested_position_value": suggested_position_value,
            "total_portfolio_value": total_portfolio_value,
            "position_profit_pct": position_profit_pct,
            "macro_adjustment": macro_adjustment,
            "analytics": portfolio_optimization_results,
            "market_data": formatted_market_data,
        },
        "ai_model_integration": {
            "used": ai_model_data is not None,
            "signal": ai_model_data.get("signal", "neutral") if ai_model_data else None,
            "confidence": ai_model_data.get("confidence", 0) if ai_model_data else None,
            "impact_on_position": kelly_fraction / (confidence * 2 - 1) if (confidence * 2 - 1) > 0 else 1.0
        } if ai_model_data else None
    }

    return optimized_decision