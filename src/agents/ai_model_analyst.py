from langchain_core.messages import HumanMessage
import json

from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.utils.api_utils import agent_endpoint
from src.tools.api import prices_to_df
from model.dl import MLAgent
from model.rl import RLTradingAgent
from model.deap_factors import FactorAgent
from src.utils.logging_config import setup_logger

# Setup logger
logger = setup_logger('ai_model_analyst_agent')

@agent_endpoint("ai_model_analyst", "AI model analyst, running deep learning, reinforcement learning and genetic programming models for prediction")
def ai_model_analyst_agent(state: AgentState):
    """Run deep learning, reinforcement learning and genetic programming models to generate prediction signals, supporting multi-asset analysis"""
    show_workflow_status("AI Model Analyst")
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    
    # Get asset list
    tickers = data.get("tickers", [])
    if isinstance(tickers, str):
        tickers = [ticker.strip() for ticker in tickers.split(',')]
    
    # If no tickers provided but ticker is provided, use single ticker
    if not tickers and data.get("ticker"):
        tickers = [data["ticker"]]
    
    # If still no stock codes, return error
    if not tickers:
        logger.warning("No stock codes provided, unable to perform AI model analysis")
        message_content = {
            "signal": "neutral",
            "confidence": 0.5,
            "reasoning": "No stock codes provided, unable to perform AI model analysis"
        }
        message = HumanMessage(
            content=json.dumps(message_content),
            name="ai_model_analyst_agent",
        )
        return {
            "messages": [message],
            "data": data,
            "metadata": state["metadata"],
        }
    
    # Get data for all stocks
    all_stock_data = data.get("all_stock_data", {})
    
    # Single asset or multi-asset processing
    if len(tickers) == 1 or not all_stock_data:
        # Single asset processing
        ticker = tickers[0]
        prices = data.get("prices", [])
        prices_df = prices_to_df(prices)
        
        # If price data is empty, return default values
        if prices_df.empty:
            logger.warning(f"Price data is empty, unable to run AI model analysis")
            message_content = {
                "signal": "neutral",
                "confidence": 0.5,
                "reasoning": "Insufficient price data, unable to run AI model analysis"
            }
            message = HumanMessage(
                content=json.dumps(message_content),
                name="ai_model_analyst_agent",
            )
            return {
                "messages": [message],
                "data": data,
                "metadata": state["metadata"],
            }
        
        # Initialize AI models
        ml_signals = run_deep_learning_model(prices_df)
        rl_signals = run_reinforcement_learning_model(prices_df)
        factor_signals = run_genetic_programming_model(prices_df)
        
        # Combine model signals
        combined_signal = combine_ai_signals(ml_signals, rl_signals, factor_signals)
        
        # Generate analysis report
        message_content = {
            "multi_asset": False,
            "primary_ticker": ticker,
            "signal": combined_signal["signal"],
            "confidence": combined_signal["confidence"],
            "model_signals": {
                "deep_learning": ml_signals,
                "reinforcement_learning": rl_signals,
                "genetic_programming": factor_signals
            },
            "reasoning": combined_signal["reasoning"]
        }
    else:
        # Multi-asset processing
        # Run AI models for each asset
        all_asset_signals = {}
        
        for ticker in tickers:
            try:
                if ticker in all_stock_data:
                    # Get current asset's price data
                    asset_prices = all_stock_data[ticker].get("prices", [])
                    asset_prices_df = prices_to_df(asset_prices)
                    
                    if not asset_prices_df.empty:
                        # Run various models
                        ml_signals = run_deep_learning_model(asset_prices_df)
                        rl_signals = run_reinforcement_learning_model(asset_prices_df)
                        factor_signals = run_genetic_programming_model(asset_prices_df)
                        
                        # Combine signals
                        combined_signal = combine_ai_signals(ml_signals, rl_signals, factor_signals)
                        
                        # Save this asset's signals
                        all_asset_signals[ticker] = {
                            "signal": combined_signal["signal"],
                            "confidence": combined_signal["confidence"],
                            "model_signals": {
                                "deep_learning": ml_signals,
                                "reinforcement_learning": rl_signals,
                                "genetic_programming": factor_signals
                            },
                            "reasoning": combined_signal["reasoning"]
                        }
                    else:
                        logger.warning(f"Asset {ticker} price data is empty")
                        all_asset_signals[ticker] = {
                            "signal": "neutral",
                            "confidence": 0.5,
                            "model_signals": {},
                            "reasoning": "Insufficient price data to run AI model analysis"
                        }
                else:
                    logger.warning(f"Asset {ticker} data not found")
                    all_asset_signals[ticker] = {
                        "signal": "neutral",
                        "confidence": 0.5,
                        "model_signals": {},
                        "reasoning": "Asset data not found"
                    }
            except Exception as e:
                logger.error(f"Error processing asset {ticker}: {e}")
                all_asset_signals[ticker] = {
                    "signal": "neutral",
                    "confidence": 0.5,
                    "model_signals": {},
                    "reasoning": f"Error processing this asset: {str(e)}"
                }
        
        # Analyze overall portfolio AI model signals
        portfolio_signal = analyze_portfolio_ai_signals(all_asset_signals)
        
        # Portfolio optimization recommendations
        portfolio_allocation = optimize_portfolio_based_on_ai(all_asset_signals)
        
        # Generate analysis report
        message_content = {
            "multi_asset": True,
            "primary_ticker": tickers[0],
            "tickers": tickers,
            "signal": portfolio_signal["signal"],
            "confidence": portfolio_signal["confidence"],
            "asset_signals": all_asset_signals,
            "portfolio_allocation": portfolio_allocation,
            "reasoning": portfolio_signal["reasoning"]
        }
    
    # Create message
    message = HumanMessage(
        content=json.dumps(message_content),
        name="ai_model_analyst_agent",
    )
    
    # Show reasoning process
    if show_reasoning:
        show_agent_reasoning(message_content, "AI Model Analyst")
        # Save reasoning information to metadata for API use
        state["metadata"]["agent_reasoning"] = message_content
    
    show_workflow_status("AI Model Analyst", "completed")
    return {
        "messages": [message],
        "data": {
            **data,
            "ai_analysis": message_content
        },
        "metadata": state["metadata"],
    }

def run_deep_learning_model(prices_df):
    """Run deep learning model to generate trading signals"""
    try:
        # Try to load trained model
        ml_agent = MLAgent(model_dir='models')
        ml_agent.load_models()
        
        # Generate trading signals
        signals = ml_agent.generate_signals(prices_df)
        
        # If unable to get signals, use default values
        if not signals or "signal" not in signals:
            return {
                "signal": "neutral", 
                "confidence": 0.5,
                "reasoning": "Deep learning model failed to generate valid signals"
            }
        
        # Return model-generated signals
        return {
            "signal": signals["signal"],
            "confidence": signals.get("confidence", 0.5),
            "reasoning": signals.get("reasoning", "Deep learning model based on LSTM and Random Forest algorithm prediction")
        }
    except Exception as e:
        logger.error(f"Error running deep learning model: {e}")
        return {
            "signal": "neutral", 
            "confidence": 0.5,
            "reasoning": f"Deep learning model execution error: {str(e)}"
        }

def run_reinforcement_learning_model(prices_df):
    """Run reinforcement learning model to generate trading signals"""
    try:
        # Try to load trained model
        rl_agent = RLTradingAgent(model_dir='models')
        rl_agent.load_model()
        
        # Generate trading signals
        signals = rl_agent.generate_signals(prices_df)
        
        # If unable to get signals, use default values
        if not signals or "signal" not in signals:
            return {
                "signal": "neutral", 
                "confidence": 0.5,
                "reasoning": "Reinforcement learning model failed to generate valid signals"
            }
        
        # Return model-generated signals
        return {
            "signal": signals["signal"],
            "confidence": signals.get("confidence", 0.5),
            "reasoning": signals.get("reasoning", "Reinforcement learning model based on PPO algorithm optimizing trading decisions")
        }
    except Exception as e:
        logger.error(f"Error running reinforcement learning model: {e}")
        return {
            "signal": "neutral", 
            "confidence": 0.5,
            "reasoning": f"Reinforcement learning model execution error: {str(e)}"
        }

def run_genetic_programming_model(prices_df):
    """Run genetic programming factor model to generate trading signals"""
    try:
        # Try to load trained model
        factor_agent = FactorAgent(model_dir='factors')
        factor_agent.load_factors()
        
        # Generate trading signals
        signals = factor_agent.generate_signals(prices_df)
        
        # If unable to get signals, use default values
        if not signals or "signal" not in signals:
            return {
                "signal": "neutral", 
                "confidence": 0.5,
                "reasoning": "Genetic programming factor model failed to generate valid signals"
            }
        
        # Return model-generated signals
        return {
            "signal": signals["signal"],
            "confidence": signals.get("confidence", 0.5),
            "reasoning": signals.get("reasoning", "Genetic programming factor model analyzes market through automated factor mining")
        }
    except Exception as e:
        logger.error(f"Error running genetic programming factor model: {e}")
        return {
            "signal": "neutral", 
            "confidence": 0.5,
            "reasoning": f"Genetic programming factor model execution error: {str(e)}"
        }

def combine_ai_signals(ml_signals, rl_signals, factor_signals):
    """Combine AI model signals to generate final signal"""
    # Convert signals to numerical values
    signal_values = {
        'bullish': 1,
        'neutral': 0,
        'bearish': -1
    }
    
    # Model weights
    weights = {
        'deep_learning': 0.35,  # LSTM and Random Forest combined model
        'reinforcement_learning': 0.35,  # PPO reinforcement learning model
        'genetic_programming': 0.30  # Genetic programming factor model
    }
    
    # Calculate weighted scores
    ml_score = signal_values.get(ml_signals["signal"], 0) * ml_signals["confidence"] * weights["deep_learning"]
    rl_score = signal_values.get(rl_signals["signal"], 0) * rl_signals["confidence"] * weights["reinforcement_learning"]
    factor_score = signal_values.get(factor_signals["signal"], 0) * factor_signals["confidence"] * weights["genetic_programming"]
    
    total_score = ml_score + rl_score + factor_score
    
    # Determine signal based on score
    if total_score > 0.15:
        signal = "bullish"
    elif total_score < -0.15:
        signal = "bearish"
    else:
        signal = "neutral"
    
    # Calculate confidence
    confidence = min(0.5 + abs(total_score), 0.9)  # Limit to 0.5-0.9 range
    
    # Create signal consistency description
    signals = [ml_signals["signal"], rl_signals["signal"], factor_signals["signal"]]
    signal_counts = {
        "bullish": signals.count("bullish"),
        "bearish": signals.count("bearish"),
        "neutral": signals.count("neutral")
    }
    
    # Generate reasoning text
    reasoning = f"AI model comprehensive analysis: Deep Learning({ml_signals['signal']}, {ml_signals['confidence']:.2f}), "
    reasoning += f"Reinforcement Learning({rl_signals['signal']}, {rl_signals['confidence']:.2f}), "
    reasoning += f"Genetic Programming({factor_signals['signal']}, {factor_signals['confidence']:.2f}). "
    
    # Add consistency analysis
    max_signal = max(signal_counts.items(), key=lambda x: x[1])
    if max_signal[1] >= 2:
        reasoning += f"{max_signal[1]} models consistently predict {max_signal[0]} signal, enhancing prediction credibility."
    else:
        reasoning += f"Model predictions are inconsistent, reducing overall confidence."
    
    # Add specific reasoning for each model
    reasoning += f"\n\nSpecific model reasoning:\n- Deep Learning: {ml_signals['reasoning']}\n"
    reasoning += f"- Reinforcement Learning: {rl_signals['reasoning']}\n"
    reasoning += f"- Genetic Programming: {factor_signals['reasoning']}"
    
    return {
        "signal": signal,
        "confidence": confidence,
        "weighted_score": total_score,
        "signal_consistency": max_signal[1] / 3,  # Consistency ratio
        "reasoning": reasoning
    }

def analyze_portfolio_ai_signals(all_asset_signals):
    """
    Analyze overall portfolio signals based on AI model signals from multiple assets
    
    Args:
        all_asset_signals: AI model signals for all assets
        
    Returns:
        dict: Overall portfolio signal
    """
    if not all_asset_signals:
        return {
            "signal": "neutral",
            "confidence": 0.5,
            "reasoning": "No asset AI model signals provided"
        }
    
    # Count various signal types
    signal_counts = {"bullish": 0, "bearish": 0, "neutral": 0}
    total_confidence = 0
    
    # Weight is signal confidence
    weighted_signals = 0
    total_weights = 0
    
    for ticker, signals in all_asset_signals.items():
        signal = signals.get("signal", "neutral")
        confidence = signals.get("confidence", 0.5)
        
        # Update counts
        signal_counts[signal] += 1
        total_confidence += confidence
        
        # Calculate weighted signals
        signal_value = 1 if signal == "bullish" else (-1 if signal == "bearish" else 0)
        weighted_signals += signal_value * confidence
        total_weights += confidence
    
    # Calculate overall signal
    if total_weights > 0:
        avg_signal = weighted_signals / total_weights
    else:
        avg_signal = 0
    
    # Determine final signal
    if avg_signal > 0.2:
        signal = "bullish"
    elif avg_signal < -0.2:
        signal = "bearish"
    else:
        signal = "neutral"
    
    # Calculate overall confidence
    avg_confidence = total_confidence / len(all_asset_signals) if all_asset_signals else 0.5
    
    # Calculate signal consistency level
    max_signal_count = max(signal_counts.values())
    consistency = max_signal_count / len(all_asset_signals) if all_asset_signals else 0
    
    # Adjust confidence
    adjusted_confidence = avg_confidence * (0.5 + 0.5 * consistency)
    
    # Generate reasoning explanation
    reasoning = f"Portfolio AI analysis: Analyzed {len(all_asset_signals)} assets, with {signal_counts['bullish']} bullish, "
    reasoning += f"{signal_counts['bearish']} bearish, and {signal_counts['neutral']} neutral.\n"
    reasoning += f"Signal consistency: {consistency:.2f}, average confidence: {avg_confidence:.2f}.\n\n"
    
    # Add assets with significant signals
    strong_bullish = []
    strong_bearish = []
    
    for ticker, signals in all_asset_signals.items():
        if signals.get("signal") == "bullish" and signals.get("confidence", 0) > 0.7:
            strong_bullish.append(ticker)
        elif signals.get("signal") == "bearish" and signals.get("confidence", 0) > 0.7:
            strong_bearish.append(ticker)
    
    if strong_bullish:
        reasoning += f"Strong bullish assets: {', '.join(strong_bullish)}\n"
    if strong_bearish:
        reasoning += f"Strong bearish assets: {', '.join(strong_bearish)}\n"
    
    return {
        "signal": signal,
        "confidence": adjusted_confidence,
        "weighted_score": avg_signal,
        "signal_consistency": consistency,
        "signal_counts": signal_counts,
        "reasoning": reasoning
    }

def optimize_portfolio_based_on_ai(all_asset_signals):
    """
    Optimize portfolio allocation based on AI model signals
    
    Args:
        all_asset_signals: AI model signals for all assets
        
    Returns:
        dict: Optimized asset allocation recommendations
    """
    if not all_asset_signals:
        return {}
    
    # Calculate weights based on AI signals
    signal_scores = {}
    total_positive_score = 0
    
    for ticker, signals in all_asset_signals.items():
        signal = signals.get("signal", "neutral")
        confidence = signals.get("confidence", 0.5)
        
        # Calculate signal score
        if signal == "bullish":
            score = confidence
        elif signal == "bearish":
            score = -confidence
        else:
            score = 0
        
        signal_scores[ticker] = score
        
        # Calculate total positive score (for subsequent weight calculation)
        if score > 0:
            total_positive_score += score
    
    # If no positive scores, use equal weights
    if total_positive_score <= 0:
        weights = {ticker: 1.0 / len(all_asset_signals) for ticker in all_asset_signals}
    else:
        # Calculate weights (only consider assets with positive signals)
        weights = {}
        for ticker, score in signal_scores.items():
            if score > 0:
                weights[ticker] = score / total_positive_score
            else:
                weights[ticker] = 0
    
    # Output asset allocation recommendations
    allocation = {}
    recommended_assets = []
    
    for ticker, weight in weights.items():
        if weight > 0:
            allocation[ticker] = round(weight, 4)
            recommended_assets.append(ticker)
    
    return {
        "allocation": allocation,
        "recommended_assets": recommended_assets,
        "reasoning": "Calculate optimal asset weights based on AI model signal strength and confidence"
    }