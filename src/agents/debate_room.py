import ast
import json
import logging

import pandas as pd
from langchain_core.messages import HumanMessage

from src.agents.regime_detector import (
    AdvancedRegimeDetector,
    adaptive_signal_aggregation,
)
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.tools.api import prices_to_df
from src.tools.openrouter_config import get_chat_completion
from src.utils.api_utils import agent_endpoint, log_llm_interaction
from src.utils.logging_config import setup_logger

# Get logger
logger = setup_logger("debate_room")


@agent_endpoint(
    "debate_room",
    "Debate room, analyzes both bullish and bearish perspectives to reach balanced investment conclusions",
)
def debate_room_agent(state: AgentState):
    """
    Enhanced debate room with regime-aware signal aggregation
    Based on 2024-2025 research: FLAG-Trader, FINSABER, Lopez-Lira frameworks
    """
    try:
        show_workflow_status("Debate Room")
        show_reasoning = state["metadata"]["show_reasoning"]
        logger.info(
            "Starting analysis of researcher perspectives and conducting debate..."
        )

        # Initialize advanced regime detector
        regime_detector = AdvancedRegimeDetector()

        # Get price data for regime analysis
        data = state["data"]
        prices = data["prices"]
        prices_df = prices_to_df(prices)

        # Extract regime features and perform analysis
        regime_features = regime_detector.extract_regime_features(prices_df)
        regime_model_results = regime_detector.fit_regime_model(regime_features)
        current_regime = regime_detector.predict_current_regime(regime_features)

        logger.info(
            f"Detected market regime: {current_regime.get('regime_name', 'unknown')} (confidence: {current_regime.get('confidence', 0):.2f})"
        )

        # Collect all agent signals - forward compatibility design (add defensive checks)
        agent_signals = {}
        researcher_messages = {}

        for msg in state["messages"]:
            # Add defensive checks to ensure msg and msg.name are not None
            if msg is None:
                continue
            if not hasattr(msg, "name") or msg.name is None:
                continue

            # Collect various agent signals
            if msg.name.endswith("_agent"):
                try:
                    if hasattr(msg, "content") and msg.content is not None:
                        content = (
                            json.loads(msg.content)
                            if isinstance(msg.content, str)
                            else msg.content
                        )

                        # Map agent names to signal types
                        agent_type_mapping = {
                            "technical_analyst_agent": "technical",
                            "fundamentals_agent": "fundamental",
                            "sentiment_agent": "sentiment",
                            "valuation_agent": "valuation",
                            "ai_model_analyst_agent": "ai_model",
                            "macro_analyst_agent": "macro",
                        }

                        if msg.name in agent_type_mapping:
                            signal_type = agent_type_mapping[msg.name]
                            # Ensure proper handling of signal values
                            raw_signal = content.get("signal", "neutral")
                            confidence = content.get("confidence", 0.5)

                            agent_signals[signal_type] = {
                                "signal": raw_signal,
                                "confidence": confidence,
                                "raw_data": content,
                            }
                            logger.debug(
                                f"Collected {signal_type} signal: {raw_signal} (confidence: {confidence})"
                            )

                        # Also maintain original researcher logic
                        if msg.name.startswith("researcher_"):
                            researcher_messages[msg.name] = msg
                            logger.debug(
                                f"Collected researcher information: {msg.name}"
                            )

                except (json.JSONDecodeError, TypeError, AttributeError) as e:
                    logger.warning(f"Unable to parse {msg.name} message content: {e}")
                    continue

        # Use new adaptive signal aggregation system
        if agent_signals and current_regime.get("regime_name") != "unknown":
            aggregated_result = adaptive_signal_aggregation(
                signals=agent_signals,
                regime_info=current_regime,
                confidence_threshold=0.6,
            )

            final_signal = aggregated_result["aggregated_signal"]
            final_confidence = aggregated_result["aggregated_confidence"]

            logger.info(
                f"Adaptive signal aggregation result: {final_signal:.3f} (confidence: {final_confidence:.3f})"
            )
            logger.debug(
                f"Original signal: {aggregated_result.get('original_signal', 'N/A'):.3f}, "
                f"Attenuation applied: {aggregated_result.get('attenuation_applied', False)}, "
                f"Dynamic threshold: {aggregated_result.get('dynamic_threshold', 'N/A')}"
            )

            # Record specific contributions of each signal
            for signal_type, contribution in aggregated_result.get(
                "signal_contributions", {}
            ).items():
                logger.debug(
                    f"{signal_type} contribution: signal={contribution['signal']}, "
                    f"confidence={contribution['confidence']:.3f}, "
                    f"weight={contribution['weight']:.3f}, "
                    f"contribution_value={contribution['contribution']:.4f}"
                )

            # Create enhanced analysis report
            enhanced_analysis = {
                "signal": final_signal,
                "confidence": final_confidence,
                "aggregation_method": "regime_aware_adaptive",
                "market_regime": current_regime,
                "regime_adjusted_weights": aggregated_result["regime_adjusted_weights"],
                "signal_contributions": aggregated_result["signal_contributions"],
                "dynamic_threshold": aggregated_result["dynamic_threshold"],
                "regime_model_performance": {
                    "model_score": regime_model_results.get("model_score", 0),
                    "regime_characteristics": regime_model_results.get(
                        "regime_characteristics", {}
                    ),
                },
            }

        else:
            # Fallback to original logic (maintain backward compatibility)
            logger.info(
                "Using traditional debate logic (insufficient signals or regime detection failed)"
            )
            enhanced_analysis = _traditional_debate_logic(
                state, researcher_messages, logger
            )

        # Ensure at least bullish and bearish researchers (maintain original validation logic)
        if (
            "researcher_bull_agent" not in researcher_messages
            or "researcher_bear_agent" not in researcher_messages
        ):
            logger.error(
                "Missing necessary researcher data: researcher_bull_agent or researcher_bear_agent"
            )
            # If no researcher data but other signals exist, can still continue
            if not agent_signals:
                raise ValueError(
                    "Missing required researcher_bull_agent or researcher_bear_agent messages"
                )

        # Process researcher data (maintain original logic for LLM analysis)
        researcher_data = {}
        for name, msg in researcher_messages.items():
            if not hasattr(msg, "content") or msg.content is None:
                logger.warning(f"Researcher {name} message content is empty")
                continue
            try:
                data_content = json.loads(msg.content)
                logger.debug(f"Successfully parsed {name} JSON content")
            except (json.JSONDecodeError, TypeError):
                try:
                    data_content = ast.literal_eval(msg.content)
                    logger.debug(f"Parsed {name} content through ast.literal_eval")
                except (ValueError, SyntaxError, TypeError):
                    logger.warning(f"Unable to parse {name} message content, skipped")
                    continue
            researcher_data[name] = data_content

        # If researcher data exists, perform LLM enhanced analysis
        if len(researcher_data) >= 2:
            llm_enhanced_analysis = _get_llm_enhanced_analysis(
                researcher_data, agent_signals, current_regime, state, logger
            )

            # Fuse adaptive aggregation results and LLM analysis
            if "signal" in enhanced_analysis and "llm_score" in llm_enhanced_analysis:
                # Use weighted average to fuse results from both methods
                regime_confidence = current_regime.get("confidence", 0.5)
                adaptive_weight = 0.7 if regime_confidence > 0.6 else 0.5
                llm_weight = 1 - adaptive_weight

                # Ensure signal is in numeric format for fusion
                current_signal = enhanced_analysis["signal"]
                if isinstance(current_signal, str):
                    # Convert string signal to numeric
                    signal_mapping = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}
                    numeric_signal = signal_mapping.get(current_signal.lower(), 0.0)
                else:
                    numeric_signal = float(current_signal)

                # Perform fusion
                fused_signal = (
                    adaptive_weight * numeric_signal
                    + llm_weight * llm_enhanced_analysis["llm_score"]
                )

                enhanced_analysis["signal"] = fused_signal
                enhanced_analysis["llm_analysis"] = llm_enhanced_analysis
                enhanced_analysis["fusion_weights"] = {
                    "adaptive_aggregation": adaptive_weight,
                    "llm_analysis": llm_weight,
                }
                enhanced_analysis["original_signal"] = (
                    current_signal  # Keep original signal for debugging
                )

        # Convert numeric signal to string signal (for compatibility with risk manager)
        enhanced_analysis = _convert_numeric_signal_to_string(enhanced_analysis)

        # Create final message
        message = HumanMessage(
            content=json.dumps(enhanced_analysis),
            name="debate_room_agent",
        )

        if show_reasoning:
            show_agent_reasoning(enhanced_analysis, "Enhanced Debate Room")
            state["metadata"]["agent_reasoning"] = enhanced_analysis

        show_workflow_status("Debate Room", "completed")
        return {
            "messages": [message],
            "data": data,
            "metadata": state["metadata"],
        }

    except Exception as e:
        logger.error(f"Error occurred during debate room processing: {e}")
        # Return default neutral result
        default_analysis = {
            "signal": "neutral",
            "confidence": 0.3,
            "error": str(e),
            "aggregation_method": "error_fallback",
        }

        message = HumanMessage(
            content=json.dumps(default_analysis),
            name="debate_room_agent",
        )

        return {
            "messages": [message],
            "data": state["data"],
            "metadata": state["metadata"],
        }


def _traditional_debate_logic(
    state: AgentState, researcher_messages: dict, logger
) -> dict:
    """Asynchronous version of traditional debate logic"""
    # Implement original simple averaging logic as fallback
    if len(researcher_messages) < 2:
        return {
            "signal": "neutral",
            "confidence": 0.3,
            "aggregation_method": "insufficient_data",
        }

    def _parse_confidence(conf_value):
        """Parse confidence value, supporting both string and numeric formats"""
        if isinstance(conf_value, str):
            if conf_value.endswith("%"):
                try:
                    return float(conf_value[:-1]) / 100.0
                except ValueError:
                    return 0.5
            try:
                return float(conf_value)
            except ValueError:
                return 0.5
        elif isinstance(conf_value, (int, float)):
            if conf_value > 1.0:
                return conf_value / 100.0
            return float(conf_value)
        else:
            return 0.5

    # Convert researcher perspectives to numeric signals
    perspective_values = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}

    # Simple average of researcher signals
    total_signal = 0
    total_confidence = 0
    count = 0

    for name, msg in researcher_messages.items():
        try:
            data = json.loads(msg.content)
            # Researchers use perspective instead of signal
            perspective = data.get("perspective", "neutral")
            raw_confidence = data.get("confidence", 0.5)

            # Parse confidence
            confidence = _parse_confidence(raw_confidence)

            # Convert perspective to numeric
            numeric_signal = perspective_values.get(perspective, 0.0)

            total_signal += numeric_signal * confidence
            total_confidence += confidence
            count += 1
        except Exception as e:
            logger.warning(f"Error parsing researcher {name} data: {e}")
            continue

    if count > 0:
        avg_signal = total_signal / total_confidence if total_confidence > 0 else 0
        avg_confidence = total_confidence / count if count > 0 else 0.3
    else:
        avg_signal = 0
        avg_confidence = 0.3

    # Convert numeric signal to string signal
    if avg_signal > 0.1:  # Reduced from 0.2 to 0.1
        signal_str = "bullish"
    elif avg_signal < -0.1:  # Reduced from -0.2 to -0.1
        signal_str = "bearish"
    else:
        signal_str = "neutral"

    return {
        "signal": signal_str,
        "confidence": avg_confidence,
        "aggregation_method": "traditional_average",
    }


def _get_llm_enhanced_analysis(
    researcher_data: dict,
    agent_signals: dict,
    current_regime: dict,
    state: AgentState,
    logger,
) -> dict:
    """Get LLM enhanced analysis"""

    def _parse_confidence_for_display(conf_value):
        """Parse confidence value for display"""
        if isinstance(conf_value, str):
            if conf_value.endswith("%"):
                try:
                    return float(conf_value[:-1]) / 100.0
                except ValueError:
                    return 0.5
            try:
                return float(conf_value)
            except ValueError:
                return 0.5
        elif isinstance(conf_value, (int, float)):
            if conf_value > 1.0:
                return conf_value / 100.0
            return float(conf_value)
        else:
            return 0.5

    # Build prompt to send to LLM (maintain original logic but enhanced)
    regime_confidence = _parse_confidence_for_display(
        current_regime.get("confidence", 0)
    )

    llm_prompt = f"""
You are a professional financial analyst. Analyze the following investment research and provide your third-party analysis.

Current Market Regime: {current_regime.get('regime_name', 'unknown')} (Confidence: {regime_confidence:.2f})

RESEARCH PERSPECTIVES:
"""

    # Add researcher perspectives
    for name, data in researcher_data.items():
        perspective = name.replace("researcher_", "").replace("_agent", "").upper()
        researcher_confidence = _parse_confidence_for_display(data.get("confidence", 0))
        llm_prompt += (
            f"\n{perspective} VIEW (Confidence: {researcher_confidence:.2f}):\n"
        )
        for point in data.get("thesis_points", []):
            llm_prompt += f"- {point}\n"

    # Add quantitative signals summary
    if agent_signals:
        llm_prompt += f"\nQUANTITATIVE SIGNALS:\n"
        for signal_type, signal_data in agent_signals.items():
            signal_value = signal_data.get("signal", "neutral")
            confidence_value = signal_data.get("confidence", 0)

            # Handle signal value display
            if isinstance(signal_value, str):
                signal_display = signal_value
            else:
                signal_display = f"{signal_value:.2f}"

            # Handle confidence display
            parsed_confidence = _parse_confidence_for_display(confidence_value)
            llm_prompt += f"- {signal_type.title()}: {signal_display} (Confidence: {parsed_confidence:.2f})\n"

    llm_prompt += f"""
MARKET REGIME CONTEXT:
- Detected regime: {current_regime.get('regime_name', 'unknown')}
- Regime confidence: {regime_confidence:.2f}

Please provide your analysis in the following JSON format:
{{
    "analysis": "Your detailed analysis evaluating the strengths and weaknesses of each perspective",
    "score": 0.5,  // Your score from -1.0 (extremely bearish) to 1.0 (extremely bullish), 0 = neutral
    "reasoning": "Brief reasoning for your score",
    "regime_considerations": "How the current market regime affects your analysis",
    "macro_factors": ["List 1-3 most important macro factors"]
}}

Ensure your response is valid JSON format and includes all fields above. Respond in English only.
"""

    try:
        logger.info("Starting LLM call to get enhanced analysis...")
        messages = [
            {
                "role": "system",
                "content": "You are a professional financial analyst. Provide analysis in English only.",
            },
            {"role": "user", "content": llm_prompt},
        ]

        llm_response = log_llm_interaction(state)(
            lambda: get_chat_completion(messages)
        )()

        if llm_response:
            # Parse LLM returned JSON
            json_start = llm_response.find("{")
            json_end = llm_response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = llm_response[json_start:json_end]
                llm_analysis = json.loads(json_str)
                llm_score = float(llm_analysis.get("score", 0))
                llm_score = max(min(llm_score, 1.0), -1.0)  # Ensure within valid range

                logger.info(f"LLM enhanced analysis completed, score: {llm_score}")
                return {
                    "llm_score": llm_score,
                    "llm_analysis": llm_analysis,
                    "llm_response": llm_response,
                }
    except Exception as e:
        logger.error(f"LLM enhanced analysis failed: {e}")

    return {"llm_score": 0, "error": "LLM analysis failed"}


def _convert_numeric_signal_to_string(analysis: dict) -> dict:
    """Convert numeric signal to string signal"""
    if "signal" in analysis and isinstance(analysis["signal"], (int, float)):
        numeric_signal = analysis["signal"]
        original_signal = numeric_signal

        # Lower conversion threshold to make system more sensitive to smaller signals
        if numeric_signal > 0.1:  # Reduced from 0.2 to 0.1
            analysis["signal"] = "bullish"
        elif numeric_signal < -0.1:  # Reduced from -0.2 to -0.1
            analysis["signal"] = "bearish"
        else:
            analysis["signal"] = "neutral"

        # Add conversion log for debugging
        logger = logging.getLogger(__name__)
        logger.info(
            f"Signal conversion: {original_signal:.4f} -> {analysis['signal']} "
            f"(threshold: ±0.1)"
        )  # Changed to INFO level for visibility in logs

    return analysis
