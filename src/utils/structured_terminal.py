import json
from typing import Any, Dict, List

from src.utils.logging_config import setup_logger

# Setup logger
logger = setup_logger("structured_terminal")

# Formatting symbols
SYMBOLS = {
    "border": "═",
    "header_left": "╔",
    "header_right": "╗",
    "footer_left": "╚",
    "footer_right": "╝",
    "separator": "─",
    "vertical": "║",
    "tree_branch": "├─",
    "tree_last": "└─",
    "section_prefix": "● ",
    "bullet": "• ",
}

# Status icons
STATUS_ICONS = {
    "bearish": "📉",
    "bullish": "📈",
    "neutral": "◽",
    "hold": "⏸️",
    "buy": "🛒",
    "sell": "💰",
    "completed": "✅",
    "in_progress": "🔄",
    "error": "❌",
    "warning": "⚠️",
}

# Agent icon and name mapping
AGENT_MAP = {
    "market_data_agent": {"icon": "📊", "name": "Market Data Analysis"},
    "technical_analyst_agent": {"icon": "📈", "name": "Technical Analysis"},
    "fundamentals_agent": {"icon": "📝", "name": "Fundamental Analysis"},
    "sentiment_agent": {"icon": "🔍", "name": "Sentiment Analysis"},
    "valuation_agent": {"icon": "💰", "name": "Valuation Analysis"},
    "researcher_bull_agent": {"icon": "🐂", "name": "Bullish Research"},
    "researcher_bear_agent": {"icon": "🐻", "name": "Bearish Research"},
    "debate_room_agent": {"icon": "🗣️", "name": "Debate Room Analysis"},
    "risk_management_agent": {"icon": "⚠️", "name": "Risk Management"},
    "macro_analyst_agent": {"icon": "🌍", "name": "Macro Analysis"},
    "portfolio_management_agent": {"icon": "📂", "name": "Portfolio Management"},
}

# Agent display order
AGENT_ORDER = [
    "market_data_agent",
    "technical_analyst_agent",
    "fundamentals_agent",
    "sentiment_agent",
    "valuation_agent",
    "researcher_bull_agent",
    "researcher_bear_agent",
    "debate_room_agent",
    "risk_management_agent",
    "macro_analyst_agent",
    "portfolio_management_agent",
]


class StructuredTerminalOutput:
    """Structured terminal output class"""

    def __init__(self):
        """Initialize"""
        self.data = {}
        self.metadata = {}

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata"""
        self.metadata[key] = value

    def add_agent_data(self, agent_name: str, data: Any) -> None:
        """Add agent data"""
        self.data[agent_name] = data

    def _format_value(self, value: Any) -> str:
        """Format single value"""
        if isinstance(value, bool):
            return "✅" if value else "❌"
        elif isinstance(value, (int, float)):
            # Special handling for large numbers
            if value > 1000000:
                if value > 1000000000:  # Above billion
                    return f"${value/1000000000:.2f}B"
                else:  # Million to billion
                    return f"${value/1000000:.2f}M"
            # Format numbers with more than 5 decimal places
            elif isinstance(value, float) and abs(value) < 0.00001:
                return f"{value:.5f}"
            elif isinstance(value, float):
                return f"{value:.4f}"
            return str(value)
        elif value is None:
            return "N/A"
        else:
            return str(value)

    def _format_dict_as_tree(
        self, data: Dict[str, Any], indent: int = 0, max_str_len: int = 500
    ) -> List[str]:
        """Format dictionary as tree structure, limiting string length"""
        result = []
        items = list(data.items())

        for i, (key, value) in enumerate(items):
            is_last = i == len(items) - 1
            prefix = SYMBOLS["tree_last"] if is_last else SYMBOLS["tree_branch"]
            indent_str = "  " * indent

            # Format current value
            formatted_value = self._format_value(value)

            # Special handling for market_data and some large data structures
            if (
                key in ["market_returns", "stock_returns"]
                and isinstance(value, str)
                and len(value) > max_str_len
            ):
                result.append(f"{indent_str}{prefix} {key}: [Data too long, omitted]")
                continue

            # Add special handling for large values
            if (
                "price" in key.lower()
                and isinstance(value, (int, float))
                and value > 1000000
            ):
                if value > 1000000000:  # Above billion
                    formatted_value = f"${value/1000000000:.2f}B"
                else:  # Million to billion
                    formatted_value = f"${value/1000000:.2f}M"
                result.append(f"{indent_str}{prefix} {key}: {formatted_value}")
                continue

            # Handle 0.0 values
            if isinstance(value, (int, float)) and value == 0.0:
                # Check if it's a scenario where 0 values should be displayed (like count, quantity, etc.)
                if any(
                    keyword in key.lower()
                    for keyword in ["count", "quantity", "number", "index"]
                ):
                    result.append(f"{indent_str}{prefix} {key}: {formatted_value}")
                else:
                    # If it's in stress_test or other scenarios that default to 0, can choose not to display
                    if (
                        "stress_test" not in key.lower()
                        and "potential_loss" not in key.lower()
                    ):
                        result.append(f"{indent_str}{prefix} {key}: {formatted_value}")
                continue

            if isinstance(value, dict) and value:
                result.append(f"{indent_str}{prefix} {key}:")
                result.extend(self._format_dict_as_tree(value, indent + 1, max_str_len))
            elif isinstance(value, list) and value:
                result.append(f"{indent_str}{prefix} {key}:")
                for j, item in enumerate(value):
                    sub_is_last = j == len(value) - 1
                    sub_prefix = (
                        SYMBOLS["tree_last"] if sub_is_last else SYMBOLS["tree_branch"]
                    )
                    if isinstance(item, dict):
                        result.append(f"{indent_str}  {sub_prefix} Agent {j+1}:")
                        result.extend(
                            [
                                "  " + line
                                for line in self._format_dict_as_tree(
                                    item, indent + 2, max_str_len
                                )
                            ]
                        )
                    else:
                        # Truncate overly long list items
                        item_str = str(item)
                        if len(item_str) > max_str_len:
                            item_str = item_str[:max_str_len] + "..."
                        result.append(f"{indent_str}  {sub_prefix} {item_str}")
            else:
                # Truncate overly long strings
                if (
                    isinstance(formatted_value, str)
                    and len(formatted_value) > max_str_len
                ):
                    formatted_value = formatted_value[:max_str_len] + "..."
                result.append(f"{indent_str}{prefix} {key}: {formatted_value}")

        return result

    def _format_market_data_section(self, data: Dict[str, Any]) -> List[str]:
        """Format market data section as concise summary"""
        result = []
        width = 80

        # Create title
        title = "📊 Market Data Summary"
        result.append(
            f"{SYMBOLS['header_left']}{SYMBOLS['border'] * ((width - len(title) - 2) // 2)} {title} {SYMBOLS['border'] * ((width - len(title) - 2) // 2)}{SYMBOLS['header_right']}"
        )

        # Add main data
        if data.get("ticker"):
            result.append(f"{SYMBOLS['vertical']} Stock Code: {data.get('ticker')}")

        if data.get("start_date") and data.get("end_date"):
            result.append(
                f"{SYMBOLS['vertical']} Analysis Period: {data.get('start_date')} to {data.get('end_date')}"
            )

        # Price summary
        prices = data.get("prices", [])
        if prices:
            # Calculate price statistics
            if len(prices) > 0:
                latest_price = prices[-1].get("close", 0)
                avg_price = sum(p.get("close", 0) for p in prices) / len(prices)
                max_price = max(p.get("high", 0) for p in prices)
                min_price = (
                    min(p.get("low", 0) for p in prices)
                    if all(p.get("low", 0) > 0 for p in prices)
                    else 0
                )

                result.append(
                    f"{SYMBOLS['vertical']} {SYMBOLS['section_prefix']}Price Statistics:"
                )
                result.append(
                    f"{SYMBOLS['vertical']}   • Latest Price: {latest_price:.2f}"
                )
                result.append(
                    f"{SYMBOLS['vertical']}   • Average Price: {avg_price:.2f}"
                )
                result.append(
                    f"{SYMBOLS['vertical']}   • Highest Price: {max_price:.2f}"
                )
                result.append(
                    f"{SYMBOLS['vertical']}   • Lowest Price: {min_price:.2f}"
                )

        # Financial metrics summary
        fin_metrics = (
            data.get("financial_metrics", [{}])[0]
            if data.get("financial_metrics")
            else {}
        )
        if fin_metrics:
            result.append(
                f"{SYMBOLS['vertical']} {SYMBOLS['section_prefix']}Key Financial Metrics:"
            )

            # Show only key metrics
            key_metrics = {
                "pe_ratio": "P/E Ratio",
                "price_to_book": "P/B Ratio",
                "return_on_equity": "ROE",
                "debt_to_equity": "Debt/Equity Ratio",
                "earnings_growth": "Earnings Growth Rate",
            }

            for key, label in key_metrics.items():
                if key in fin_metrics:
                    value = fin_metrics[key]
                    result.append(f"{SYMBOLS['vertical']}   • {label}: {value}")

        # Add footer
        result.append(
            f"{SYMBOLS['footer_left']}{SYMBOLS['border'] * (width - 2)}{SYMBOLS['footer_right']}"
        )

        return result

    def _format_agent_section(self, agent_name: str, data: Any) -> List[str]:
        """Format agent section"""
        result = []

        # Get agent information
        agent_info = AGENT_MAP.get(agent_name, {"icon": "🔄", "name": agent_name})
        icon = agent_info["icon"]
        display_name = agent_info["name"]

        # Create title
        width = 80
        title = f"{icon} {display_name} Analysis"
        result.append(
            f"{SYMBOLS['header_left']}{SYMBOLS['border'] * ((width - len(title) - 2) // 2)} {title} {SYMBOLS['border'] * ((width - len(title) - 2) // 2)}{SYMBOLS['header_right']}"
        )

        # Add content
        if isinstance(data, dict):
            if agent_name == "market_data_agent":
                # Use simplified market data display
                return self._format_market_data_section(data)

            # Special handling for portfolio_management_agent and macro_analyst_agent
            if agent_name == "portfolio_management_agent":
                # Try to extract action and confidence
                if "action" in data:
                    action = data.get("action", "")
                    action_icon = STATUS_ICONS.get(action.lower(), "")
                    result.append(
                        f"{SYMBOLS['vertical']} Trading Action: {action_icon} {action.upper() if action else ''}"
                    )

                if "quantity" in data:
                    quantity = data.get("quantity", 0)
                    result.append(f"{SYMBOLS['vertical']} Trading Quantity: {quantity}")

                if "confidence" in data:
                    conf = data.get("confidence", 0)
                    if isinstance(conf, (int, float)) and conf <= 1:
                        conf_str = f"{conf*100:.0f}%"
                    else:
                        conf_str = str(conf)
                    result.append(
                        f"{SYMBOLS['vertical']} Decision Confidence: {conf_str}"
                    )

                # Display signals from each agent
                if "agent_signals" in data:
                    result.append(
                        f"{SYMBOLS['vertical']} {SYMBOLS['section_prefix']}Analyst Opinions:"
                    )

                    for signal_info in data["agent_signals"]:
                        agent = signal_info.get("agent", "")
                        signal = signal_info.get("signal", "")
                        conf = signal_info.get("confidence", 1.0)

                        # Skip empty signals
                        if not agent or not signal:
                            continue

                        # Get signal icon
                        signal_icon = STATUS_ICONS.get(signal.lower(), "")

                        # Format confidence
                        if isinstance(conf, (int, float)) and conf <= 1:
                            conf_str = f"{conf*100:.0f}%"
                        else:
                            conf_str = str(conf)

                        result.append(
                            f"{SYMBOLS['vertical']}   • {agent}: {signal_icon} {signal} (Confidence: {conf_str})"
                        )

                # Decision reasoning
                if "reasoning" in data:
                    reasoning = data["reasoning"]
                    result.append(
                        f"{SYMBOLS['vertical']} {SYMBOLS['section_prefix']}Decision Reasoning:"
                    )
                    if isinstance(reasoning, str):
                        # Split long text into multiple lines, each line not exceeding width-4 characters
                        for i in range(0, len(reasoning), width - 4):
                            line = reasoning[i : i + width - 4]
                            result.append(f"{SYMBOLS['vertical']}   {line}")
            elif agent_name == "macro_analyst_agent":
                # Handle macro analysis
                if isinstance(data, dict):
                    # Extract key information
                    macro_env = data.get("macro_environment", "")
                    impact = data.get("impact_on_stock", "")
                    key_factors = data.get("key_factors", [])

                    # Add highlighted macro environment and impact
                    env_icon = (
                        "📈"
                        if macro_env == "positive"
                        else "📉" if macro_env == "negative" else "◽"
                    )
                    impact_icon = (
                        "📈"
                        if impact == "positive"
                        else "📉" if impact == "negative" else "◽"
                    )

                    result.append(
                        f"{SYMBOLS['vertical']} Macro Environment: {env_icon} {macro_env}"
                    )
                    result.append(
                        f"{SYMBOLS['vertical']} Impact on Stock: {impact_icon} {impact}"
                    )

                    # Add key factors list
                    if key_factors:
                        result.append(
                            f"{SYMBOLS['vertical']} {SYMBOLS['section_prefix']}Key Factors:"
                        )
                        for i, factor in enumerate(
                            key_factors[:5]
                        ):  # Show at most 5 factors
                            result.append(f"{SYMBOLS['vertical']}   • {factor}")

                    # Add simplified reasoning
                    reasoning = data.get("reasoning", "")
                    if reasoning:
                        # Truncate first 100 characters as summary
                        reasoning_summary = (
                            reasoning[:100] + "..."
                            if len(reasoning) > 100
                            else reasoning
                        )
                        result.append(
                            f"{SYMBOLS['vertical']} {SYMBOLS['section_prefix']}Analysis Summary:"
                        )
                        result.append(f"{SYMBOLS['vertical']}   {reasoning_summary}")
            else:
                # Standard handling for other agents
                # Extract signal and confidence (if available)
                if "signal" in data:
                    signal = data.get("signal", "")
                    # Ensure signal is string type
                    if isinstance(signal, (int, float)):
                        # Convert numeric signal to string
                        if signal > 0.2:
                            signal = "bullish"
                        elif signal < -0.2:
                            signal = "bearish"
                        else:
                            signal = "neutral"
                    signal_str = str(signal)
                    signal_icon = STATUS_ICONS.get(signal_str.lower(), "")
                    result.append(
                        f"{SYMBOLS['vertical']} Signal: {signal_icon} {signal_str}"
                    )

                if "confidence" in data:
                    conf = data.get("confidence", "")
                    if isinstance(conf, (int, float)) and conf <= 1:
                        conf_str = f"{conf*100:.0f}%"
                    else:
                        conf_str = str(conf)
                    result.append(f"{SYMBOLS['vertical']} Confidence: {conf_str}")

            # Add other data
            tree_lines = self._format_dict_as_tree(data)
            for line in tree_lines:
                result.append(f"{SYMBOLS['vertical']} {line}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                prefix = (
                    SYMBOLS["tree_last"]
                    if i == len(data) - 1
                    else SYMBOLS["tree_branch"]
                )
                result.append(f"{SYMBOLS['vertical']} {prefix} {item}")
        else:
            result.append(f"{SYMBOLS['vertical']} {data}")

        # Add footer
        result.append(
            f"{SYMBOLS['footer_left']}{SYMBOLS['border'] * (width - 2)}{SYMBOLS['footer_right']}"
        )

        return result

    def generate_output(self) -> str:
        """Generate formatted output"""
        width = 80
        result = []

        # Add title
        ticker = self.metadata.get("ticker", "Unknown")
        title = f"Stock Code {ticker} Investment Analysis Report"
        result.append(SYMBOLS["border"] * width)
        result.append(f"{title:^{width}}")
        result.append(SYMBOLS["border"] * width)

        # Add date range (if available)
        if "start_date" in self.metadata and "end_date" in self.metadata:
            date_range = f"Analysis Period: {self.metadata['start_date']} to {self.metadata['end_date']}"
            result.append(f"{date_range:^{width}}")
            result.append("")

        # Add each agent's output in order
        for agent_name in AGENT_ORDER:
            if agent_name in self.data:
                result.extend(
                    self._format_agent_section(agent_name, self.data[agent_name])
                )
                result.append("")  # Add empty line

        # Add ending separator
        result.append(SYMBOLS["border"] * width)

        return "\n".join(result)

    def print_output(self) -> None:
        """Print formatted output"""
        output = self.generate_output()

        # Add ANSI color codes
        colored_output = output
        colored_output = colored_output.replace(
            "bullish", "\033[32mbullish\033[0m"
        )  # Green
        colored_output = colored_output.replace(
            "bearish", "\033[31mbearish\033[0m"
        )  # Red
        colored_output = colored_output.replace(
            "neutral", "\033[33mneutral\033[0m"
        )  # Yellow
        colored_output = colored_output.replace(
            "positive", "\033[32mpositive\033[0m"
        )  # Green
        colored_output = colored_output.replace(
            "negative", "\033[31mnegative\033[0m"
        )  # Red
        colored_output = colored_output.replace("BUY", "\033[32mBUY\033[0m")  # Green
        colored_output = colored_output.replace("SELL", "\033[31mSELL\033[0m")  # Red
        colored_output = colored_output.replace("HOLD", "\033[33mHOLD\033[0m")  # Yellow

        # Direct print output, not limited by log level
        print("\n" + colored_output)


# Create global instance
terminal = StructuredTerminalOutput()


def extract_agent_data(state: Dict[str, Any], agent_name: str) -> Any:
    """
    Extract data for specified agent from state

    Args:
        state: Workflow state
        agent_name: Agent name

    Returns:
        Extracted agent data
    """
    # Special handling for portfolio_management_agent
    if agent_name == "portfolio_management_agent":
        # Try to get data from the last message
        messages = state.get("messages", [])
        if messages and hasattr(messages[-1], "content"):
            content = messages[-1].content
            # Try to parse JSON
            if isinstance(content, str):
                try:
                    # If it's a JSON string, try to parse
                    if content.strip().startswith("{") and content.strip().endswith(
                        "}"
                    ):
                        return json.loads(content)
                    # If JSON string is contained in other text, try to extract and parse
                    json_start = content.find("{")
                    json_end = content.rfind("}")
                    if json_start >= 0 and json_end > json_start:
                        json_str = content[json_start : json_end + 1]
                        return json.loads(json_str)
                except json.JSONDecodeError:
                    # If parsing fails, return original content
                    return {"message": content}
            return {"message": content}

    # First try to get from all_agent_reasoning in metadata
    metadata = state.get("metadata", {})
    all_reasoning = metadata.get("all_agent_reasoning", {})

    # Find matching agent data
    for name, data in all_reasoning.items():
        if agent_name in name:
            return data

    # If not found in all_agent_reasoning, try to get from agent_reasoning
    if (
        agent_name == metadata.get("current_agent_name")
        and "agent_reasoning" in metadata
    ):
        return metadata["agent_reasoning"]

    # Try to get from messages
    messages = state.get("messages", [])
    for message in messages:
        if hasattr(message, "name") and message.name and agent_name in message.name:
            # Try to parse message content
            try:
                if hasattr(message, "content"):
                    content = message.content
                    # Try to parse JSON
                    if isinstance(content, str) and (
                        content.startswith("{") or content.startswith("[")
                    ):
                        try:
                            return json.loads(content)
                        except json.JSONDecodeError:
                            pass
                    return content
            except Exception:
                pass

    # If all else fails, return None
    return None


def process_final_state(state: Dict[str, Any]) -> None:
    """
    Process final state, extract all agent data

    Args:
        state: Final state of the workflow
    """
    # Extract metadata
    data = state.get("data", {})

    # Set metadata
    terminal.set_metadata("ticker", data.get("ticker", "Unknown"))
    if "start_date" in data and "end_date" in data:
        terminal.set_metadata("start_date", data["start_date"])
        terminal.set_metadata("end_date", data["end_date"])

    # Extract each agent's data
    for agent_name in AGENT_ORDER:
        agent_data = extract_agent_data(state, agent_name)
        if agent_data:
            terminal.add_agent_data(agent_name, agent_data)


def print_structured_output(state: Dict[str, Any]) -> None:
    """
    Process final state and print structured output

    Args:
        state: Final state of the workflow
    """
    try:
        # Process final state
        process_final_state(state)

        # Print output
        terminal.print_output()
    except Exception as e:
        logger.error(f"Error generating structured output: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
