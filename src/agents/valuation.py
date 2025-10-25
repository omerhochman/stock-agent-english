import json

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage

from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.calc.calculate_beta import calculate_beta
from src.calc.portfolio_optimization import portfolio_volatility
from src.tools.api import get_market_data, prices_to_df
from src.tools.factor_data_api import (
    get_market_returns,
    get_multiple_index_data,
    get_risk_free_rate,
)
from src.utils.api_utils import agent_endpoint
from src.utils.logging_config import setup_logger

logger = setup_logger("valuation_agent")


@agent_endpoint(
    "valuation",
    "Valuation analyst, using DCF and owner earnings methods to assess company intrinsic value",
)
def valuation_agent(state: AgentState):
    """Responsible for valuation analysis, using multiple methods to assess company intrinsic value"""
    show_workflow_status("Valuation Agent")
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    metrics = data["financial_metrics"][0]
    current_financial_line_item = data["financial_line_items"][0]
    previous_financial_line_item = data["financial_line_items"][1]
    market_cap = data["market_cap"]
    ticker = data.get("ticker", "")

    # Extract price data for Beta calculation
    prices = data.get("prices", [])
    start_date = data.get("start_date")
    end_date = data.get("end_date")

    reasoning = {}

    # Calculate working capital change
    working_capital_change = (
        current_financial_line_item.get("working_capital") or 0
    ) - (previous_financial_line_item.get("working_capital") or 0)

    # 1. Get real market data
    # Convert price data to DataFrame
    prices_df = prices_to_df(prices)

    # 2. Get market data, calculate actual market volatility
    market_data = {}
    market_volatility = None  # Initialize as None for subsequent default value judgment

    try:
        # Get major market index data
        indices = [
            "000300",
            "399001",
            "399006",
        ]  # CSI 300, Shenzhen Component Index, ChiNext Index
        market_indices = get_multiple_index_data(indices, start_date, end_date)

        # Create market index return series
        market_returns_dict = {}
        for idx_name, idx_data in market_indices.items():
            if isinstance(idx_data, pd.DataFrame) and "close" in idx_data.columns:
                idx_df = idx_data.copy()
                idx_df["return"] = idx_df["close"].pct_change()
                market_returns_dict[idx_name] = idx_df["return"].dropna()

        # Use CSI 300 as main market benchmark
        if "000300" in market_returns_dict:
            market_returns = market_returns_dict["000300"]
            # Calculate actual market volatility
            market_volatility = market_returns.std() * np.sqrt(252)
            market_data["market_returns"] = market_returns
            market_data["market_volatility"] = market_volatility
            logger.info(
                f"Successfully obtained market data - Market volatility: {market_volatility:.2%}"
            )
        else:
            logger.warning(
                "Unable to get CSI 300 index data, will try other methods to get market data"
            )
    except Exception as e:
        logger.error(f"Failed to get market data: {e}")

    # 3. Get risk metrics (market return rate and risk-free rate)
    try:
        # Get real market return rate and risk-free rate
        market_returns_data = get_market_returns(
            start_date=start_date, end_date=end_date
        )
        risk_free_data = get_risk_free_rate(start_date=start_date, end_date=end_date)

        # Use actual data
        market_return = (
            market_returns_data.mean() * 252 if not market_returns_data.empty else 0.08
        )
        risk_free_rate = risk_free_data.mean() if not risk_free_data.empty else 0.03

        logger.info(
            f"Obtained real market data: Market return={market_return:.2%}, Risk-free rate={risk_free_rate:.2%}"
        )
    except Exception as e:
        # Use reasonable default values when unable to get data
        logger.warning(
            f"Unable to get market return rate and risk-free rate data: {e}, using default values"
        )
        market_return = 0.08  # Default 8% market return rate
        risk_free_rate = 0.03  # Default 3% risk-free rate

    # 4. Calculate Beta using improved Beta calculation function
    beta_value = calculate_beta(
        ticker, "000300", start_date, end_date
    )  # Use function from calc/calculate_beta.py
    logger.info(
        f"Calculated Beta value using improved Beta calculation function: {beta_value:.2f}"
    )

    # 5. If market volatility not successfully obtained, calculate using portfolio volatility function
    if market_volatility is None:
        if not prices_df.empty:
            try:
                # Use portfolio_volatility to calculate stock volatility
                stock_returns = prices_df["close"].pct_change().dropna()
                stock_volatility = portfolio_volatility(
                    np.ones(1), np.array([[stock_returns.var()]])
                ) * np.sqrt(252)

                # Use beta value and stock volatility to reverse calculate market volatility
                if beta_value > 0:
                    market_volatility = stock_volatility / beta_value
                else:
                    market_volatility = 0.15  # Default value

                logger.info(
                    f"Reverse calculated market volatility using portfolio_volatility and beta: {market_volatility:.2%}"
                )
            except Exception as e:
                logger.error(f"Failed to calculate market volatility: {e}")
                market_volatility = 0.15  # Default value
        else:
            market_volatility = 0.15  # Default value
            logger.warning("No price data, using default market volatility: 15%")

    # 6. Calculate Weighted Average Cost of Capital (WACC)
    required_return = calculate_required_return(
        ticker=ticker,
        beta=beta_value,
        risk_free_rate=risk_free_rate,
        market_premium=market_return - risk_free_rate,
    )

    # 7. Use owner earnings method for valuation (improved Buffett method)
    owner_earnings_value = calculate_owner_earnings_value(
        net_income=current_financial_line_item.get("net_income"),
        depreciation=current_financial_line_item.get("depreciation_and_amortization"),
        capex=current_financial_line_item.get("capital_expenditure"),
        working_capital_change=working_capital_change,
        growth_rate=metrics.get("earnings_growth", 0.05),
        required_return=required_return,
        margin_of_safety=0.25,
    )

    # 8. Use DCF valuation (multi-stage model)
    dcf_value = calculate_multistage_dcf(
        free_cash_flow=current_financial_line_item.get("free_cash_flow"),
        revenue_growth=metrics.get("revenue_growth", 0.05),
        earnings_growth=metrics.get("earnings_growth", 0.05),
        net_margin=metrics.get("net_margin", 0.1),
        wacc=required_return,
    )

    # 9. Get industry data for relative valuation analysis
    # First try to get actual industry averages from market data tools
    try:
        # Get industry data
        industry_data = get_market_data(ticker)

        # Extract industry averages
        industry_avg_pe = industry_data.get("industry_avg_pe", 15)
        industry_avg_pb = industry_data.get("industry_avg_pb", 1.5)
        industry_growth = industry_data.get("industry_growth", 0.05)

        logger.info(
            f"Obtained industry data: PE={industry_avg_pe}, PB={industry_avg_pb}, Growth Rate={industry_growth}"
        )
    except Exception as e:
        logger.warning(f"Unable to obtain industry data: {e}, using default values")
        # Use reasonable default values
        industry_avg_pe = 15
        industry_avg_pb = 1.5
        industry_growth = 0.05

    # Perform relative valuation analysis
    pe_ratio = metrics.get("pe_ratio", 0)
    price_to_book = metrics.get("price_to_book", 0)

    # Adjust PE based on company and industry growth rates
    company_growth = metrics.get("earnings_growth", 0.05)
    growth_premium = (
        max(0, company_growth - industry_growth) * 10
    )  # Give 10 PE premium for each 1% additional growth
    adjusted_industry_pe = industry_avg_pe + growth_premium

    # Calculate relative PE ratio and PB ratio
    pe_relative = pe_ratio / adjusted_industry_pe if adjusted_industry_pe else 1
    pb_relative = price_to_book / industry_avg_pb if industry_avg_pb else 1

    # Intrinsic value based on relative valuation
    earnings_per_share = metrics.get("earnings_per_share", 0)
    relative_value = (
        earnings_per_share * adjusted_industry_pe if earnings_per_share else 0
    )

    # 10. Calculate residual income model valuation (Residual Income Model)
    book_value_per_share = metrics.get("book_value_per_share", 0)
    roi = metrics.get("return_on_equity", 0.1)
    residual_income_value = 0

    if book_value_per_share and roi > 0:
        # Calculate excess return (ROE - required return)
        excess_return = max(0, roi - required_return)
        # Simplified residual income model (excess return persists for 5 years, then linearly decays to 0)
        residual_income = book_value_per_share * excess_return
        present_value = 0
        for i in range(1, 11):  # 10-year period
            year_factor = (
                1 - (i / 10) * 0.5 if i <= 5 else max(0, 1 - 0.5 - (i - 5) / 5 * 0.5)
            )
            residual_income_year = residual_income * year_factor
            present_value += residual_income_year / ((1 + required_return) ** i)

        residual_income_value = book_value_per_share + present_value

    # Add to valuation list
    all_valuations = [
        {"method": "DCF", "value": dcf_value, "weight": 0.35},
        {"method": "Owner Earnings", "value": owner_earnings_value, "weight": 0.35},
        {"method": "Relative Valuation", "value": relative_value, "weight": 0.15},
        {"method": "Residual Income", "value": residual_income_value, "weight": 0.15},
    ]

    # Calculate weighted average valuation
    weighted_sum = sum(v["value"] * v["weight"] for v in all_valuations)
    weight_sum = sum(v["weight"] for v in all_valuations)
    weighted_value = weighted_sum / weight_sum if weight_sum else 0

    # Calculate valuation gap
    dcf_gap = (dcf_value - market_cap) / market_cap if market_cap else 0
    owner_earnings_gap = (
        (owner_earnings_value - market_cap) / market_cap if market_cap else 0
    )
    relative_gap = (relative_value - market_cap) / market_cap if market_cap else 0
    residual_income_gap = (
        (residual_income_value - market_cap) / market_cap if market_cap else 0
    )
    weighted_gap = (weighted_value - market_cap) / market_cap if market_cap else 0

    # 11. Determine investment signal
    if weighted_gap > 0.15:  # 15% or more undervalued
        signal = "bullish"
        confidence = min(abs(weighted_gap), 0.50)  # Maximum 50% confidence
    elif weighted_gap < -0.15:  # 15% or more overvalued
        signal = "bearish"
        confidence = min(abs(weighted_gap), 0.50)  # Maximum 50% confidence
    else:
        signal = "neutral"
        confidence = 0.25  # Lower confidence for neutral signals

    # Adjust confidence based on consistency of valuation models
    # Calculate variance between models, higher consistency increases confidence
    valuation_signals = []
    for val in all_valuations:
        gap = (val["value"] - market_cap) / market_cap if market_cap else 0
        if gap > 0.15:
            valuation_signals.append(1)  # Bullish
        elif gap < -0.15:
            valuation_signals.append(-1)  # Bearish
        else:
            valuation_signals.append(0)  # Neutral

    # Calculate signal variance, low variance indicates high consistency
    signal_variance = np.var(valuation_signals) if len(valuation_signals) > 1 else 1
    consistency_boost = max(
        0, 1 - signal_variance * 0.5
    )  # 50% boost when variance is 0
    confidence = min(
        confidence * (1 + consistency_boost * 0.5), 1.0
    )  # Maximum 50% boost

    # 12. Record reasoning
    reasoning["dcf_analysis"] = {
        "signal": (
            "bullish" if dcf_gap > 0.15 else "bearish" if dcf_gap < -0.15 else "neutral"
        ),
        "details": f"Intrinsic Value: ${dcf_value:,.2f}, Market Cap: ${market_cap:,.2f}, Difference: {dcf_gap:.1%}",
        "model_details": {
            "stages": "Multi-stage DCF",
            "wacc": f"{required_return:.1%}",
            "beta": f"{beta_value:.2f}",
            "terminal_growth": "3.0%",
        },
    }

    reasoning["owner_earnings_analysis"] = {
        "signal": (
            "bullish"
            if owner_earnings_gap > 0.15
            else "bearish" if owner_earnings_gap < -0.15 else "neutral"
        ),
        "details": f"Owner Earnings Value: ${owner_earnings_value:,.2f}, Market Cap: ${market_cap:,.2f}, Difference: {owner_earnings_gap:.1%}",
        "model_details": {
            "required_return": f"{required_return:.1%}",
            "margin_of_safety": "25%",
            "growth_rate": f"{metrics.get('earnings_growth', 0.05):.1%}",
        },
    }

    reasoning["relative_valuation"] = {
        "signal": (
            "bullish"
            if relative_gap > 0.15
            else "bearish" if relative_gap < -0.15 else "neutral"
        ),
        "details": f"Relative Valuation: ${relative_value:,.2f}, Market Cap: ${market_cap:,.2f}, Difference: {relative_gap:.1%}",
        "model_details": {
            "pe_ratio": f"{pe_ratio:.2f} (Industry Average Adjusted: {adjusted_industry_pe:.2f})",
            "pb_ratio": f"{price_to_book:.2f} (Industry Average: {industry_avg_pb:.2f})",
            "growth_premium": f"{growth_premium:.1f}",
        },
    }

    reasoning["residual_income_valuation"] = {
        "signal": (
            "bullish"
            if residual_income_gap > 0.15
            else "bearish" if residual_income_gap < -0.15 else "neutral"
        ),
        "details": f"Residual Income Value: ${residual_income_value:,.2f}, Market Cap: ${market_cap:,.2f}, Difference: {residual_income_gap:.1%}",
        "model_details": {
            "book_value": f"${book_value_per_share:.2f}",
            "roe": f"{roi:.1%}",
            "excess_return": f"{max(0, roi - required_return):.1%}",
        },
    }

    reasoning["weighted_valuation"] = {
        "signal": signal,
        "details": f"Weighted Valuation: ${weighted_value:,.2f}, Market Cap: ${market_cap:,.2f}, Difference: {weighted_gap:.1%}",
        "weights": {v["method"]: f"{v['weight']*100:.0f}%" for v in all_valuations},
        "consistency": f"{consistency_boost:.2f}",
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
            "market_volatility": float(market_volatility),
        },
    }

    message = HumanMessage(
        content=json.dumps(message_content),
        name="valuation_agent",
    )

    if show_reasoning:
        show_agent_reasoning(message_content, "Valuation Analysis Agent")
        # Save reasoning information to metadata for API use
        state["metadata"]["agent_reasoning"] = message_content

    show_workflow_status("Valuation Agent", "completed")
    return {
        "messages": [message],
        "data": {**data, "valuation_analysis": message_content},
        "metadata": state["metadata"],
    }


def calculate_required_return(ticker, beta, risk_free_rate=0.03, market_premium=0.055):
    """
    Calculate required return/cost of capital using CAPM model

    Args:
        ticker: Stock code
        beta: Calculated beta value
        risk_free_rate: Risk-free rate
        market_premium: Market risk premium

    Returns:
        float: Required return/WACC estimate
    """
    # Use CAPM formula: WACC = Risk-free rate + Beta * Market risk premium
    required_return = risk_free_rate + (beta * market_premium)

    # Ensure reasonable range
    required_return = max(min(required_return, 0.20), 0.05)  # 5%-20% range

    return required_return


def calculate_owner_earnings_value(
    net_income: float,
    depreciation: float,
    capex: float,
    working_capital_change: float,
    growth_rate: float = 0.05,
    required_return: float = 0.15,
    margin_of_safety: float = 0.25,
    num_years: int = 5,
) -> float:
    """
    Calculate company value using improved owner earnings method, considering diminishing marginal growth.

    Args:
        net_income: Net income
        depreciation: Depreciation and amortization
        capex: Capital expenditures
        working_capital_change: Working capital change
        growth_rate: Initial expected growth rate
        required_return: Required return rate
        margin_of_safety: Safety margin ratio
        num_years: Forecast years

    Returns:
        float: Valuation based on owner earnings method
    """
    try:
        # Data validity check
        if not all(
            isinstance(x, (int, float))
            for x in [net_income, depreciation, capex, working_capital_change]
        ):
            return 0

        # Calculate initial owner earnings
        owner_earnings = net_income + depreciation - capex - working_capital_change

        if owner_earnings <= 0:
            return 0

        # Adjust growth rate to ensure reasonableness
        growth_rate = min(max(growth_rate, 0), 0.25)  # Limit to 0-25% range

        # Calculate present value of forecast period earnings
        future_values = []
        for year in range(1, num_years + 1):
            # Diminishing growth rate model - growth rate decays over time
            year_growth = growth_rate * (1 - year / (2 * num_years))
            future_value = owner_earnings * (1 + year_growth) ** year
            discounted_value = future_value / (1 + required_return) ** year
            future_values.append(discounted_value)

        # Use two-stage model to calculate terminal value
        # Stage 1: First num_years years, calculated above
        # Stage 2: Perpetual growth stage
        terminal_growth = min(
            growth_rate * 0.4, 0.03
        )  # Take 40% of growth rate or 3%, whichever is smaller
        terminal_value = (
            future_values[-1]
            * (1 + terminal_growth)
            / (required_return - terminal_growth)
        )
        terminal_value_discounted = terminal_value / (1 + required_return) ** num_years

        # Calculate total value and apply safety margin
        intrinsic_value = sum(future_values) + terminal_value_discounted
        value_with_safety_margin = intrinsic_value * (1 - margin_of_safety)

        return max(value_with_safety_margin, 0)  # Ensure no negative values returned

    except Exception as e:
        logger.error(f"Owner earnings calculation error: {e}")
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
    Calculate intrinsic value using three-stage DCF model

    Stage 1: High growth period - use provided growth rate
    Stage 2: Transition period - growth rate linearly decreases to perpetual growth rate
    Stage 3: Perpetual period - use terminal growth rate

    Args:
        free_cash_flow: Current free cash flow
        revenue_growth: Revenue growth rate
        earnings_growth: Earnings growth rate
        net_margin: Net profit margin
        wacc: Weighted average cost of capital
        high_growth_years: High growth stage years
        transition_years: Transition period years
        terminal_growth_rate: Terminal growth rate

    Returns:
        float: Three-stage DCF valuation
    """
    try:
        if not isinstance(free_cash_flow, (int, float)) or free_cash_flow <= 0:
            return 0

        # Use net profit margin adjusted growth rate for more accurate predictions
        # High growth rates and profitability usually have positive correlation
        adjusted_growth = (earnings_growth + revenue_growth) / 2
        adjusted_growth = min(max(adjusted_growth, 0), 0.25)  # Limit to 0-25% range

        # Adjust prediction reliability based on net profit margin
        reliability_factor = min(
            max(net_margin / 0.15, 0.5), 1.5
        )  # Ratio of net margin to industry average (15%)
        adjusted_growth = adjusted_growth * reliability_factor

        # Stage 1: High growth period
        high_growth_values = []
        current_fcf = free_cash_flow

        for year in range(1, high_growth_years + 1):
            current_fcf = current_fcf * (1 + adjusted_growth)
            present_value = current_fcf / ((1 + wacc) ** year)
            high_growth_values.append(present_value)

        # Stage 2: Transition period
        transition_values = []
        for year in range(1, transition_years + 1):
            # Linearly reduce growth rate
            transition_growth = adjusted_growth - (
                adjusted_growth - terminal_growth_rate
            ) * (year / transition_years)
            current_fcf = current_fcf * (1 + transition_growth)
            present_value = current_fcf / ((1 + wacc) ** (high_growth_years + year))
            transition_values.append(present_value)

        # Stage 3: Perpetual period
        final_fcf = current_fcf * (1 + terminal_growth_rate)
        terminal_value = final_fcf / (wacc - terminal_growth_rate)
        present_terminal_value = terminal_value / (
            (1 + wacc) ** (high_growth_years + transition_years)
        )

        # Calculate total present value
        total_value = (
            sum(high_growth_values) + sum(transition_values) + present_terminal_value
        )

        return max(total_value, 0)  # Ensure no negative values returned

    except Exception as e:
        logger.error(f"DCF calculation error: {e}")
        return 0
