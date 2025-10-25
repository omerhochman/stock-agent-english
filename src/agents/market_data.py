from datetime import datetime, timedelta

import pandas as pd

from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.tools.api import (
    get_financial_metrics,
    get_financial_statements,
    get_market_data,
    get_price_history,
)
from src.utils.api_utils import agent_endpoint
from src.utils.logging_config import setup_logger

# Setup logging
logger = setup_logger("market_data_agent")


@agent_endpoint(
    "market_data",
    "Market data collection, responsible for obtaining stock price history, financial indicators and market information",
)
def market_data_agent(state: AgentState):
    """Responsible for gathering and preprocessing market data"""
    show_workflow_status("Market Data Agent")
    show_reasoning = state["metadata"]["show_reasoning"]

    messages = state["messages"]
    data = state["data"]

    # Set default dates
    current_date = datetime.now()
    yesterday = current_date - timedelta(days=1)
    end_date = data["end_date"] or yesterday.strftime("%Y-%m-%d")

    # Ensure end_date is not in the future
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
    if end_date_obj > yesterday:
        end_date = yesterday.strftime("%Y-%m-%d")
        end_date_obj = yesterday

    if not data["start_date"]:
        # Calculate 1 year before end_date
        start_date = end_date_obj - timedelta(
            days=365
        )  # Default to get one year of data
        start_date = start_date.strftime("%Y-%m-%d")
    else:
        start_date = data["start_date"]

    # Check if multiple stock codes are provided
    tickers = data.get("tickers", [])
    if isinstance(tickers, str):
        tickers = [ticker.strip() for ticker in tickers.split(",")]

    # If no tickers provided but ticker is provided, use single ticker
    if not tickers and data.get("ticker"):
        tickers = [data["ticker"]]

    # If still no stock codes, return error
    if not tickers:
        logger.warning("No stock code provided, unable to get market data")
        return {
            "messages": messages,
            "data": {
                **data,
                "error": "No stock code provided, please provide stock code to get market data",
            },
            "metadata": state["metadata"],
        }

    # Collect data for multiple stocks
    all_stock_data = {}
    primary_ticker = tickers[0]  # Primary stock for analysis

    for ticker in tickers:
        logger.info(f"Getting data for {ticker}...")

        # Get price data and validate
        prices_df = get_price_history(ticker, start_date, end_date)
        if prices_df is None or prices_df.empty:
            logger.warning(
                f"Warning: Unable to get price data for {ticker}, continuing with empty data"
            )
            prices_df = pd.DataFrame(columns=["close", "open", "high", "low", "volume"])

        # Get financial metrics
        try:
            financial_metrics = get_financial_metrics(ticker)
        except Exception as e:
            logger.error(f"Failed to get financial metrics for {ticker}: {str(e)}")
            financial_metrics = {}

        # Get financial statements
        try:
            financial_line_items = get_financial_statements(ticker)
        except Exception as e:
            logger.error(f"Failed to get {ticker} financial statements: {str(e)}")
            financial_line_items = {}

        # Get market data
        try:
            market_data = get_market_data(ticker)
        except Exception as e:
            logger.error(f"Failed to get {ticker} market data: {str(e)}")
            market_data = {"market_cap": 0}

        # Ensure data format is correct
        if not isinstance(prices_df, pd.DataFrame):
            prices_df = pd.DataFrame(columns=["close", "open", "high", "low", "volume"])

        # Convert price data to dictionary format
        prices_dict = prices_df.to_dict("records")

        # Save single stock data
        all_stock_data[ticker] = {
            "prices": prices_dict,
            "financial_metrics": financial_metrics,
            "financial_line_items": financial_line_items,
            "market_cap": market_data.get("market_cap", 0),
            "market_data": market_data,
        }

    # Save reasoning information to metadata for API use
    market_data_summary = {
        "tickers": tickers,
        "primary_ticker": primary_ticker,
        "start_date": start_date,
        "end_date": end_date,
        "data_collected": {
            ticker: {
                "price_history": len(all_stock_data[ticker]["prices"]) > 0,
                "financial_metrics": len(all_stock_data[ticker]["financial_metrics"])
                > 0,
                "financial_statements": len(
                    all_stock_data[ticker]["financial_line_items"]
                )
                > 0,
                "market_data": len(all_stock_data[ticker]["market_data"]) > 0,
            }
            for ticker in tickers
        },
        "summary": f"Collected market data for {', '.join(tickers)} from {start_date} to {end_date}, including price history, financial metrics and market information",
    }

    if show_reasoning:
        show_agent_reasoning(market_data_summary, "Market Data Agent")
        state["metadata"]["agent_reasoning"] = market_data_summary

    # Save primary stock data in original location to maintain compatibility with other Agents
    return {
        "messages": messages,
        "data": {
            **data,
            "ticker": primary_ticker,
            "tickers": tickers,
            "prices": all_stock_data[primary_ticker]["prices"],
            "start_date": start_date,
            "end_date": end_date,
            "financial_metrics": all_stock_data[primary_ticker]["financial_metrics"],
            "financial_line_items": all_stock_data[primary_ticker][
                "financial_line_items"
            ],
            "market_cap": all_stock_data[primary_ticker]["market_cap"],
            "market_data": all_stock_data[primary_ticker]["market_data"],
            "all_stock_data": all_stock_data,  # Contains data for all stocks
        },
        "metadata": state["metadata"],
    }
