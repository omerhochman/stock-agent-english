from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.tools.api import get_financial_metrics, get_financial_statements, get_market_data, get_price_history
from src.utils.logging_config import setup_logger
from src.utils.api_utils import agent_endpoint

from datetime import datetime, timedelta
import pandas as pd

# 设置日志记录
logger = setup_logger('market_data_agent')


@agent_endpoint("market_data", "市场数据收集，负责获取股价历史、财务指标和市场信息")
def market_data_agent(state: AgentState):
    """Responsible for gathering and preprocessing market data"""
    show_workflow_status("Market Data Agent")
    show_reasoning = state["metadata"]["show_reasoning"]

    messages = state["messages"]
    data = state["data"]

    # Set default dates
    current_date = datetime.now()
    yesterday = current_date - timedelta(days=1)
    end_date = data["end_date"] or yesterday.strftime('%Y-%m-%d')

    # Ensure end_date is not in the future
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    if end_date_obj > yesterday:
        end_date = yesterday.strftime('%Y-%m-%d')
        end_date_obj = yesterday

    if not data["start_date"]:
        # Calculate 1 year before end_date
        start_date = end_date_obj - timedelta(days=365)  # 默认获取一年的数据
        start_date = start_date.strftime('%Y-%m-%d')
    else:
        start_date = data["start_date"]

    # 检查是否提供了多个股票代码
    tickers = data.get("tickers", [])
    if isinstance(tickers, str):
        tickers = [ticker.strip() for ticker in tickers.split(',')]
    
    # 如果没有提供tickers但提供了ticker，则使用单一ticker
    if not tickers and data.get("ticker"):
        tickers = [data["ticker"]]
    
    # 如果仍然没有股票代码，则返回错误
    if not tickers:
        logger.warning("未提供股票代码，无法获取市场数据")
        return {
            "messages": messages,
            "data": {
                **data,
                "error": "未提供股票代码，请提供股票代码以获取市场数据"
            },
            "metadata": state["metadata"],
        }

    # 收集多个股票的数据
    all_stock_data = {}
    primary_ticker = tickers[0]  # 主要分析的股票

    for ticker in tickers:
        logger.info(f"正在获取 {ticker} 的数据...")
        
        # 获取价格数据并验证
        prices_df = get_price_history(ticker, start_date, end_date)
        if prices_df is None or prices_df.empty:
            logger.warning(f"警告：无法获取{ticker}的价格数据，将使用空数据继续")
            prices_df = pd.DataFrame(
                columns=['close', 'open', 'high', 'low', 'volume'])

        # 获取财务指标
        try:
            financial_metrics = get_financial_metrics(ticker)
        except Exception as e:
            logger.error(f"获取{ticker}财务指标失败: {str(e)}")
            financial_metrics = {}

        # 获取财务报表
        try:
            financial_line_items = get_financial_statements(ticker)
        except Exception as e:
            logger.error(f"获取{ticker}财务报表失败: {str(e)}")
            financial_line_items = {}

        # 获取市场数据
        try:
            market_data = get_market_data(ticker)
        except Exception as e:
            logger.error(f"获取{ticker}市场数据失败: {str(e)}")
            market_data = {"market_cap": 0}

        # 确保数据格式正确
        if not isinstance(prices_df, pd.DataFrame):
            prices_df = pd.DataFrame(
                columns=['close', 'open', 'high', 'low', 'volume'])

        # 转换价格数据为字典格式
        prices_dict = prices_df.to_dict('records')

        # 保存单只股票的数据
        all_stock_data[ticker] = {
            "prices": prices_dict,
            "financial_metrics": financial_metrics,
            "financial_line_items": financial_line_items,
            "market_cap": market_data.get("market_cap", 0),
            "market_data": market_data,
        }

    # 保存推理信息到metadata供API使用
    market_data_summary = {
        "tickers": tickers,
        "primary_ticker": primary_ticker,
        "start_date": start_date,
        "end_date": end_date,
        "data_collected": {
            ticker: {
                "price_history": len(all_stock_data[ticker]["prices"]) > 0,
                "financial_metrics": len(all_stock_data[ticker]["financial_metrics"]) > 0,
                "financial_statements": len(all_stock_data[ticker]["financial_line_items"]) > 0,
                "market_data": len(all_stock_data[ticker]["market_data"]) > 0,
            }
            for ticker in tickers
        },
        "summary": f"为{', '.join(tickers)}收集了从{start_date}到{end_date}的市场数据，包括价格历史、财务指标和市场信息"
    }

    if show_reasoning:
        show_agent_reasoning(market_data_summary, "Market Data Agent")
        state["metadata"]["agent_reasoning"] = market_data_summary

    # 将主要股票数据保存在原始位置，以保持与其他Agent的兼容性
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
            "financial_line_items": all_stock_data[primary_ticker]["financial_line_items"],
            "market_cap": all_stock_data[primary_ticker]["market_cap"],
            "market_data": all_stock_data[primary_ticker]["market_data"],
            "all_stock_data": all_stock_data,  # 包含所有股票的数据
        },
        "metadata": state["metadata"],
    }