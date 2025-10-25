import os
from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger("tushare_adapter")

# Get TuShare API key
TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN", "")


def get_tushare_price_data(
    tushare_code: str, start_date: str, end_date: str, adjust: str
) -> pd.DataFrame:
    """Get price data from TuShare"""
    try:
        import tushare as ts

        if not TUSHARE_TOKEN:
            logger.warning("TuShare token not found")
            return pd.DataFrame()

        ts.set_token(TUSHARE_TOKEN)
        pro = ts.pro_api()

        logger.info(f"Fetching price history using TuShare for {tushare_code}")

        # Convert adjustment type
        adj = None
        if adjust == "qfq":
            adj = "qfq"
        elif adjust == "hfq":
            adj = "hfq"

        # Try using pro_bar to get adjusted data
        try:
            if adj:
                df = ts.pro_bar(
                    ts_code=tushare_code,
                    adj=adj,
                    start_date=start_date,
                    end_date=end_date,
                )
            else:
                df = ts.pro_bar(
                    ts_code=tushare_code, start_date=start_date, end_date=end_date
                )
        except Exception as e:
            logger.warning(f"TuShare pro_bar failed: {str(e)}, trying daily API...")
            # Alternative method: get daily data
            df = pro.daily(
                ts_code=tushare_code, start_date=start_date, end_date=end_date
            )

            # If adjustment is needed
            if adj and not df.empty:
                adj_factor = pro.adj_factor(
                    ts_code=tushare_code, start_date=start_date, end_date=end_date
                )
                if not adj_factor.empty:
                    df = df.merge(adj_factor, on=["ts_code", "trade_date"])

                    # Forward adjustment
                    if adj == "qfq":
                        latest_factor = adj_factor["adj_factor"].iloc[0]
                        df["adj_factor"] = df["adj_factor"] / latest_factor

                    # Apply adjustment factor
                    for col in ["open", "high", "low", "close"]:
                        if col in df.columns:
                            df[col] = df[col] * df["adj_factor"]

        # Convert TuShare data format
        if not df.empty:
            # Create unified column name mapping
            column_mapping = {
                "trade_date": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "vol": "volume",
                "volume": "volume",  # pro_bar may use volume column name
                "amount": "amount",
                "pct_chg": "pct_change",
                "change": "change_amount",
                "turnover_rate": "turnover",
                "turn": "turnover",  # pro_bar may use turn column name
            }

            # Only rename existing columns
            rename_cols = {
                col: new_col
                for col, new_col in column_mapping.items()
                if col in df.columns
            }
            df = df.rename(columns=rename_cols)

            # Ensure date format is unified
            if "date" not in df.columns and "trade_date" in df.columns:
                df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
            elif "date" in df.columns:
                if df["date"].dtype == object:  # If it's string format
                    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

            # Ensure all numeric columns are converted to float type
            for col in [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "pct_change",
                "change_amount",
                "turnover",
            ]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Sort
            df = df.sort_values("date", ascending=True)

            logger.info(f"Successfully retrieved data from TuShare: {len(df)} records")
            return df
        return pd.DataFrame()
    except (ImportError, Exception) as e:
        logger.warning(f"TuShare error: {str(e)}")
        return pd.DataFrame()


def get_tushare_financial_metrics(tushare_code: str) -> List[Dict[str, Any]]:
    """Get financial metrics from TuShare"""
    try:
        import tushare as ts

        if not TUSHARE_TOKEN:
            logger.warning("TuShare token not found")
            return [{}]

        ts.set_token(TUSHARE_TOKEN)
        pro = ts.pro_api()

        logger.info(f"Fetching financial metrics using TuShare for {tushare_code}")

        # Get basic information
        basic_info = pro.daily_basic(ts_code=tushare_code)

        if not basic_info.empty:
            basic_info = basic_info.iloc[0]

            # Get financial indicators
            fin_indicator = pro.fina_indicator(ts_code=tushare_code)
            latest_fin = (
                fin_indicator.iloc[0] if not fin_indicator.empty else pd.Series()
            )

            # Get income statement
            income = pro.income(ts_code=tushare_code)
            latest_income = income.iloc[0] if not income.empty else pd.Series()
            prev_income = income.iloc[1] if len(income) > 1 else pd.Series()

            # Calculate growth rates
            revenue_current = float(latest_income.get("revenue", 0))
            revenue_prev = float(prev_income.get("revenue", 0))
            revenue_growth = (
                (revenue_current - revenue_prev) / revenue_prev
                if revenue_prev > 0
                else 0
            )

            net_profit_current = float(latest_income.get("n_income", 0))
            net_profit_prev = float(prev_income.get("n_income", 0))
            earnings_growth = (
                (net_profit_current - net_profit_prev) / net_profit_prev
                if net_profit_prev > 0
                else 0
            )

            # Build metrics dictionary
            agent_metrics = {
                # Profitability indicators
                "return_on_equity": float(latest_fin.get("roe", 0)),
                "net_margin": float(latest_fin.get("netprofit_margin", 0)),
                "operating_margin": (
                    float(latest_fin.get("profit_dedt", 0))
                    / float(latest_income.get("revenue", 1))
                    if float(latest_income.get("revenue", 0)) > 0
                    else 0
                ),
                # Growth indicators
                "revenue_growth": revenue_growth,
                "earnings_growth": earnings_growth,
                "book_value_growth": float(latest_fin.get("equity_yoy", 0)),
                # Financial health indicators
                "current_ratio": float(latest_fin.get("current_ratio", 0)),
                "debt_to_equity": float(latest_fin.get("debt_to_assets", 0)),
                "free_cash_flow_per_share": float(latest_fin.get("fcff_ps", 0)),
                "earnings_per_share": float(latest_fin.get("eps", 0)),
                # Valuation ratios
                "pe_ratio": float(basic_info.get("pe", 0)),
                "price_to_book": float(basic_info.get("pb", 0)),
                "price_to_sales": float(basic_info.get("ps", 0)),
            }

            return [agent_metrics]
        return [{}]
    except (ImportError, Exception) as e:
        logger.warning(f"TuShare financial metrics error: {str(e)}")
        return [{}]


def get_tushare_financial_statements(tushare_code: str) -> List[Dict[str, Any]]:
    """Get financial statements data from TuShare"""
    try:
        import tushare as ts

        if not TUSHARE_TOKEN:
            logger.warning("TuShare token not found")
            return [{}, {}]

        ts.set_token(TUSHARE_TOKEN)
        pro = ts.pro_api()

        logger.info(f"Fetching financial statements using TuShare for {tushare_code}")

        # Get balance sheet
        balance = pro.balancesheet(ts_code=tushare_code)
        if not balance.empty:
            latest_balance = balance.iloc[0]
            previous_balance = balance.iloc[1] if len(balance) > 1 else balance.iloc[0]

            # Get income statement
            income = pro.income(ts_code=tushare_code)
            latest_income = income.iloc[0] if not income.empty else pd.Series()
            previous_income = income.iloc[1] if len(income) > 1 else income.iloc[0]

            # Get cash flow statement
            cashflow = pro.cashflow(ts_code=tushare_code)
            latest_cashflow = cashflow.iloc[0] if not cashflow.empty else pd.Series()
            previous_cashflow = (
                cashflow.iloc[1] if len(cashflow) > 1 else cashflow.iloc[0]
            )

            # Build financial data items
            current_item = {
                "net_income": float(latest_income.get("n_income", 0)),
                "operating_revenue": float(latest_income.get("revenue", 0)),
                "operating_profit": float(latest_income.get("operate_profit", 0)),
                "working_capital": float(latest_balance.get("total_cur_assets", 0))
                - float(latest_balance.get("total_cur_liab", 0)),
                "depreciation_and_amortization": float(
                    latest_cashflow.get("depreciation_amort_cba", 0)
                ),
                "capital_expenditure": abs(
                    float(latest_cashflow.get("stot_outflows_inv_act", 0))
                ),
                "free_cash_flow": float(latest_cashflow.get("n_cashflow_act", 0))
                - abs(float(latest_cashflow.get("stot_outflows_inv_act", 0))),
            }

            previous_item = {
                "net_income": float(previous_income.get("n_income", 0)),
                "operating_revenue": float(previous_income.get("revenue", 0)),
                "operating_profit": float(previous_income.get("operate_profit", 0)),
                "working_capital": float(previous_balance.get("total_cur_assets", 0))
                - float(previous_balance.get("total_cur_liab", 0)),
                "depreciation_and_amortization": float(
                    previous_cashflow.get("depreciation_amort_cba", 0)
                ),
                "capital_expenditure": abs(
                    float(previous_cashflow.get("stot_outflows_inv_act", 0))
                ),
                "free_cash_flow": float(previous_cashflow.get("n_cashflow_act", 0))
                - abs(float(previous_cashflow.get("stot_outflows_inv_act", 0))),
            }

            return [current_item, previous_item]
        return [{}, {}]
    except (ImportError, Exception) as e:
        logger.warning(f"TuShare financial statements error: {str(e)}")
        return [{}, {}]


def get_tushare_market_data(tushare_code: str) -> Dict[str, Any]:
    """Get market data from TuShare"""
    try:
        import tushare as ts

        if not TUSHARE_TOKEN:
            logger.warning("TuShare token not found")
            return {}

        ts.set_token(TUSHARE_TOKEN)
        pro = ts.pro_api()

        logger.info(f"Fetching market data using TuShare for {tushare_code}")

        # Get basic market data
        daily_basic = pro.daily_basic(ts_code=tushare_code)
        if not daily_basic.empty:
            latest_data = daily_basic.iloc[0]

            # Get historical data to calculate 52-week high and low
            today = datetime.now()
            start_date = (today - timedelta(days=365)).strftime("%Y%m%d")
            end_date = today.strftime("%Y%m%d")

            hist_data = pro.daily(
                ts_code=tushare_code, start_date=start_date, end_date=end_date
            )

            if not hist_data.empty:
                week_high = hist_data["high"].max()
                week_low = hist_data["low"].min()
                avg_volume = hist_data.head(30)["vol"].mean()
            else:
                week_high = latest_data.get("high", 0)
                week_low = latest_data.get("low", 0)
                avg_volume = latest_data.get("vol", 0)

            return {
                "market_cap": float(latest_data.get("total_mv", 0)),  # Total market cap
                "volume": float(latest_data.get("vol", 0)),  # Volume
                "average_volume": float(avg_volume),  # Average volume
                "fifty_two_week_high": float(week_high),  # 52-week high
                "fifty_two_week_low": float(week_low),  # 52-week low
            }
        return {}
    except (ImportError, Exception) as e:
        logger.warning(f"TuShare market data error: {str(e)}")
        return {}
