"""
Market Data API - Provides market-related data acquisition and processing functionality
"""

import os
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from .base import data_api, logger


def get_market_returns(
    index_code: str = "000300",
    start_date: str = None,
    end_date: str = None,
    freq: str = "D",
) -> pd.Series:
    """
    Get market return data (default uses CSI 300 index)

    Args:
        index_code: Index code, default is CSI 300
        start_date: Start date, format: YYYY-MM-DD
        end_date: End date, format: YYYY-MM-DD
        freq: Data frequency, 'D' for daily, 'W' for weekly, 'M' for monthly

    Returns:
        Series containing market returns, indexed by date
    """
    logger.info(
        f"Getting market return data: {start_date} to {end_date}, index code: {index_code}, frequency: {freq}"
    )

    try:
        # Try using TuShare API
        try:
            import tushare as ts

            token = os.environ.get("TUSHARE_TOKEN", "")
            if not token:
                logger.warning("TUSHARE_TOKEN environment variable not found")
                raise ValueError("TuShare token not available")

            ts.set_token(token)
            pro = ts.pro_api()

            # Handle date format
            if not end_date:
                end_date = datetime.now().strftime("%Y%m%d")
            else:
                if "-" in end_date:
                    end_date = end_date.replace("-", "")

            if not start_date:
                # Default to get one year of data
                start_date = (
                    datetime.strptime(end_date, "%Y%m%d") - timedelta(days=365)
                ).strftime("%Y%m%d")
            else:
                if "-" in start_date:
                    start_date = start_date.replace("-", "")

            # Process index code
            if not index_code.endswith((".SH", ".SZ", ".BJ")):
                if index_code.startswith("0"):
                    index_code = f"{index_code}.SH"
                elif index_code.startswith(("1", "3")):
                    index_code = f"{index_code}.SZ"
                else:
                    index_code = f"{index_code}.SH"

            # Get index data
            index_data = None
            try:
                if freq == "W":
                    index_data = pro.index_weekly(
                        ts_code=index_code, start_date=start_date, end_date=end_date
                    )
                elif freq == "M":
                    index_data = pro.index_monthly(
                        ts_code=index_code, start_date=start_date, end_date=end_date
                    )
                else:  # Daily frequency
                    index_data = pro.index_daily(
                        ts_code=index_code, start_date=start_date, end_date=end_date
                    )
            except Exception as e:
                logger.warning(f"Failed to get index data using TuShare: {e}")

            # If TuShare specific frequency API call fails, try to get daily data and resample
            if index_data is None or index_data.empty:
                try:
                    logger.info("Trying to get daily data and resample")
                    index_data = pro.index_daily(
                        ts_code=index_code, start_date=start_date, end_date=end_date
                    )
                except Exception as e:
                    logger.warning(f"Failed to get daily index data using TuShare: {e}")

            if index_data is not None and not index_data.empty:
                # Process data
                index_data["trade_date"] = pd.to_datetime(index_data["trade_date"])
                index_data = index_data.sort_values("trade_date")
                index_data = index_data.set_index("trade_date")

                # If resampling is needed
                if freq != "D" and "close" in index_data.columns:
                    if freq == "W":
                        close_price = index_data["close"].resample("W-FRI").last()
                    elif freq == "M":
                        close_price = index_data["close"].resample("M").last()
                    else:
                        close_price = index_data["close"]
                else:
                    close_price = index_data["close"]

                # Calculate returns
                market_returns = close_price.pct_change().dropna()

                logger.info(
                    f"Successfully got market return data using TuShare: {len(market_returns)} records"
                )
                return market_returns
            else:
                logger.warning("TuShare returned empty index data, trying AKShare")
        except Exception as e:
            logger.warning(f"Failed to get market returns using TuShare: {e}")
            logger.warning("Trying AKShare")

        # Try using AKShare
        try:
            import akshare as ak

            # Process index code
            if index_code.endswith((".SH", ".SZ", ".BJ")):
                index_code = index_code[:-3]

            # Add prefix to index code
            if not index_code.startswith(("sh", "sz", "bj")):
                if index_code.startswith("0"):
                    ak_index_code = f"sh{index_code}"
                elif index_code.startswith(("1", "3")):
                    ak_index_code = f"sz{index_code}"
                else:
                    ak_index_code = f"sh{index_code}"
            else:
                ak_index_code = index_code

            # Handle date format
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if not start_date:
                start_date = (
                    datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365)
                ).strftime("%Y-%m-%d")

            # Ensure date format is YYYY-MM-DD
            if len(start_date) == 8:
                start_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
            if len(end_date) == 8:
                end_date = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"

            # Get index data
            if freq == "W":
                period = "weekly"
            elif freq == "M":
                period = "monthly"
            else:
                period = "daily"

            index_data = ak.stock_zh_index_daily(symbol=ak_index_code, period=period)

            if not index_data.empty:
                # Process data
                index_data["date"] = pd.to_datetime(index_data["date"])
                index_data = index_data.sort_values("date")
                index_data = index_data.set_index("date")

                # Filter date range
                index_data = index_data[index_data.index >= pd.to_datetime(start_date)]
                index_data = index_data[index_data.index <= pd.to_datetime(end_date)]

                # Calculate returns
                market_returns = index_data["close"].pct_change().dropna()

                logger.info(
                    f"Successfully got market return data using AKShare: {len(market_returns)} records"
                )
                return market_returns
            else:
                logger.warning("AKShare returned empty index data, using mock data")
                return _generate_mock_market_returns(start_date, end_date, freq)
        except Exception as e:
            logger.warning(f"Failed to get market returns using AKShare: {e}")
            logger.warning("Will use mock data")

    except Exception as e:
        logger.error(f"Error getting market returns: {e}")
        logger.error(traceback.format_exc())

    # If all attempts fail, use mock data
    return _generate_mock_market_returns(start_date, end_date, freq)


def _generate_mock_market_returns(
    start_date: str, end_date: str, freq: str = "D"
) -> pd.Series:
    """
    Generate mock market return data

    Args:
        start_date: Start date
        end_date: End date
        freq: Data frequency, 'D' for daily, 'W' for weekly, 'M' for monthly

    Returns:
        Mock market return Series
    """
    logger.info(f"Generating mock market return data, frequency: {freq}")

    # Handle date parameters
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if not start_date:
        # Default to generate one year of data
        start_date = (
            datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365)
        ).strftime("%Y-%m-%d")

    # Ensure consistent date format
    if isinstance(start_date, str) and len(start_date) == 8:
        start_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
    if isinstance(end_date, str) and len(end_date) == 8:
        end_date = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"

    # Generate date range based on frequency
    if freq == "W":
        date_range = pd.date_range(start=start_date, end=end_date, freq="W-FRI")
        # Adjust mean and std for weekly
        mean_return = 0.002
        std_return = 0.02
    elif freq == "M":
        date_range = pd.date_range(start=start_date, end=end_date, freq="M")
        # Adjust mean and std for monthly
        mean_return = 0.008
        std_return = 0.04
    else:  # Daily frequency
        date_range = pd.date_range(start=start_date, end=end_date, freq="B")
        # Adjust mean and std for daily
        mean_return = 0.0005
        std_return = 0.01

    # Generate mock market returns
    np.random.seed(42)  # Set random seed for reproducibility
    returns = np.random.normal(mean_return, std_return, len(date_range))

    # Create Series
    market_returns = pd.Series(returns, index=date_range)

    return market_returns


def get_stock_returns(
    symbols: Union[str, list],
    start_date: str = None,
    end_date: str = None,
    freq: str = "D",
) -> Dict[str, pd.Series]:
    """
    Get return data for one or more stocks

    Args:
        symbols: Single stock code or list of stock codes
        start_date: Start date, format: YYYY-MM-DD
        end_date: End date, format: YYYY-MM-DD
        freq: Data frequency, 'D' for daily, 'W' for weekly, 'M' for monthly

    Returns:
        Dictionary containing stock returns, key is stock code, value is return Series
    """
    logger.info(
        f"Getting stock return data: {start_date} to {end_date}, frequency: {freq}"
    )

    # Convert to list type
    if isinstance(symbols, str):
        symbols = [symbols]

    returns_dict = {}

    for symbol in symbols:
        try:
            # Try using TuShare to get data
            try:
                import tushare as ts

                token = os.environ.get("TUSHARE_TOKEN", "")
                if token:
                    ts.set_token(token)
                    pro = ts.pro_api()

                    # Handle date format
                    ts_start = start_date
                    ts_end = end_date

                    if ts_start and "-" in ts_start:
                        ts_start = ts_start.replace("-", "")

                    if ts_end and "-" in ts_end:
                        ts_end = ts_end.replace("-", "")

                    # Handle stock code
                    ts_code = symbol
                    if not ts_code.endswith((".SH", ".SZ", ".BJ")):
                        if ts_code.startswith("6"):
                            ts_code = f"{ts_code}.SH"
                        elif ts_code.startswith(("0", "3")):
                            ts_code = f"{ts_code}.SZ"
                        elif ts_code.startswith("4"):
                            ts_code = f"{ts_code}.BJ"
                        else:
                            ts_code = f"{ts_code}.SH"

                    # Get data for different periods based on frequency
                    stock_data = None

                    try:
                        if freq == "W":
                            stock_data = pro.weekly(
                                ts_code=ts_code, start_date=ts_start, end_date=ts_end
                            )
                        elif freq == "M":
                            stock_data = pro.monthly(
                                ts_code=ts_code, start_date=ts_start, end_date=ts_end
                            )
                        else:  # Daily frequency
                            stock_data = pro.daily(
                                ts_code=ts_code, start_date=ts_start, end_date=ts_end
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to get stock {symbol} data using TuShare: {e}"
                        )

                    # If specific frequency API call fails, try to get daily data and resample
                    if stock_data is None or stock_data.empty:
                        try:
                            logger.info(
                                f"Trying to get daily data and resample: {ts_code}"
                            )
                            stock_data = pro.daily(
                                ts_code=ts_code, start_date=ts_start, end_date=ts_end
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to get daily stock data using TuShare: {e}"
                            )

                    if stock_data is not None and not stock_data.empty:
                        # Process data
                        stock_data["trade_date"] = pd.to_datetime(
                            stock_data["trade_date"]
                        )
                        stock_data = stock_data.sort_values("trade_date")
                        stock_data = stock_data.set_index("trade_date")

                        # If resampling is needed
                        if freq != "D" and "close" in stock_data.columns:
                            if freq == "W":
                                close_price = (
                                    stock_data["close"].resample("W-FRI").last()
                                )
                            elif freq == "M":
                                close_price = stock_data["close"].resample("M").last()
                            else:
                                close_price = stock_data["close"]
                        else:
                            close_price = stock_data["close"]

                        # Calculate returns
                        returns = close_price.pct_change().dropna()

                        if not returns.empty:
                            returns_dict[symbol] = returns
                            logger.info(
                                f"Successfully got stock {symbol} return data using TuShare: {len(returns)} records"
                            )
                            continue
            except ImportError:
                logger.warning("tushare library not found, trying AKShare")

            # Use data API to get stock price data
            prices = data_api.get_price_data(symbol, start_date, end_date)

            if prices.empty:
                logger.warning(f"Unable to get price data for stock {symbol}")
                continue

            # Ensure date column is datetime type
            if "date" in prices.columns:
                prices["date"] = pd.to_datetime(prices["date"])
                prices = prices.set_index("date")

            # Resample based on frequency
            if freq == "W" and "close" in prices.columns:
                close_price = prices["close"].resample("W-FRI").last()
            elif freq == "M" and "close" in prices.columns:
                close_price = prices["close"].resample("M").last()
            elif "close" in prices.columns:
                close_price = prices["close"]
            else:
                logger.warning(
                    f"Stock {symbol} price data does not contain close column"
                )
                continue

            # Calculate returns
            returns = close_price.pct_change().dropna()

            if not returns.empty:
                returns_dict[symbol] = returns
                logger.info(
                    f"Successfully got stock {symbol} return data: {len(returns)} records"
                )

        except Exception as e:
            logger.error(f"Error getting stock {symbol} returns: {e}")
            logger.error(traceback.format_exc())

    return returns_dict


def get_multi_stock_returns(
    symbols: list, start_date: str = None, end_date: str = None, freq: str = "D"
) -> pd.DataFrame:
    """
    Get return data for multiple stocks and merge into DataFrame

    Args:
        symbols: List of stock codes
        start_date: Start date, format: YYYY-MM-DD
        end_date: End date, format: YYYY-MM-DD
        freq: Data frequency, 'D' for daily, 'W' for weekly, 'M' for monthly

    Returns:
        DataFrame containing multiple stock returns, columns are stock codes, index is date
    """
    logger.info(
        f"Getting multi-stock return data: {start_date} to {end_date}, stock count: {len(symbols)}, frequency: {freq}"
    )

    # Get returns for each stock
    returns_dict = get_stock_returns(symbols, start_date, end_date, freq)

    # If no data is obtained, return empty DataFrame
    if not returns_dict:
        logger.warning("No stock return data obtained")
        return pd.DataFrame()

    # Merge into DataFrame
    df = pd.DataFrame(returns_dict)

    return df


def get_stock_covariance_matrix(
    symbols: list,
    start_date: str = None,
    end_date: str = None,
    method: str = "sample",
    freq: str = "D",
) -> tuple:
    """
    Calculate covariance matrix and average returns for multiple stock returns

    Args:
        symbols: List of stock codes
        start_date: Start date, format: YYYY-MM-DD
        end_date: End date, format: YYYY-MM-DD
        method: Covariance matrix estimation method, options "sample", "ewma"
        freq: Data frequency, 'D' for daily, 'W' for weekly, 'M' for monthly

    Returns:
        Tuple of (covariance matrix DataFrame, average returns Series)
    """
    logger.info(
        f"Calculating stock covariance matrix: {start_date} to {end_date}, stock count: {len(symbols)}, method: {method}"
    )

    # Get multi-stock return data
    returns_df = get_multi_stock_returns(symbols, start_date, end_date, freq)

    # If no data is obtained, return empty DataFrame
    if returns_df.empty:
        logger.warning(
            "No stock return data obtained, cannot calculate covariance matrix"
        )
        return pd.DataFrame(), pd.Series()

    try:
        # Calculate covariance matrix
        if method == "ewma":
            # Import EWMA covariance matrix estimation function
            from src.calc.covariance_estimation import estimate_covariance_ewma

            cov_matrix = estimate_covariance_ewma(returns_df)
        else:
            # Use sample covariance matrix
            cov_matrix = returns_df.cov()

        # Calculate average returns
        expected_returns = returns_df.mean()

        # Annualization (based on frequency)
        if freq == "D":
            annualization_factor = 252
        elif freq == "W":
            annualization_factor = 52
        elif freq == "M":
            annualization_factor = 12
        else:
            annualization_factor = 252

        expected_returns = expected_returns * annualization_factor
        cov_matrix = cov_matrix * annualization_factor

        return cov_matrix, expected_returns
    except Exception as e:
        logger.error(f"Error calculating covariance matrix: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame(), pd.Series()


def get_index_data(
    index_symbol: str = "000300",
    fields: list = None,
    start_date: str = None,
    end_date: str = None,
    freq: str = "D",
) -> pd.DataFrame:
    """
    Get index data

    Args:
        index_symbol: Index code, default is CSI 300
        fields: List of fields to get, if None then get all available fields
        start_date: Start date, format: YYYY-MM-DD
        end_date: End date, format: YYYY-MM-DD
        freq: Data frequency, 'D' for daily, 'W' for weekly, 'M' for monthly

    Returns:
        DataFrame containing index data
    """
    logger.info(
        f"Getting index data: {index_symbol}, {start_date} to {end_date}, frequency: {freq}"
    )

    try:
        # Try using TuShare API
        try:
            import tushare as ts

            token = os.environ.get("TUSHARE_TOKEN", "")
            if token:
                ts.set_token(token)
                pro = ts.pro_api()

                # Handle date format
                ts_start = start_date
                ts_end = end_date

                if ts_start and "-" in ts_start:
                    ts_start = ts_start.replace("-", "")

                if ts_end and "-" in ts_end:
                    ts_end = ts_end.replace("-", "")

                # Process index code
                ts_code = index_symbol
                if not ts_code.endswith((".SH", ".SZ", ".BJ")):
                    if ts_code.startswith("0"):
                        ts_code = f"{ts_code}.SH"
                    elif ts_code.startswith(("1", "3")):
                        ts_code = f"{ts_code}.SZ"
                    else:
                        ts_code = f"{ts_code}.SH"

                # Get index data
                index_data = None

                try:
                    if freq == "W":
                        index_data = pro.index_weekly(
                            ts_code=ts_code, start_date=ts_start, end_date=ts_end
                        )
                    elif freq == "M":
                        index_data = pro.index_monthly(
                            ts_code=ts_code, start_date=ts_start, end_date=ts_end
                        )
                    else:  # Daily frequency
                        index_data = pro.index_daily(
                            ts_code=ts_code, start_date=ts_start, end_date=ts_end
                        )
                except Exception as e:
                    logger.warning(f"Failed to get index data using TuShare: {e}")

                # If specific frequency API call fails, try to get daily data and resample
                if index_data is None or index_data.empty:
                    try:
                        logger.info("Trying to get daily data and resample")
                        index_data = pro.index_daily(
                            ts_code=ts_code, start_date=ts_start, end_date=ts_end
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to get daily index data using TuShare: {e}"
                        )

                if index_data is not None and not index_data.empty:
                    # Process data
                    index_data["trade_date"] = pd.to_datetime(index_data["trade_date"])
                    index_data = index_data.sort_values("trade_date")

                    # If resampling is needed
                    if freq != "D" and index_data["trade_date"].dt.freq != freq:
                        index_data = index_data.set_index("trade_date")

                        if freq == "W":
                            # Use specific resampling methods for OHLC columns
                            resampled_data = pd.DataFrame()
                            if "open" in index_data.columns:
                                resampled_data["open"] = (
                                    index_data["open"].resample("W-FRI").first()
                                )
                            if "high" in index_data.columns:
                                resampled_data["high"] = (
                                    index_data["high"].resample("W-FRI").max()
                                )
                            if "low" in index_data.columns:
                                resampled_data["low"] = (
                                    index_data["low"].resample("W-FRI").min()
                                )
                            if "close" in index_data.columns:
                                resampled_data["close"] = (
                                    index_data["close"].resample("W-FRI").last()
                                )
                            if "vol" in index_data.columns:
                                resampled_data["vol"] = (
                                    index_data["vol"].resample("W-FRI").sum()
                                )
                            if "amount" in index_data.columns:
                                resampled_data["amount"] = (
                                    index_data["amount"].resample("W-FRI").sum()
                                )

                            # Add other fields
                            for col in index_data.columns:
                                if col not in [
                                    "open",
                                    "high",
                                    "low",
                                    "close",
                                    "vol",
                                    "amount",
                                ]:
                                    resampled_data[col] = (
                                        index_data[col].resample("W-FRI").last()
                                    )

                            index_data = resampled_data.reset_index()

                        elif freq == "M":
                            # Use specific resampling methods for OHLC columns
                            resampled_data = pd.DataFrame()
                            if "open" in index_data.columns:
                                resampled_data["open"] = (
                                    index_data["open"].resample("M").first()
                                )
                            if "high" in index_data.columns:
                                resampled_data["high"] = (
                                    index_data["high"].resample("M").max()
                                )
                            if "low" in index_data.columns:
                                resampled_data["low"] = (
                                    index_data["low"].resample("M").min()
                                )
                            if "close" in index_data.columns:
                                resampled_data["close"] = (
                                    index_data["close"].resample("M").last()
                                )
                            if "vol" in index_data.columns:
                                resampled_data["vol"] = (
                                    index_data["vol"].resample("M").sum()
                                )
                            if "amount" in index_data.columns:
                                resampled_data["amount"] = (
                                    index_data["amount"].resample("M").sum()
                                )

                            # Add other fields
                            for col in index_data.columns:
                                if col not in [
                                    "open",
                                    "high",
                                    "low",
                                    "close",
                                    "vol",
                                    "amount",
                                ]:
                                    resampled_data[col] = (
                                        index_data[col].resample("M").last()
                                    )

                            index_data = resampled_data.reset_index()

                    # If fields are specified, only return specified fields
                    if fields:
                        available_fields = set(index_data.columns)
                        requested_fields = set(fields)
                        missing_fields = requested_fields - available_fields

                        if missing_fields:
                            logger.warning(
                                f"Requested fields not available: {missing_fields}"
                            )

                        selected_fields = list(requested_fields & available_fields)
                        if not selected_fields:
                            logger.warning("No available requested fields")
                            return pd.DataFrame()

                        return index_data[selected_fields]

                    logger.info(
                        f"Successfully got index data using TuShare: {len(index_data)} records"
                    )
                    return index_data
                else:
                    logger.warning("TuShare returned empty index data, trying AKShare")
        except ImportError:
            logger.warning("tushare library not found, trying AKShare")

        # Try using AKShare
        try:
            import akshare as ak

            # Process index code
            if index_symbol.endswith((".SH", ".SZ", ".BJ")):
                index_symbol = index_symbol[:-3]

            # Add prefix to index code
            if not index_symbol.startswith(("sh", "sz", "bj")):
                if index_symbol.startswith("0"):
                    ak_index_code = f"sh{index_symbol}"
                elif index_symbol.startswith(("1", "3")):
                    ak_index_code = f"sz{index_symbol}"
                else:
                    ak_index_code = f"sh{index_symbol}"
            else:
                ak_index_code = index_symbol

            # Handle date format
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if not start_date:
                start_date = (
                    datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365)
                ).strftime("%Y-%m-%d")

            # Ensure date format is YYYY-MM-DD
            if len(start_date) == 8:
                start_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
            if len(end_date) == 8:
                end_date = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"

            # Get index data
            if freq == "W":
                period = "weekly"
            elif freq == "M":
                period = "monthly"
            else:
                period = "daily"

            index_data = ak.stock_zh_index_daily(symbol=ak_index_code, period=period)

            if not index_data.empty:
                # Process date
                index_data["date"] = pd.to_datetime(index_data["date"])

                # Filter date range
                index_data = index_data[
                    (index_data["date"] >= pd.to_datetime(start_date))
                    & (index_data["date"] <= pd.to_datetime(end_date))
                ]

                # If fields are specified, only return specified fields
                if fields:
                    # Map field names
                    field_mapping = {
                        "open": "open",
                        "high": "high",
                        "low": "low",
                        "close": "close",
                        "vol": "volume",
                        "amount": "amount",
                        "change": "change",
                        "pct_chg": "pct_change",
                    }

                    # Convert field names
                    converted_fields = []
                    for field in fields:
                        if (
                            field in field_mapping
                            and field_mapping[field] in index_data.columns
                        ):
                            converted_fields.append(field_mapping[field])
                        elif field in index_data.columns:
                            converted_fields.append(field)

                    if not converted_fields:
                        logger.warning("No available requested fields")
                        return pd.DataFrame()

                    if "date" not in converted_fields:
                        converted_fields.append("date")

                    return index_data[converted_fields]

                logger.info(
                    f"Successfully got index data using AKShare: {len(index_data)} records"
                )
                return index_data
            else:
                logger.warning("AKShare returned empty index data")
                return pd.DataFrame()
        except ImportError:
            logger.warning("akshare library not found")
        except Exception as e:
            logger.warning(f"Failed to get index data using AKShare: {e}")

    except Exception as e:
        logger.error(f"Error getting index data: {e}")
        logger.error(traceback.format_exc())

    # If all attempts fail, return empty DataFrame
    logger.warning("All attempts to get index data failed")
    return pd.DataFrame()


def get_multiple_index_data(
    index_symbols: List[str],
    start_date: str = None,
    end_date: str = None,
    freq: str = "D",
) -> Dict[str, pd.DataFrame]:
    """
    Get data for multiple indices

    Args:
        index_symbols: List of index codes
        start_date: Start date, format: YYYY-MM-DD
        end_date: End date, format: YYYY-MM-DD
        freq: Data frequency, 'D' for daily, 'W' for weekly, 'M' for monthly

    Returns:
        Dictionary containing multiple index data, key is index code, value is index data DataFrame
    """
    logger.info(
        f"Getting multiple index data: {start_date} to {end_date}, index count: {len(index_symbols)}, frequency: {freq}"
    )

    index_data_dict = {}

    for symbol in index_symbols:
        try:
            index_data = get_index_data(symbol, None, start_date, end_date, freq)

            if not index_data.empty:
                index_data_dict[symbol] = index_data
                logger.info(
                    f"Successfully got index {symbol} data: {len(index_data)} records"
                )
            else:
                logger.warning(f"Unable to get index {symbol} data")

        except Exception as e:
            logger.error(f"Error getting index {symbol} data: {e}")
            logger.error(traceback.format_exc())

    return index_data_dict
