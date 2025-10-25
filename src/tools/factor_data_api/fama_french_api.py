"""
Fama-French Factor API - Provides calculation and retrieval functionality for Fama-French three-factor model related data
"""

import os
import traceback
from datetime import datetime, timedelta
from typing import Dict

import numpy as np
import pandas as pd

from .base import logger
from .market_data_api import _generate_mock_market_returns, get_market_returns
from .risk_free_api import _generate_mock_risk_free_rate, get_risk_free_rate


def calculate_fama_french_factors_tushare(
    start_date: str, end_date: str, freq: str = "W"
) -> Dict[str, pd.Series]:
    """
    Calculate Fama-French three-factor data using TuShare

    Args:
        start_date: Start date, format: YYYY-MM-DD or YYYYMMDD
        end_date: End date, format: YYYY-MM-DD or YYYYMMDD
        freq: Data frequency, 'D' for daily, 'W' for weekly, 'M' for monthly

    Returns:
        Dictionary containing market risk premium, SMB, HML factors
    """
    logger.info(
        f"Calculating Fama-French three-factor data using TuShare: {start_date} to {end_date}, frequency: {freq}"
    )

    try:
        import tushare as ts

        token = os.environ.get("TUSHARE_TOKEN", "")
        if not token:
            logger.warning("TUSHARE_TOKEN environment variable not found")
            raise ValueError("TuShare token not available")

        ts.set_token(token)
        pro = ts.pro_api()

        # Process date format
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

        # 1. Get market index data as market returns (using CSI 300 index)
        logger.info("Getting market index data...")

        if freq == "W":
            market_df = pro.index_weekly(
                ts_code="000300.SH", start_date=start_date, end_date=end_date
            )
        elif freq == "M":
            market_df = pro.index_monthly(
                ts_code="000300.SH", start_date=start_date, end_date=end_date
            )
        else:  # Daily frequency
            market_df = pro.index_daily(
                ts_code="000300.SH", start_date=start_date, end_date=end_date
            )

        if market_df.empty:
            logger.warning(
                "Unable to get market index data, trying daily frequency data and resampling"
            )
            market_df = pro.index_daily(
                ts_code="000300.SH", start_date=start_date, end_date=end_date
            )
            if market_df.empty:
                logger.warning(
                    "Still unable to get market data, will use simulated data"
                )
                return _generate_mock_fama_french_factors(start_date, end_date, freq)

        market_df["trade_date"] = pd.to_datetime(market_df["trade_date"])
        market_df = market_df.sort_values("trade_date")
        market_df = market_df.set_index("trade_date")

        # If resampling is needed
        if freq == "W" and not market_df.empty and market_df.index.freqstr != "W-FRI":
            close_price = market_df["close"].resample("W-FRI").last()
        elif freq == "M" and not market_df.empty and market_df.index.freqstr != "M":
            close_price = market_df["close"].resample("M").last()
        else:
            close_price = market_df["close"]

        market_return = close_price.pct_change().dropna()

        # 2. Generate date list for calculating SMB and HML (quarterly)
        portfolio_dates = (
            pd.date_range(
                start=pd.to_datetime(start_date, format="%Y%m%d"),
                end=pd.to_datetime(end_date, format="%Y%m%d"),
                freq="QE",  # Quarter end
            )
            .strftime("%Y%m%d")
            .tolist()
        )

        # 3. Initialize factor data frame
        factors_weekly = pd.DataFrame(index=market_return.index)
        factors_weekly["SMB"] = np.nan
        factors_weekly["HML"] = np.nan

        # 4. Calculate SMB and HML factors for each date
        for date in portfolio_dates:
            logger.info(f"Calculating SMB and HML factors for {date}...")

            # Get market cap and PB data for all stocks on that day
            try:
                daily_basic = pro.daily_basic(
                    trade_date=date, fields="ts_code,total_mv,pb,close"
                )
                if daily_basic.empty:
                    logger.info(f"No data for date {date}, skipping")
                    continue
            except Exception as e:
                logger.warning(f"Error getting basic data for {date}: {e}")
                continue

            # Filter out stocks with negative or zero PB
            daily_basic = daily_basic[daily_basic["pb"] > 0]

            # Divide into large and small groups by market cap
            daily_basic["size_group"] = pd.qcut(
                daily_basic["total_mv"], q=2, labels=["small", "big"]
            )

            # Divide into three groups by PB inverse (B/M)
            daily_basic["bm"] = 1 / daily_basic["pb"]  # Calculate B/M ratio
            daily_basic["bm_group"] = pd.qcut(
                daily_basic["bm"], q=3, labels=["low", "medium", "high"]
            )

            # Form six portfolios
            portfolio_groups = {}
            for size in ["small", "big"]:
                for bm in ["low", "medium", "high"]:
                    portfolio_groups[f"{size}_{bm}"] = daily_basic[
                        (daily_basic["size_group"] == size)
                        & (daily_basic["bm_group"] == bm)
                    ]["ts_code"].tolist()

            # Calculate returns for each portfolio
            portfolio_returns = {}

            for port_name, stocks in portfolio_groups.items():
                if not stocks:
                    portfolio_returns[port_name] = 0
                    continue

                # Limit number of stocks
                sample_stocks = stocks[:20] if len(stocks) > 20 else stocks

                # Calculate portfolio returns
                port_returns = []
                for stock in sample_stocks:
                    try:
                        # Get stock return data
                        if freq == "W":
                            stock_data = pro.weekly(
                                ts_code=stock,
                                start_date=date,
                                end_date=end_date,
                                fields="ts_code,trade_date,close",
                            )
                        elif freq == "M":
                            stock_data = pro.monthly(
                                ts_code=stock,
                                start_date=date,
                                end_date=end_date,
                                fields="ts_code,trade_date,close",
                            )
                        else:  # Daily frequency
                            stock_data = pro.daily(
                                ts_code=stock,
                                start_date=date,
                                end_date=end_date,
                                fields="ts_code,trade_date,close",
                            )

                        if not stock_data.empty and len(stock_data) > 1:
                            stock_data["trade_date"] = pd.to_datetime(
                                stock_data["trade_date"]
                            )
                            stock_data = stock_data.sort_values("trade_date")
                            stock_data["return"] = stock_data["close"].pct_change()
                            stock_data = stock_data.dropna()

                            if not stock_data.empty:
                                port_returns.append(
                                    stock_data[["trade_date", "return"]].set_index(
                                        "trade_date"
                                    )
                                )
                    except Exception as e:
                        logger.debug(f"Error processing stock {stock}: {e}")
                        continue

                if port_returns:
                    try:
                        # Merge portfolio returns
                        all_returns = pd.concat(port_returns, axis=1)
                        portfolio_weekly_return = all_returns.mean(axis=1)

                        # Add portfolio returns to factors_weekly
                        for week_date, ret in portfolio_weekly_return.items():
                            # Find nearest Friday or month end
                            if freq == "W":
                                period_end = week_date + pd.Timedelta(
                                    days=(4 - week_date.weekday()) % 7
                                )
                            elif freq == "M":
                                period_end = pd.date_range(
                                    week_date, periods=2, freq="M"
                                )[0]
                            else:
                                period_end = week_date

                            if period_end in factors_weekly.index:
                                if port_name not in factors_weekly.columns:
                                    factors_weekly[port_name] = np.nan
                                factors_weekly.loc[period_end, port_name] = ret
                    except Exception as e:
                        logger.warning(f"Error processing portfolio {port_name}: {e}")
                        continue

            # 5. Calculate SMB and HML factors
            for idx in factors_weekly.index:
                # Check if all required portfolio data is available
                port_names = [
                    "small_low",
                    "small_medium",
                    "small_high",
                    "big_low",
                    "big_medium",
                    "big_high",
                ]
                if all(port in factors_weekly.columns for port in port_names):
                    try:
                        # Calculate SMB
                        small_avg = (
                            factors_weekly.loc[idx, "small_low"]
                            + factors_weekly.loc[idx, "small_medium"]
                            + factors_weekly.loc[idx, "small_high"]
                        ) / 3

                        big_avg = (
                            factors_weekly.loc[idx, "big_low"]
                            + factors_weekly.loc[idx, "big_medium"]
                            + factors_weekly.loc[idx, "big_high"]
                        ) / 3

                        factors_weekly.loc[idx, "SMB"] = small_avg - big_avg

                        # Calculate HML
                        high_avg = (
                            factors_weekly.loc[idx, "small_high"]
                            + factors_weekly.loc[idx, "big_high"]
                        ) / 2

                        low_avg = (
                            factors_weekly.loc[idx, "small_low"]
                            + factors_weekly.loc[idx, "big_low"]
                        ) / 2

                        factors_weekly.loc[idx, "HML"] = high_avg - low_avg
                    except Exception as e:
                        logger.debug(f"Error calculating factors for {idx}: {e}")
                        continue

        # 6. Get risk-free rate
        rf_series = get_risk_free_rate(
            start_date=start_date, end_date=end_date, freq=freq
        )

        # 7. Final data integration
        final_factors = pd.DataFrame(
            {
                "MKT": market_return,
                "SMB": factors_weekly["SMB"],
                "HML": factors_weekly["HML"],
                "RF": rf_series,
            }
        )

        # Align indices
        common_index = market_return.index.intersection(rf_series.index)
        final_factors = final_factors.loc[common_index]

        # Use forward fill to handle missing values
        final_factors = final_factors.ffill()

        # Calculate market risk premium (Mkt-RF)
        final_factors["MKT_RF"] = final_factors["MKT"] - final_factors["RF"]

        # Convert results to dictionary
        factors_dict = {
            "market_returns": final_factors["MKT"],
            "market_excess_returns": final_factors["MKT_RF"],
            "smb": final_factors["SMB"],
            "hml": final_factors["HML"],
            "risk_free_rate": final_factors["RF"],
        }

        # Handle NaN values
        for key in factors_dict:
            factors_dict[key] = factors_dict[key].fillna(0)

        logger.info(
            f"Successfully calculated Fama-French three factors, {len(factors_dict['market_returns'])} records"
        )
        return factors_dict

    except Exception as e:
        logger.error(f"TuShare Fama-French three-factor calculation failed: {e}")
        logger.error(traceback.format_exc())
        return _generate_mock_fama_french_factors(start_date, end_date, freq)


def get_fama_french_factors(
    start_date: str = None,
    end_date: str = None,
    freq: str = "D",
    use_cache: bool = True,
) -> Dict[str, pd.Series]:
    """
    Get Fama-French three-factor model factor data

    Args:
        start_date: Start date, format: YYYY-MM-DD
        end_date: End date, format: YYYY-MM-DD
        freq: Data frequency, 'D' for daily, 'W' for weekly, 'M' for monthly
        use_cache: Whether to use cached data

    Returns:
        Dictionary containing market risk premium, SMB, HML factors
    """
    logger.info(
        f"Getting Fama-French three-factor data: {start_date} to {end_date}, frequency: {freq}"
    )

    # Cache file path
    cache_dir = os.path.join(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        ),
        "data",
        "cache",
    )
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"fama_french_{freq}.csv")

    # Check cache
    if use_cache and os.path.exists(cache_file):
        try:
            cache_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            logger.info(
                f"Successfully loaded Fama-French factor data from cache: {len(cache_data)} records"
            )

            # Filter date range
            if start_date:
                start_date = pd.to_datetime(start_date)
                cache_data = cache_data[cache_data.index >= start_date]
            if end_date:
                end_date = pd.to_datetime(end_date)
                cache_data = cache_data[cache_data.index <= end_date]

            if not cache_data.empty:
                # Convert to dictionary
                factors_dict = {
                    "market_returns": cache_data["MKT"],
                    "market_excess_returns": cache_data["MKT_RF"],
                    "smb": cache_data["SMB"],
                    "hml": cache_data["HML"],
                    "risk_free_rate": cache_data["RF"],
                }
                return factors_dict
        except Exception as e:
            logger.warning(f"Failed to load Fama-French factor data from cache: {e}")

    try:
        # Try to calculate Fama-French factors using TuShare
        try:
            import tushare as ts

            if os.environ.get("TUSHARE_TOKEN", ""):
                factors_dict = calculate_fama_french_factors_tushare(
                    start_date, end_date, freq
                )

                # Save to cache
                try:
                    # Convert to DataFrame for saving
                    factors_df = pd.DataFrame(
                        {
                            "MKT": factors_dict["market_returns"],
                            "MKT_RF": factors_dict["market_excess_returns"],
                            "SMB": factors_dict["smb"],
                            "HML": factors_dict["hml"],
                            "RF": factors_dict["risk_free_rate"],
                        }
                    )
                    factors_df.to_csv(cache_file)
                    logger.info(f"Fama-French factor data saved to cache: {cache_file}")
                except Exception as e:
                    logger.warning(
                        f"Failed to save Fama-French factor data to cache: {e}"
                    )

                return factors_dict
            else:
                logger.warning(
                    "TuShare token not found, cannot use TuShare to calculate factors"
                )
        except ImportError:
            logger.warning(
                "tushare library not found, cannot use TuShare to calculate factors"
            )

        # If TuShare calculation fails, use simulated data
        return _generate_mock_fama_french_factors(start_date, end_date, freq)

    except Exception as e:
        logger.error(f"Error getting Fama-French factors: {e}")
        logger.error(traceback.format_exc())
        return _generate_mock_fama_french_factors(start_date, end_date, freq)


def _generate_mock_fama_french_factors(
    start_date: str, end_date: str, freq: str = "D"
) -> Dict[str, pd.Series]:
    """
    Generate simulated Fama-French three-factor data

    Args:
        start_date: Start date
        end_date: End date
        freq: Data frequency, 'D' for daily, 'W' for weekly, 'M' for monthly

    Returns:
        Dictionary containing market risk premium, SMB, HML factors
    """
    logger.info(
        f"Generating simulated Fama-French three-factor data, frequency: {freq}"
    )

    # Process date parameters
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

    # Get market returns
    market_returns = get_market_returns(
        start_date=start_date, end_date=end_date, freq=freq
    )

    # If market returns is empty, create a simulated market return
    if market_returns.empty:
        market_returns = _generate_mock_market_returns(start_date, end_date, freq)

    # Use market returns index
    date_range = market_returns.index

    # Generate simulated SMB and HML factors
    np.random.seed(42)  # Set random seed for reproducibility

    # SMB factor, small mean, moderate volatility
    # Adjust parameters based on frequency
    if freq == "W":
        smb_mean, smb_std = 0.0005, 0.01
        hml_mean, hml_std = 0.0008, 0.015
    elif freq == "M":
        smb_mean, smb_std = 0.002, 0.02
        hml_mean, hml_std = 0.003, 0.025
    else:  # Daily frequency
        smb_mean, smb_std = 0.0001, 0.005
        hml_mean, hml_std = 0.0002, 0.006

    smb = pd.Series(
        np.random.normal(smb_mean, smb_std, len(date_range)), index=date_range
    )

    # HML factor, moderate mean, high volatility
    hml = pd.Series(
        np.random.normal(hml_mean, hml_std, len(date_range)), index=date_range
    )

    # Get risk-free rate
    risk_free = get_risk_free_rate(start_date=start_date, end_date=end_date, freq=freq)

    # If risk-free rate is empty, create a simulated risk-free rate
    if risk_free.empty:
        risk_free = _generate_mock_risk_free_rate(start_date, end_date, freq)

    # Align indices of all data
    common_index = date_range
    if len(risk_free) > 0:
        common_index = date_range.intersection(risk_free.index)

    # Calculate market risk premium
    market_rf = market_returns.reindex(common_index) - risk_free.reindex(common_index)

    # Create result dictionary
    factors = {
        "market_returns": market_returns.reindex(common_index),
        "market_excess_returns": market_rf,
        "smb": smb.reindex(common_index),
        "hml": hml.reindex(common_index),
        "risk_free_rate": risk_free.reindex(common_index),
    }

    # Fill missing values with 0
    for key in factors:
        factors[key] = factors[key].fillna(0)

    return factors
