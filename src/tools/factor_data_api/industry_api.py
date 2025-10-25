"""
Industry Data API - Provides functionality for obtaining and processing industry-related data
"""

import traceback
from typing import List, Union

import pandas as pd

from .base import logger
from .market_data_api import get_index_data


def get_industry_index_returns(
    industry_codes: Union[str, List[str]],
    start_date: str = None,
    end_date: str = None,
    freq: str = "D",
) -> pd.DataFrame:
    """
    Get industry index return data

    Args:
        industry_codes: Industry index code or code list, e.g. "801780" (banking)
        start_date: Start date, format: YYYY-MM-DD
        end_date: End date, format: YYYY-MM-DD
        freq: Data frequency, 'D' for daily, 'W' for weekly, 'M' for monthly

    Returns:
        DataFrame containing multiple industry index returns
    """
    logger.info(
        f"Getting industry index returns: {start_date} to {end_date}, frequency: {freq}"
    )

    # Convert to list
    if isinstance(industry_codes, str):
        industry_codes = [industry_codes]

    # Define industry name mapping
    industry_names = {
        "801010": "农林牧渔",
        "801020": "采掘",
        "801030": "化工",
        "801040": "钢铁",
        "801050": "有色金属",
        "801080": "电子",
        "801110": "家用电器",
        "801120": "食品饮料",
        "801130": "纺织服装",
        "801140": "轻工制造",
        "801150": "医药生物",
        "801160": "公用事业",
        "801170": "交通运输",
        "801180": "房地产",
        "801200": "商业贸易",
        "801210": "休闲服务",
        "801230": "综合",
        "801710": "建筑材料",
        "801720": "建筑装饰",
        "801730": "电气设备",
        "801740": "国防军工",
        "801750": "计算机",
        "801760": "传媒",
        "801770": "通信",
        "801780": "银行",
        "801790": "非银金融",
        "801880": "汽车",
        "801890": "机械设备",
        "801950": "煤炭",
        "801960": "石油石化",
        "801970": "环保",
        "801980": "美容护理",
    }

    # Get index return data
    returns_dict = {}

    for code in industry_codes:
        try:
            # Get index data
            index_data = get_index_data(code, None, start_date, end_date, freq)

            if not index_data.empty:
                # Ensure date column is date type
                if "date" in index_data.columns:
                    index_data["date"] = pd.to_datetime(index_data["date"])
                    index_data = index_data.set_index("date")
                elif "trade_date" in index_data.columns:
                    index_data["trade_date"] = pd.to_datetime(index_data["trade_date"])
                    index_data = index_data.set_index("trade_date")

                # Calculate returns
                if "close" in index_data.columns:
                    returns = index_data["close"].pct_change().dropna()

                    # Get industry name
                    industry_name = industry_names.get(code, code)

                    returns_dict[industry_name] = returns
                    logger.info(
                        f"Successfully obtained industry {industry_name}({code}) return data: {len(returns)} records"
                    )
                else:
                    logger.warning(
                        f"Industry index {code} data does not contain close column"
                    )
            else:
                logger.warning(f"Unable to get industry index {code} data")

        except Exception as e:
            logger.error(f"Error getting industry index {code} returns: {e}")
            logger.error(traceback.format_exc())

    # If no data was obtained, return empty DataFrame
    if not returns_dict:
        logger.warning("No industry index return data obtained")
        return pd.DataFrame()

    # Merge into DataFrame
    returns_df = pd.DataFrame(returns_dict)

    return returns_df


def get_industry_rotation_factors(
    start_date: str = None, end_date: str = None, freq: str = "W"
) -> pd.DataFrame:
    """
    Get industry rotation factor data (including industry momentum, valuation, growth, etc.)

    Args:
        start_date: Start date, format: YYYY-MM-DD
        end_date: End date, format: YYYY-MM-DD
        freq: Data frequency, recommended 'W' for weekly or 'M' for monthly

    Returns:
        DataFrame containing industry rotation factors
    """
    logger.info(
        f"Getting industry rotation factor data: {start_date} to {end_date}, frequency: {freq}"
    )

    try:
        # 1. Get main industry index codes
        main_industries = [
            "801010",
            "801020",
            "801030",
            "801040",
            "801050",
            "801080",
            "801110",
            "801120",
            "801150",
            "801180",
            "801730",
            "801750",
            "801760",
            "801770",
            "801780",
            "801790",
            "801880",
            "801890",
        ]

        # 2. Get industry return data
        industry_returns = get_industry_index_returns(
            main_industries, start_date, end_date, freq
        )

        if industry_returns.empty:
            logger.warning("Unable to get industry return data")
            return pd.DataFrame()

        # 3. Calculate industry momentum factors
        factors_df = pd.DataFrame(index=industry_returns.index)

        # Calculate momentum factors for each industry (past 1/3/6 months returns)
        for industry in industry_returns.columns:
            # Past 1 month momentum
            factors_df[f"{industry}_MOM_1M"] = (
                industry_returns[industry].rolling(window=4 if freq == "W" else 1).sum()
            )

            # Past 3 months momentum
            factors_df[f"{industry}_MOM_3M"] = (
                industry_returns[industry]
                .rolling(window=12 if freq == "W" else 3)
                .sum()
            )

            # Past 6 months momentum
            factors_df[f"{industry}_MOM_6M"] = (
                industry_returns[industry]
                .rolling(window=24 if freq == "W" else 6)
                .sum()
            )

        # 4. Try to get industry valuation data (such as PE, PB, etc.)
        # This part may need to be constructed or use simulated data if TuShare and AKShare don't provide it directly

        logger.info(
            f"Successfully calculated industry rotation factors: {len(factors_df)} records"
        )
        return factors_df

    except Exception as e:
        logger.error(f"Error getting industry rotation factors: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()


def get_sector_index_returns(
    start_date: str = None, end_date: str = None, freq: str = "D"
) -> pd.DataFrame:
    """
    Get main sector index return data (Shanghai Composite, Shenzhen Component, ChiNext, SME Board, etc.)

    Args:
        start_date: Start date, format: YYYY-MM-DD
        end_date: End date, format: YYYY-MM-DD
        freq: Data frequency, 'D' for daily, 'W' for weekly, 'M' for monthly

    Returns:
        DataFrame containing main sector index returns
    """
    logger.info(
        f"Getting main sector index returns: {start_date} to {end_date}, frequency: {freq}"
    )

    # Define main sector indices
    sector_indices = {
        "sh000001": "上证指数",
        "sz399001": "深证成指",
        "sz399006": "创业板指",
        "sz399005": "中小板指",
        "sh000300": "沪深300",
        "sh000905": "中证500",
        "sh000016": "上证50",
        "sh000852": "中证1000",
    }

    # Get returns for each index
    returns_dict = {}

    for code, name in sector_indices.items():
        try:
            # Process index code
            if code.startswith("sh"):
                index_code = code[2:]
            elif code.startswith("sz"):
                index_code = code[2:]
            else:
                index_code = code

            # Get index data
            index_data = get_index_data(index_code, None, start_date, end_date, freq)

            if not index_data.empty:
                # Ensure date column is date type
                if "date" in index_data.columns:
                    index_data["date"] = pd.to_datetime(index_data["date"])
                    index_data = index_data.set_index("date")
                elif "trade_date" in index_data.columns:
                    index_data["trade_date"] = pd.to_datetime(index_data["trade_date"])
                    index_data = index_data.set_index("trade_date")

                # Calculate returns
                if "close" in index_data.columns:
                    returns = index_data["close"].pct_change().dropna()
                    returns_dict[name] = returns
                    logger.info(
                        f"Successfully obtained {name} return data: {len(returns)} records"
                    )
                else:
                    logger.warning(f"Index {code} data does not contain close column")
            else:
                logger.warning(f"Unable to get index {code} data")

        except Exception as e:
            logger.error(f"Error getting index {code} returns: {e}")
            logger.error(traceback.format_exc())

    # If no data was obtained, return empty DataFrame
    if not returns_dict:
        logger.warning("No sector index return data obtained")
        return pd.DataFrame()

    # Merge into DataFrame
    returns_df = pd.DataFrame(returns_dict)

    return returns_df


def get_style_index_returns(
    start_date: str = None, end_date: str = None, freq: str = "D"
) -> pd.DataFrame:
    """
    Get style index return data (large-cap growth, small-cap value, etc.)

    Args:
        start_date: Start date, format: YYYY-MM-DD
        end_date: End date, format: YYYY-MM-DD
        freq: Data frequency, 'D' for daily, 'W' for weekly, 'M' for monthly

    Returns:
        DataFrame containing style index returns
    """
    logger.info(
        f"Getting style index returns: {start_date} to {end_date}, frequency: {freq}"
    )

    # Define style indices
    style_indices = {
        "sh000919": "300价值",
        "sh000918": "300成长",
        "sh000922": "中证红利",
        "sh000925": "基本面50",
        "sh000978": "医药100",
        "sh000991": "全指医药",
    }

    # 获取各指数收益率
    returns_dict = {}

    for code, name in style_indices.items():
        try:
            # Process index code
            if code.startswith("sh"):
                index_code = code[2:]
            elif code.startswith("sz"):
                index_code = code[2:]
            else:
                index_code = code

            # Get index data
            index_data = get_index_data(index_code, None, start_date, end_date, freq)

            if not index_data.empty:
                # Ensure date column is date type
                if "date" in index_data.columns:
                    index_data["date"] = pd.to_datetime(index_data["date"])
                    index_data = index_data.set_index("date")
                elif "trade_date" in index_data.columns:
                    index_data["trade_date"] = pd.to_datetime(index_data["trade_date"])
                    index_data = index_data.set_index("trade_date")

                # Calculate returns
                if "close" in index_data.columns:
                    returns = index_data["close"].pct_change().dropna()
                    returns_dict[name] = returns
                    logger.info(
                        f"Successfully obtained {name} return data: {len(returns)} records"
                    )
                else:
                    logger.warning(f"Index {code} data does not contain close column")
            else:
                logger.warning(f"Unable to get index {code} data")

        except Exception as e:
            logger.error(f"Error getting index {code} returns: {e}")
            logger.error(traceback.format_exc())

    # If no data was obtained, return empty DataFrame
    if not returns_dict:
        logger.warning("No style index return data obtained")
        return pd.DataFrame()

    # Merge into DataFrame
    returns_df = pd.DataFrame(returns_dict)

    return returns_df
