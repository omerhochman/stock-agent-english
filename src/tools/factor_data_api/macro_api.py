"""
Macroeconomic Data API - Provides functionality for obtaining and processing macroeconomic data
"""

import traceback
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .base import logger
from .risk_free_api import get_risk_free_rate


def get_macro_economic_data(
    indicator_type: str = "gdp", start_date: str = None, end_date: str = None
) -> pd.DataFrame:
    """
    Get macroeconomic indicator data

    Args:
        indicator_type: Indicator type, such as "gdp", "cpi", "interest_rate", "m2"
        start_date: Start date, format: YYYY-MM-DD
        end_date: End date, format: YYYY-MM-DD

    Returns:
        DataFrame containing macroeconomic data
    """
    logger.info(
        f"Getting macroeconomic data: {indicator_type}, {start_date} to {end_date}"
    )

    try:
        # Try using AKShare to get macroeconomic data
        try:
            import akshare as ak

            # Process date format
            if start_date and len(start_date) == 8:
                start_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
            if end_date and len(end_date) == 8:
                end_date = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"

            # Get different data based on indicator type
            if indicator_type == "gdp":
                # Get GDP data
                gdp_data = ak.macro_china_gdp()

                if not gdp_data.empty:
                    # Process data
                    gdp_data = gdp_data.rename(
                        columns={"季度": "date"}
                    )  # 季度 = quarter

                    # Convert date format
                    gdp_data["date"] = pd.to_datetime(gdp_data["date"])

                    # Filter date range
                    if start_date:
                        start_date = pd.to_datetime(start_date)
                        gdp_data = gdp_data[gdp_data["date"] >= start_date]
                    if end_date:
                        end_date = pd.to_datetime(end_date)
                        gdp_data = gdp_data[gdp_data["date"] <= end_date]

                    logger.info(
                        f"Successfully obtained GDP data: {len(gdp_data)} records"
                    )
                    return gdp_data
                else:
                    logger.warning("Unable to get GDP data")

            elif indicator_type == "cpi":
                # Get CPI data
                cpi_data = ak.macro_china_cpi()

                if not cpi_data.empty:
                    # Process data
                    cpi_data = cpi_data.rename(columns={"月份": "date"})  # 月份 = month

                    # Convert date format
                    cpi_data["date"] = pd.to_datetime(cpi_data["date"])

                    # Filter date range
                    if start_date:
                        start_date = pd.to_datetime(start_date)
                        cpi_data = cpi_data[cpi_data["date"] >= start_date]
                    if end_date:
                        end_date = pd.to_datetime(end_date)
                        cpi_data = cpi_data[cpi_data["date"] <= end_date]

                    logger.info(
                        f"Successfully obtained CPI data: {len(cpi_data)} records"
                    )
                    return cpi_data
                else:
                    logger.warning("Unable to get CPI data")

            elif indicator_type == "interest_rate":
                # Get interest rate data
                interest_data = get_risk_free_rate(
                    start_date, end_date, freq="D"
                ).to_frame(name="interest_rate")
                interest_data = interest_data.reset_index().rename(
                    columns={"index": "date"}
                )

                logger.info(
                    f"Successfully obtained interest rate data: {len(interest_data)} records"
                )
                return interest_data

            elif indicator_type == "m2":
                # Get M2 data
                m2_data = ak.macro_china_money_supply()

                if not m2_data.empty:
                    # Process data
                    m2_data = m2_data.rename(columns={"月份": "date"})  # 月份 = month

                    # Convert date format
                    m2_data["date"] = pd.to_datetime(m2_data["date"])

                    # Filter date range
                    if start_date:
                        start_date = pd.to_datetime(start_date)
                        m2_data = m2_data[m2_data["date"] >= start_date]
                    if end_date:
                        end_date = pd.to_datetime(end_date)
                        m2_data = m2_data[m2_data["date"] <= end_date]

                    logger.info(
                        f"Successfully obtained M2 data: {len(m2_data)} records"
                    )
                    return m2_data
                else:
                    logger.warning("Unable to get M2 data")

            else:
                logger.warning(f"Unsupported indicator type: {indicator_type}")

        except ImportError:
            logger.warning(
                "akshare library not found, unable to get macroeconomic data"
            )
        except Exception as e:
            logger.warning(f"Failed to get macroeconomic data using AKShare: {e}")

    except Exception as e:
        logger.error(f"Error occurred while getting macroeconomic data: {e}")
        logger.error(traceback.format_exc())

    # If all attempts fail, use mock data
    return _generate_mock_macro_data(indicator_type, start_date, end_date)


def _generate_mock_macro_data(
    indicator_type: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    Generate mock macroeconomic data

    Args:
        indicator_type: Indicator type
        start_date: Start date
        end_date: End date

    Returns:
        Mock macroeconomic data DataFrame
    """
    logger.info(f"Generating mock {indicator_type} data")

    # Process date parameters
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if not start_date:
        # Default to generate five years of data
        start_date = (
            datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365 * 5)
        ).strftime("%Y-%m-%d")

    # Ensure date format consistency
    if isinstance(start_date, str) and len(start_date) == 8:
        start_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
    if isinstance(end_date, str) and len(end_date) == 8:
        end_date = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"

    # Generate different frequency date ranges based on indicator type
    if indicator_type == "gdp":
        # GDP data is quarterly data
        date_range = pd.date_range(start=start_date, end=end_date, freq="QE")
    elif indicator_type in ["cpi", "m2"]:
        # CPI and M2 data are monthly data
        date_range = pd.date_range(start=start_date, end=end_date, freq="M")
    else:
        # Other data defaults to monthly data
        date_range = pd.date_range(start=start_date, end=end_date, freq="M")

    # Generate mock data
    np.random.seed(42)  # Set random seed for reproducibility

    if indicator_type == "gdp":
        # Create GDP data
        # Base GDP value (unit: 100 million yuan)
        base_gdp = 100000
        # Growth rate range (6%-8%)
        growth_rates = np.random.uniform(0.06, 0.08, len(date_range))

        # Calculate cumulative GDP
        gdp_values = []
        current_gdp = base_gdp
        for rate in growth_rates:
            current_gdp *= 1 + rate
            gdp_values.append(current_gdp)

        # Create DataFrame
        mock_data = pd.DataFrame(
            {
                "date": date_range,
                "GDP_Quarterly": gdp_values,
                "GDP_Cumulative": np.cumsum(gdp_values),
                "GDP_Year_Over_Year_Growth": growth_rates * 100,
            }
        )

    elif indicator_type == "cpi":
        # Create CPI data
        # CPI year-over-year growth range (1%-4%)
        cpi_yoy = np.random.uniform(0.01, 0.04, len(date_range)) * 100

        # Create DataFrame
        mock_data = pd.DataFrame(
            {
                "date": date_range,
                "National_Year_Over_Year": cpi_yoy,
                "National_Month_Over_Month": np.random.uniform(
                    -0.5, 1.5, len(date_range)
                ),
                "Urban_Year_Over_Year": cpi_yoy
                + np.random.uniform(-0.5, 0.5, len(date_range)),
                "Rural_Year_Over_Year": cpi_yoy
                + np.random.uniform(-0.5, 0.5, len(date_range)),
            }
        )

    elif indicator_type == "m2":
        # Create M2 data
        # Base M2 value (unit: 100 million yuan)
        base_m2 = 2000000
        # Growth rate range (8%-12%)
        growth_rates = np.random.uniform(0.08, 0.12, len(date_range))

        # Calculate cumulative M2
        m2_values = []
        current_m2 = base_m2
        for rate in growth_rates:
            current_m2 *= 1 + rate / 12  # Monthly growth rate
            m2_values.append(current_m2)

        # Create DataFrame
        mock_data = pd.DataFrame(
            {
                "date": date_range,
                "Money_and_Quasi_Money_M2": m2_values,
                "M2_Year_Over_Year_Growth": growth_rates * 100,
                "M1": np.array(m2_values) * 0.3,  # M1 is about 30% of M2
                "M1_Year_Over_Year_Growth": growth_rates * 100
                + np.random.uniform(-2, 2, len(date_range)),
            }
        )

    else:
        # Create default mock data
        mock_data = pd.DataFrame(
            {
                "date": date_range,
                "value": np.random.normal(100, 10, len(date_range)),
                "yoy_change": np.random.uniform(-5, 8, len(date_range)),
            }
        )

    return mock_data
