from typing import List

import numpy as np
import pandas as pd


class DataProcessor:
    """
    Data Processor
    Provides data cleaning, transformation and preprocessing functions
    """

    def __init__(self):
        pass

    def clean_price_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean price data

        Args:
            data: Raw price data

        Returns:
            pd.DataFrame: Cleaned data
        """
        if data is None or data.empty:
            return pd.DataFrame()

        # Remove duplicate rows
        data = data.drop_duplicates()

        # Handle missing values
        data = data.dropna()

        # Ensure price columns are numeric type
        price_columns = ["open", "high", "low", "close", "volume"]
        for col in price_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")

        # Remove outliers (prices that are 0 or negative)
        for col in ["open", "high", "low", "close"]:
            if col in data.columns:
                data = data[data[col] > 0]

        # Ensure high price >= low price
        if "high" in data.columns and "low" in data.columns:
            data = data[data["high"] >= data["low"]]

        # Sort by date
        if data.index.name != "date" and "date" in data.columns:
            data = data.set_index("date")

        data = data.sort_index()

        return data

    def calculate_returns(self, prices: pd.Series, method: str = "simple") -> pd.Series:
        """
        Calculate returns

        Args:
            prices: Price series
            method: Calculation method ('simple' or 'log')

        Returns:
            pd.Series: Returns series
        """
        if method == "simple":
            returns = prices.pct_change()
        elif method == "log":
            returns = np.log(prices / prices.shift(1))
        else:
            raise ValueError("method must be 'simple' or 'log'")

        return returns.dropna()

    def align_data(self, *dataframes: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Align time indices of multiple dataframes

        Args:
            *dataframes: Dataframes to align

        Returns:
            List[pd.DataFrame]: List of aligned dataframes
        """
        if len(dataframes) < 2:
            return list(dataframes)

        # Find common time index
        common_index = dataframes[0].index
        for df in dataframes[1:]:
            common_index = common_index.intersection(df.index)

        # Align all dataframes
        aligned_dfs = []
        for df in dataframes:
            aligned_df = df.reindex(common_index)
            aligned_dfs.append(aligned_df)

        return aligned_dfs

    def resample_data(self, data: pd.DataFrame, frequency: str) -> pd.DataFrame:
        """
        Resample data to specified frequency

        Args:
            data: Original data
            frequency: Target frequency ('D', 'W', 'M', etc.)

        Returns:
            pd.DataFrame: Resampled data
        """
        if data.empty:
            return data

        # Aggregation rules for price data
        agg_rules = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }

        # Only use existing columns
        available_rules = {
            col: rule for col, rule in agg_rules.items() if col in data.columns
        }

        resampled = data.resample(frequency).agg(available_rules)

        return resampled.dropna()
