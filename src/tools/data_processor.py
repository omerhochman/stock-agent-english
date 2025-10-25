from datetime import datetime
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd


class DataProcessor:
    """Data processing class providing data cleaning, transformation and enhancement functionality"""

    def process_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process price data including cleaning, standardization and enhancement

        Args:
            df: DataFrame containing price data

        Returns:
            Processed DataFrame
        """
        if df.empty or len(df) < 5:
            return df

        # Create copy to avoid modifying original data
        processed_df = df.copy()

        # 1. Standardize date column
        processed_df = self._standardize_date_column(processed_df)

        # 2. Handle missing values
        processed_df = self._handle_missing_values(processed_df)

        # 3. Calculate derived metrics
        processed_df = self._calculate_derived_metrics(processed_df)

        # 4. Final data cleaning
        processed_df = self._final_data_cleanup(processed_df)

        # 5. Sort data
        if "date" in processed_df.columns:
            processed_df = processed_df.sort_values("date", ascending=True)

        return processed_df

    def _standardize_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize date column"""
        if "date" not in df.columns:
            # Try to find other possible names for date column
            date_columns = [
                col
                for col in df.columns
                if any(x in col.lower() for x in ["date", "time", "date"])
            ]
            if date_columns:
                df = df.rename(columns={date_columns[0]: "date"})

        # Ensure date column is datetime type
        if "date" in df.columns:
            if df["date"].dtype != "datetime64[ns]":
                try:
                    df["date"] = pd.to_datetime(df["date"])
                except Exception as e:
                    print(f"Warning: Date conversion failed - {e}")

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values"""
        # Missing value handling for price columns
        price_columns = ["open", "high", "low", "close"]
        available_columns = [col for col in price_columns if col in df.columns]

        if available_columns:
            # For price columns, use forward fill
            df[available_columns] = df[available_columns].ffill()

            # If there are still missing values (e.g., at sequence start), use backward fill
            df[available_columns] = df[available_columns].bfill()

        # Missing value handling for volume columns
        volume_columns = ["volume", "amount"]
        available_vol_columns = [col for col in volume_columns if col in df.columns]

        if available_vol_columns:
            # For volume, fill missing values with 0
            df[available_vol_columns] = df[available_vol_columns].fillna(0)

        return df

    def _calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived metrics"""
        # Check if required columns exist
        required_columns = ["open", "high", "low", "close"]
        if not all(col in df.columns for col in required_columns):
            return df

        # Calculate daily returns
        if "daily_return" not in df.columns:
            df["daily_return"] = df["close"].pct_change()

        # Calculate true range - fix potential broadcasting errors
        if "true_range" not in df.columns:
            high_low = df["high"] - df["low"]
            high_close_prev = (df["high"] - df["close"].shift(1)).abs()
            low_close_prev = (df["low"] - df["close"].shift(1)).abs()

            # Ensure all sequences have consistent length, handle NaN values
            high_low = high_low.fillna(0)
            high_close_prev = high_close_prev.fillna(0)
            low_close_prev = low_close_prev.fillna(0)

            # Use numpy's maximum function instead of pandas concat and max
            df["true_range"] = np.maximum.reduce(
                [high_low, high_close_prev, low_close_prev]
            )

        # Calculate typical price
        if "typical_price" not in df.columns:
            df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3

        # Calculate moving averages (including ma10 needed by system)
        if "ma5" not in df.columns:
            df["ma5"] = df["close"].rolling(window=5).mean()

        if "ma10" not in df.columns:
            df["ma10"] = df["close"].rolling(window=10).mean()

        if "ma20" not in df.columns:
            df["ma20"] = df["close"].rolling(window=20).mean()

        # Calculate MACD indicator
        if "macd" not in df.columns:
            exp1 = df["close"].ewm(span=12, adjust=False).mean()
            exp2 = df["close"].ewm(span=26, adjust=False).mean()
            df["macd"] = exp1 - exp2
            df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
            df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Calculate RSI indicator (using correct naming)
        if "rsi" not in df.columns:
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df["rsi"] = 100 - (100 / (1 + rs))
            # Fill NaN values
            df["rsi"] = df["rsi"].fillna(50)

        return df

    def process_financial_data(
        self, data: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Process financial data, clean and enhance financial indicators and report data

        Args:
            data: Financial data, can be dictionary or list of dictionaries

        Returns:
            Processed financial data
        """
        if isinstance(data, dict):
            return self._process_single_financial_item(data)
        elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
            return [self._process_single_financial_item(item) for item in data]
        else:
            return data

    def _process_single_financial_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process single financial data item"""
        processed_item = item.copy()

        # Ensure all numeric values are float type
        for key, value in processed_item.items():
            if isinstance(value, (int, float)):
                processed_item[key] = float(value)
            elif isinstance(value, str) and value.replace(".", "", 1).isdigit():
                processed_item[key] = float(value)

        # Calculate derived indicators (if sufficient base data available)
        if all(k in processed_item for k in ["operating_revenue", "operating_profit"]):
            processed_item["operating_margin"] = (
                processed_item["operating_profit"] / processed_item["operating_revenue"]
                if processed_item["operating_revenue"] != 0
                else 0
            )

        if all(k in processed_item for k in ["net_income", "operating_revenue"]):
            processed_item["net_margin"] = (
                processed_item["net_income"] / processed_item["operating_revenue"]
                if processed_item["operating_revenue"] != 0
                else 0
            )

        # Add timestamp
        if "timestamp" not in processed_item:
            processed_item["timestamp"] = datetime.now().isoformat()

        return processed_item

    def enrich_data(
        self, df: pd.DataFrame, technical_indicators: bool = True
    ) -> pd.DataFrame:
        """
        Enhance data by adding additional technical indicators

        Args:
            df: Original data DataFrame
            technical_indicators: Whether to add technical indicators

        Returns:
            Enhanced DataFrame
        """
        if df.empty:
            return df

        enhanced_df = df.copy()

        # Add technical indicators
        if technical_indicators and "close" in enhanced_df.columns:
            # Calculate MACD
            enhanced_df = self._add_macd(enhanced_df)

            # Calculate RSI
            enhanced_df = self._add_rsi(enhanced_df)

            # Calculate Bollinger Bands
            enhanced_df = self._add_bollinger_bands(enhanced_df)

        return enhanced_df

    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD indicator"""
        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = exp1 - exp2
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        return df

    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI indicator"""
        if "rsi" not in df.columns:
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df["rsi"] = 100 - (100 / (1 + rs))
            # Fill NaN values
            df["rsi"] = df["rsi"].fillna(50)
        return df

    def _add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Bands indicator"""
        df["20ma"] = df["close"].rolling(window=20).mean()
        df["upper_band"] = df["20ma"] + (df["close"].rolling(window=20).std() * 2)
        df["lower_band"] = df["20ma"] - (df["close"].rolling(window=20).std() * 2)
        return df

    def _final_data_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final data cleanup, handle infinite values and outliers"""
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], np.nan)

        # Handle outliers in numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            if col in df.columns:
                # Calculate reasonable range (using quantiles)
                q1 = df[col].quantile(0.01)
                q99 = df[col].quantile(0.99)

                # Limit extreme values
                if pd.notna(q1) and pd.notna(q99) and q99 > q1:
                    df[col] = df[col].clip(lower=q1, upper=q99)

                # Fill remaining NaN values
                df[col] = df[col].fillna(df[col].median())

        return df


# Create singleton instance
data_processor = DataProcessor()
