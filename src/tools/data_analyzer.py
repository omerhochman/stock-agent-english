import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.tools.api import get_price_history


def analyze_stock_data(symbol: str, start_date: str = None, end_date: str = None):
    """
    Get stock historical data, calculate technical indicators, and save as CSV file

    Args:
        symbol: Stock code
        start_date: Start date, format: YYYY-MM-DD
        end_date: End date, format: YYYY-MM-DD
    """
    # Get historical data
    df = get_price_history(symbol, start_date, end_date)

    if df.empty:
        print("No data retrieved")
        return

    # Calculate technical indicators
    df = calculate_technical_indicators(df)

    # Save as CSV file
    output_file = f"{symbol}_analysis_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(output_file, index=False)
    print(f"Data saved to file: {output_file}")

    # Print basic statistics
    print_statistics(df)


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate various technical indicators"""
    # Create copy to avoid modifying original data
    df_with_indicators = df.copy()
    
    # 1. Moving averages
    df_with_indicators = calculate_moving_averages(df_with_indicators)
    
    # 2. MACD
    df_with_indicators = calculate_macd(df_with_indicators)
    
    # 3. RSI
    df_with_indicators = calculate_rsi(df_with_indicators)
    
    # 4. Bollinger Bands
    df_with_indicators = calculate_bollinger_bands(df_with_indicators)
    
    # 5. Volume related indicators
    df_with_indicators = calculate_volume_indicators(df_with_indicators)
    
    # 6. Price momentum indicators
    df_with_indicators = calculate_momentum_indicators(df_with_indicators)
    
    # 7. Volatility indicators
    df_with_indicators = calculate_volatility_indicators(df_with_indicators)
    
    return df_with_indicators


def calculate_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate moving averages"""
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma60'] = df['close'].rolling(window=60).mean()
    return df


def calculate_macd(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate MACD indicator"""
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['signal_line']
    return df


def calculate_rsi(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate RSI indicator"""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df


def calculate_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Bollinger Bands indicator"""
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    return df


def calculate_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volume related indicators"""
    df['volume_ma5'] = df['volume'].rolling(window=5).mean()
    df['volume_ma20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma5']
    return df


def calculate_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate momentum indicators"""
    df['price_momentum'] = df['close'].pct_change(periods=5)
    df['price_acceleration'] = df['price_momentum'].diff()
    return df


def calculate_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volatility indicators"""
    df['daily_return'] = df['close'].pct_change()
    df['volatility_5d'] = df['daily_return'].rolling(
        window=5).std() * np.sqrt(252)
    df['volatility_20d'] = df['daily_return'].rolling(
        window=20).std() * np.sqrt(252)
    return df


def print_statistics(df: pd.DataFrame):
    """Print basic statistics"""
    print("\nBasic Statistics:")
    print(f"Data time range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total records: {len(df)}")
    print("\nNaN value statistics:")
    print(df.isna().sum())


if __name__ == "__main__":
    # Test code
    symbol = "600519"  # Kweichow Moutai
    current_date = datetime.now()
    end_date = current_date.strftime("%Y-%m-%d")  # Use today as end date
    start_date = (current_date - timedelta(days=365)).strftime("%Y-%m-%d")

    print(f"Analysis time range: {start_date} to {end_date}")
    analyze_stock_data(symbol, start_date, end_date)