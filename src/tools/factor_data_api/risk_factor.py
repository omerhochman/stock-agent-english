import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from src.tools.api import get_price_history

class RiskFactorAPI:
    """Risk Factor API, provides risk factor data calculation and acquisition functionality"""
    
    def calculate_beta(self, symbol: str, market_index: str = 'sh000300',
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> float:
        """
        Calculate beta coefficient of stock relative to market index
        
        Args:
            symbol: Stock code
            market_index: Market index code, default is CSI 300
            start_date: Start date, format: YYYY-MM-DD
            end_date: End date, format: YYYY-MM-DD
            
        Returns:
            Beta coefficient
        """
        # Handle date parameters
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if start_date is None:
            start_date = (datetime.strptime(end_date, '%Y-%m-%d') - 
                        timedelta(days=365)).strftime('%Y-%m-%d')
        
        # Get stock price data
        stock_df = get_price_history(symbol, start_date, end_date)
        
        # Get index price data
        market_df = get_price_history(market_index, start_date, end_date)
        
        if stock_df.empty or market_df.empty:
            return 0.0
        
        # Calculate stock daily returns
        if 'daily_return' not in stock_df.columns and 'close' in stock_df.columns:
            stock_df['daily_return'] = stock_df['close'].pct_change().fillna(0)
        
        # Calculate market index daily returns
        if 'daily_return' not in market_df.columns and 'close' in market_df.columns:
            market_df['daily_return'] = market_df['close'].pct_change().fillna(0)
        
        # Ensure data alignment
        merged_df = pd.DataFrame()
        merged_df['stock_return'] = stock_df['daily_return'].reset_index(drop=True)
        merged_df['market_return'] = market_df['daily_return'].reset_index(drop=True)
        
        # Remove missing values
        merged_df = merged_df.dropna()
        
        # Calculate beta coefficient
        if len(merged_df) > 30:  # Ensure sufficient data points
            # Use covariance method to calculate beta
            covariance = merged_df['stock_return'].cov(merged_df['market_return'])
            market_variance = merged_df['market_return'].var()
            
            # Avoid division by zero error
            if market_variance != 0:
                beta = covariance / market_variance
                return beta
        
        return 1.0  # Default beta is 1
    
    def calculate_alpha(self, symbol: str, market_index: str = 'sh000300',
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      risk_free_rate: float = 0.03) -> float:
        """
        Calculate stock alpha value

        Args:
            symbol: Stock code
            market_index: Market index code, default CSI 300
            start_date: Start date, format: YYYY-MM-DD
            end_date: End date, format: YYYY-MM-DD
            risk_free_rate: Risk-free rate, default 3%
            
        Returns:
            Alpha value
        """
        # Handle date parameters
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if start_date is None:
            start_date = (datetime.strptime(end_date, '%Y-%m-%d') - 
                        timedelta(days=365)).strftime('%Y-%m-%d')
        
        # Get stock price data
        stock_df = get_price_history(symbol, start_date, end_date)
        
        # Get index price data
        market_df = get_price_history(market_index, start_date, end_date)
        
        if stock_df.empty or market_df.empty:
            return 0.0
        
        # Calculate stock cumulative return
        if 'close' in stock_df.columns:
            start_price = stock_df['close'].iloc[0]
            end_price = stock_df['close'].iloc[-1]
            stock_return = (end_price / start_price) - 1
        else:
            return 0.0
        
        # Calculate market cumulative return
        if 'close' in market_df.columns:
            start_price = market_df['close'].iloc[0]
            end_price = market_df['close'].iloc[-1]
            market_return = (end_price / start_price) - 1
        else:
            return 0.0
        
        # Calculate beta coefficient
        beta = self.calculate_beta(symbol, market_index, start_date, end_date)
        
        # Calculate alpha
        days = (stock_df['date'].iloc[-1] - stock_df['date'].iloc[0]).days
        annualized_rf = (1 + risk_free_rate) ** (days / 365) - 1
        
        alpha = stock_return - (annualized_rf + beta * (market_return - annualized_rf))
        
        return alpha
    
    def calculate_volatility(self, symbol: str, 
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           window: int = 20) -> float:
        """
        Calculate stock volatility
        
        Args:
            symbol: Stock code
            start_date: Start date, format: YYYY-MM-DD
            end_date: End date, format: YYYY-MM-DD
            window: Rolling window size, default 20
            
        Returns:
            Annualized volatility
        """
        # Get price data
        df = get_price_history(symbol, start_date, end_date)
        
        if df.empty or 'close' not in df.columns:
            return 0.0
        
        # Calculate daily returns
        df['daily_return'] = df['close'].pct_change().fillna(0)
        
        # Calculate rolling volatility
        df['volatility'] = df['daily_return'].rolling(window=window).std() * np.sqrt(252)
        
        # Return latest volatility
        latest_volatility = df['volatility'].dropna().iloc[-1] if not df['volatility'].dropna().empty else 0.0
        
        return latest_volatility
    
    def calculate_sharpe_ratio(self, symbol: str,
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None,
                             risk_free_rate: float = 0.03) -> float:
        """
        Calculate stock Sharpe ratio
        
        Args:
            symbol: Stock code
            start_date: Start date, format: YYYY-MM-DD
            end_date: End date, format: YYYY-MM-DD
            risk_free_rate: Risk-free rate, default 3%
            
        Returns:
            Sharpe ratio
        """
        # Get price data
        df = get_price_history(symbol, start_date, end_date)
        
        if df.empty or 'close' not in df.columns:
            return 0.0
        
        # Calculate daily returns
        df['daily_return'] = df['close'].pct_change().fillna(0)
        
        # Calculate annualized return
        days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
        annualized_return = ((1 + df['daily_return']).prod()) ** (252 / len(df)) - 1
        
        # Calculate annualized volatility
        annualized_volatility = df['daily_return'].std() * np.sqrt(252)
        
        # Calculate Sharpe ratio
        daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
        excess_return = annualized_return - risk_free_rate
        
        # Check numerical stability
        if annualized_volatility == 0 or np.isclose(annualized_volatility, 0, atol=1e-10):
            return 0.0
        
        if not np.isfinite(annualized_volatility) or not np.isfinite(excess_return):
            return 0.0
        
        sharpe_ratio = excess_return / annualized_volatility
        
        # Ensure finite return value
        if not np.isfinite(sharpe_ratio):
            return 0.0
        
        return sharpe_ratio
    
    def calculate_maximum_drawdown(self, symbol: str,
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None) -> float:
        """
        Calculate stock maximum drawdown
        
        Args:
            symbol: Stock code
            start_date: Start date, format: YYYY-MM-DD
            end_date: End date, format: YYYY-MM-DD
            
        Returns:
            Maximum drawdown ratio
        """
        # Get price data
        df = get_price_history(symbol, start_date, end_date)
        
        if df.empty or 'close' not in df.columns:
            return 0.0
        
        # Calculate cumulative maximum
        df['cum_max'] = df['close'].cummax()
        
        # Calculate drawdown ratio
        df['drawdown'] = (df['cum_max'] - df['close']) / df['cum_max']
        
        # Get maximum drawdown
        max_drawdown = df['drawdown'].max()
        
        return max_drawdown