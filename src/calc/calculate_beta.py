def calculate_beta(ticker, market_index="000300", start_date=None, end_date=None):
    """
    Calculate stock Beta value using actual market data
    
    Args:
        ticker: Stock code
        market_index: Market index code, default is CSI 300
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
    
    Returns:
        float: Beta value
    """
    from src.tools.api import get_price_history, prices_to_df
    from src.tools.factor_data_api import get_market_returns, get_index_data
    from src.utils.logging_config import setup_logger
    import pandas as pd
    
    logger = setup_logger('calculate_beta')
    
    try:
        # 1. Get stock price data
        stock_prices = get_price_history(ticker, start_date, end_date)
        if stock_prices is None or len(stock_prices) == 0:
            logger.warning(f"Unable to get stock {ticker} price data, using default Beta value")
            return 1.0  # Return market average when no data
        
        stock_df = prices_to_df(stock_prices)
        stock_returns = stock_df['close'].pct_change().dropna()
        
        # 2. Get market index data
        try:
            # Try to get market data using factor_data_api
            market_data = get_market_returns(index_code=market_index, start_date=start_date, end_date=end_date)
            if market_data is None or len(market_data) == 0:
                logger.info(f"Unable to get market data from factor_data_api, trying alternative approach")
                # Alternative approach: directly get index prices and calculate returns
                market_prices = get_index_data(index_symbol=market_index, fields=["date", "close"], start_date=start_date, end_date=end_date)
                if market_prices is not None and not market_prices.empty:
                    market_df = pd.DataFrame(market_prices)
                    market_df['date'] = pd.to_datetime(market_df['date'])
                    market_df.set_index('date', inplace=True)
                    market_returns = market_df['close'].pct_change().dropna()
                else:
                    logger.warning(f"Unable to get market index {market_index} data, using default Beta value")
                    return 1.0  # Return default value when no market data
            else:
                market_returns = market_data
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            # Unable to get market data, try using price API directly
            try:
                market_prices = get_price_history(market_index, start_date, end_date)
                if market_prices is None or len(market_prices) == 0:
                    logger.warning(f"Unable to get market index data through price API, using default Beta value")
                    return 1.0
                    
                market_df = prices_to_df(market_prices)
                market_returns = market_df['close'].pct_change().dropna()
            except Exception as subex:
                logger.error(f"Alternative method to get market data also failed: {subex}")
                return 1.0  # Return default value when all methods fail
        
        try:
            # 3. Ensure both series have common date indices
            # First convert both to same time format
            if not isinstance(stock_returns.index, pd.DatetimeIndex):
                stock_returns.index = pd.to_datetime(stock_returns.index)
            
            if not isinstance(market_returns.index, pd.DatetimeIndex):
                market_returns.index = pd.to_datetime(market_returns.index)
                
            # Convert both date indices to strings to eliminate timezone and other subtle differences
            stock_returns.index = stock_returns.index.strftime('%Y-%m-%d')
            market_returns.index = market_returns.index.strftime('%Y-%m-%d')
            
            # Convert back to date type to maintain sorting capability
            stock_returns.index = pd.to_datetime(stock_returns.index)
            market_returns.index = pd.to_datetime(market_returns.index)
                
            common_dates = stock_returns.index.intersection(market_returns.index)
            
            # Debug output
            logger.info(f"Stock date range: {stock_returns.index.min()} to {stock_returns.index.max()}")
            logger.info(f"Market date range: {market_returns.index.min()} to {market_returns.index.max()}")
            logger.info(f"Number of common dates: {len(common_dates)}")
            
            if len(common_dates) < 15:  # Need sufficient data points
                logger.warning(f"Insufficient overlap between stock and market data, only {len(common_dates)} common dates, using default Beta value")
                return 1.0
            
            # 4. Calculate Beta
            stock_ret = stock_returns[common_dates]
            market_ret = market_returns[common_dates]
            
            covariance = stock_ret.cov(market_ret)
            market_variance = market_ret.var()
            
            if market_variance > 0:
                beta = covariance / market_variance
                logger.info(f"Successfully calculated Beta value for {ticker}: {beta:.2f}")
            else:
                logger.warning(f"Market variance is zero, cannot calculate Beta value, using default value")
                beta = 1.0
            
            # Ensure beta is within reasonable range
            if not (0.0 <= beta <= 3.0):
                logger.warning(f"Calculated Beta value {beta:.2f} exceeds reasonable range, adjusting to limit range")
                beta = max(min(beta, 3.0), 0.2)
                
            return beta
        except Exception as e:
            logger.error(f"Error calculating Beta: {e}")
            logger.error(f"Stock returns index type: {type(stock_returns.index)}")
            logger.error(f"Market returns index type: {type(market_returns.index)}")
            return 1.0  # Return market average when error occurs
    except Exception as e:
        logger.error(f"Error calculating Beta value: {e}")
        return 1.0  # Return market average when error occurs