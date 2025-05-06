def calculate_beta(ticker, market_index="000300", start_date=None, end_date=None):
    """
    使用实际市场数据计算股票的Beta值
    
    Args:
        ticker: 股票代码
        market_index: 市场指数代码，默认为沪深300
        start_date: 开始日期 YYYY-MM-DD
        end_date: 结束日期 YYYY-MM-DD
    
    Returns:
        float: Beta值
    """
    from src.tools.api import get_price_history, prices_to_df
    from src.tools.factor_data_api import get_market_returns, get_index_data
    import pandas as pd
    import logging
    
    logger = logging.getLogger('calculate_beta')
    
    try:
        # 1. 获取股票价格数据
        stock_prices = get_price_history(ticker, start_date, end_date)
        if stock_prices is None or len(stock_prices) == 0:
            logger.warning(f"无法获取股票 {ticker} 价格数据，使用默认Beta值")
            return 1.0  # 无数据时返回市场平均值
        
        stock_df = prices_to_df(stock_prices)
        stock_returns = stock_df['close'].pct_change().dropna()
        
        # 2. 获取市场指数数据
        try:
            # 尝试使用factor_data_api获取市场数据
            market_data = get_market_returns(index_code=market_index, start_date=start_date, end_date=end_date)
            if market_data is None or len(market_data) == 0:
                logger.info(f"无法从factor_data_api获取市场数据，尝试备选方案")
                # 备选方案：直接获取指数价格并计算收益率
                market_prices = get_index_data(index_symbol=market_index, fields=["date", "close"], start_date=start_date, end_date=end_date)
                if market_prices is not None and not market_prices.empty:
                    market_df = pd.DataFrame(market_prices)
                    market_df['date'] = pd.to_datetime(market_df['date'])
                    market_df.set_index('date', inplace=True)
                    market_returns = market_df['close'].pct_change().dropna()
                else:
                    logger.warning(f"无法获取市场指数 {market_index} 数据，使用默认Beta值")
                    return 1.0  # 无市场数据时返回默认值
            else:
                market_returns = market_data
        except Exception as e:
            logger.error(f"获取市场数据出错: {e}")
            # 无法获取市场数据，尝试直接使用价格API
            try:
                market_prices = get_price_history(market_index, start_date, end_date)
                if market_prices is None or len(market_prices) == 0:
                    logger.warning(f"无法通过价格API获取市场指数数据，使用默认Beta值")
                    return 1.0
                    
                market_df = prices_to_df(market_prices)
                market_returns = market_df['close'].pct_change().dropna()
            except Exception as subex:
                logger.error(f"备选方法获取市场数据也失败: {subex}")
                return 1.0  # 所有方法都失败时返回默认值
        
        try:
            # 3. 确保两个序列有共同的日期索引
            # 先将两者转换为相同的时间格式
            if not isinstance(stock_returns.index, pd.DatetimeIndex):
                stock_returns.index = pd.to_datetime(stock_returns.index)
            
            if not isinstance(market_returns.index, pd.DatetimeIndex):
                market_returns.index = pd.to_datetime(market_returns.index)
                
            # 将两者的日期索引转换为字符串以消除时区等细微差异的影响
            stock_returns.index = stock_returns.index.strftime('%Y-%m-%d')
            market_returns.index = market_returns.index.strftime('%Y-%m-%d')
            
            # 再次转换为日期类型以保持排序能力
            stock_returns.index = pd.to_datetime(stock_returns.index)
            market_returns.index = pd.to_datetime(market_returns.index)
                
            common_dates = stock_returns.index.intersection(market_returns.index)
            
            # 调试输出
            logger.info(f"股票日期范围: {stock_returns.index.min()} 到 {stock_returns.index.max()}")
            logger.info(f"市场日期范围: {market_returns.index.min()} 到 {market_returns.index.max()}")
            logger.info(f"共同日期数量: {len(common_dates)}")
            
            if len(common_dates) < 15:  # 需要足够的数据点
                logger.warning(f"股票和市场数据重叠时间段不足，只有{len(common_dates)}个共同日期，使用默认Beta值")
                return 1.0
            
            # 4. 计算Beta
            stock_ret = stock_returns[common_dates]
            market_ret = market_returns[common_dates]
            
            covariance = stock_ret.cov(market_ret)
            market_variance = market_ret.var()
            
            if market_variance > 0:
                beta = covariance / market_variance
                logger.info(f"成功计算 {ticker} 的Beta值: {beta:.2f}")
            else:
                logger.warning(f"市场方差为零，无法计算Beta值，使用默认值")
                beta = 1.0
            
            # 确保beta在合理范围内
            if not (0.0 <= beta <= 3.0):
                logger.warning(f"计算的Beta值 {beta:.2f} 超出合理范围，调整为限制范围")
                beta = max(min(beta, 3.0), 0.2)
                
            return beta
        except Exception as e:
            logger.error(f"计算Beta时出错: {e}")
            logger.error(f"股票收益率索引类型: {type(stock_returns.index)}")
            logger.error(f"市场收益率索引类型: {type(market_returns.index)}")
            return 1.0  # 出错时返回市场平均值
    except Exception as e:
        logger.error(f"计算Beta值出错: {e}")
        return 1.0  # 出错时返回市场平均值