import numpy as np
import pandas as pd

def estimate_covariance_ewma(returns: pd.DataFrame, 
                            lambda_param: float = 0.94, 
                            min_periods: int = 20) -> pd.DataFrame:
    """
    使用指数加权移动平均法(EWMA)估计协方差矩阵
    
    EWMA是RiskMetrics采用的标准方法，使用指数衰减的权重计算协方差
    公式: σ_t^2 = (1-λ) * r_{t-1}^2 + λ * σ_{t-1}^2
    
    Args:
        returns: 收益率DataFrame，每列为一个资产，每行为一个时间点
        lambda_param: 衰减因子，通常在0.9到0.99之间，值越大表示历史数据权重越高
        min_periods: 计算需要的最小样本数
        
    Returns:
        协方差矩阵DataFrame
    """
    # 去除缺失值
    returns_clean = returns.fillna(0)
    
    # 资产数量
    n_assets = returns_clean.shape[1]
    
    # 准备EWMA协方差矩阵
    # 初始化为样本协方差矩阵
    sample_cov = returns_clean.cov().values
    
    # 设置初始协方差矩阵
    ewma_cov = sample_cov.copy()
    
    # 收益率矩阵
    returns_array = returns_clean.values
    
    # 计算EWMA协方差矩阵
    for t in range(1, len(returns_clean)):
        # 获取当前收益率向量
        r_t = returns_array[t-1, :]
        
        # 计算外积 r_t * r_t^T，得到协方差矩阵的即时估计
        outer_product = np.outer(r_t, r_t)
        
        # 更新EWMA协方差矩阵
        ewma_cov = lambda_param * ewma_cov + (1 - lambda_param) * outer_product
    
    # 转换回DataFrame
    ewma_cov_df = pd.DataFrame(
        ewma_cov, 
        index=returns_clean.columns, 
        columns=returns_clean.columns
    )
    
    return ewma_cov_df