import pandas as pd
import numpy as np
from typing import Dict, Optional

def analyze_asset_correlations(returns_df: pd.DataFrame) -> Dict:
    """
    分析资产间的相关性并提供多样化建议
    
    Args:
        returns_df: 包含多个资产收益率的DataFrame
        
    Returns:
        Dict: 包含相关性分析和多样化建议的字典
    """
    # 计算相关系数矩阵
    correlation = returns_df.corr()
    
    # 计算平均相关性
    avg_corr = correlation.values[np.triu_indices_from(correlation.values, k=1)].mean()
    
    # 找出高相关性对
    high_corr_pairs = []
    for i in range(len(correlation.columns)):
        for j in range(i+1, len(correlation.columns)):
            if correlation.iloc[i, j] > 0.7:  # 阈值可调整
                high_corr_pairs.append({
                    'asset1': correlation.columns[i],
                    'asset2': correlation.columns[j],
                    'correlation': float(correlation.iloc[i, j])
                })
    
    # 找出低相关性对
    low_corr_pairs = []
    for i in range(len(correlation.columns)):
        for j in range(i+1, len(correlation.columns)):
            if correlation.iloc[i, j] < 0.3:  # 阈值可调整
                low_corr_pairs.append({
                    'asset1': correlation.columns[i],
                    'asset2': correlation.columns[j],
                    'correlation': float(correlation.iloc[i, j])
                })
    
    # 生成多样化建议
    diversification_tips = []
    
    if avg_corr > 0.6:
        diversification_tips.append("投资组合整体相关性较高，可能需要增加低相关性资产来提高分散化效果")
    
    if high_corr_pairs:
        diversification_tips.append("考虑减少高相关性资产对中的一个，以避免冗余风险")
    
    if len(low_corr_pairs) < len(correlation.columns) / 4:
        diversification_tips.append("组合中低相关性资产较少，可以考虑增加其他行业或资产类别")
    
    # 计算主成分分析(PCA)来评估风险来源
    try:
        from sklearn.decomposition import PCA
        pca = PCA()
        pca.fit(returns_df)
        
        # 计算前3个主成分解释的方差比例
        explained_variance = pca.explained_variance_ratio_[:3].sum()
        
        if explained_variance > 0.8:
            diversification_tips.append(f"前3个主成分解释了{explained_variance:.1%}的方差，表明大部分风险来自少数几个来源，需要更好的风险分散")
    except ImportError:
        # 如果没有sklearn库，跳过PCA分析
        pass
    
    return {
        'average_correlation': float(avg_corr),
        'high_correlation_pairs': high_corr_pairs,
        'low_correlation_pairs': low_corr_pairs,
        'diversification_tips': diversification_tips
    }

def calculate_optimal_weights_for_correlation(returns_df: pd.DataFrame) -> Dict:
    """
    计算最小相关性投资组合权重
    
    Args:
        returns_df: 包含多个资产收益率的DataFrame
        
    Returns:
        Dict: 包含最小相关性投资组合的权重
    """
    try:
        # 计算协方差矩阵
        cov_matrix = returns_df.cov()
        
        # 使用最优化找到最小方差投资组合
        import scipy.optimize as sco
        
        # 资产数量
        n_assets = len(returns_df.columns)
        
        # 目标函数：投资组合方差
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        # 约束条件
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # 权重和为1
        bounds = tuple((0, 1) for _ in range(n_assets))  # 权重在0和1之间
        
        # 初始权重猜测
        init_weights = np.array([1.0/n_assets] * n_assets)
        
        # 最优化
        result = sco.minimize(portfolio_variance, init_weights, method='SLSQP',
                              bounds=bounds, constraints=constraints)
        
        # 检查是否成功
        if result['success']:
            # 构建权重字典
            weights_dict = {returns_df.columns[i]: float(result['x'][i]) 
                           for i in range(n_assets)}
            
            return {
                'success': True,
                'weights': weights_dict,
                'portfolio_variance': float(result['fun'])
            }
        else:
            return {
                'success': False,
                'error': result['message'],
                'weights': {col: 1.0/n_assets for col in returns_df.columns}  # 默认等权重
            }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'weights': {col: 1.0/len(returns_df.columns) for col in returns_df.columns}  # 默认等权重
        }

def cluster_assets(returns_df: pd.DataFrame, n_clusters: Optional[int] = None) -> Dict:
    """
    使用聚类分析将资产分组
    
    Args:
        returns_df: 包含多个资产收益率的DataFrame
        n_clusters: 聚类数量，默认为None（自动确定）
        
    Returns:
        Dict: 包含聚类结果
    """
    try:
        # 计算相关性矩阵
        corr_matrix = returns_df.corr()
        
        # 将相关性矩阵转换为距离矩阵
        distance_matrix = 1 - np.abs(corr_matrix)
        
        # 如果没有指定聚类数量，自动确定
        if n_clusters is None:
            # 使用资产数量的平方根作为默认聚类数量
            n_clusters = max(2, int(np.sqrt(len(returns_df.columns))))
        
        # 使用层次聚类
        from scipy.cluster.hierarchy import linkage, fcluster
        
        # 执行聚类
        Z = linkage(distance_matrix, method='ward')
        clusters = fcluster(Z, t=n_clusters, criterion='maxclust')
        
        # 构建结果
        cluster_dict = {}
        for i, cluster_id in enumerate(clusters):
            cluster_name = f"Cluster_{cluster_id}"
            if cluster_name not in cluster_dict:
                cluster_dict[cluster_name] = []
            cluster_dict[cluster_name].append(returns_df.columns[i])
        
        return {
            'success': True,
            'n_clusters': n_clusters,
            'clusters': cluster_dict,
            'asset_clusters': {returns_df.columns[i]: int(clusters[i]) 
                             for i in range(len(returns_df.columns))}
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'n_clusters': 0,
            'clusters': {},
            'asset_clusters': {col: 0 for col in returns_df.columns}
        }