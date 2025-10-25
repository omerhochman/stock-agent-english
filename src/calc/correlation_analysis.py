from typing import Dict, Optional

import numpy as np
import pandas as pd


def analyze_asset_correlations(returns_df: pd.DataFrame) -> Dict:
    """
    Analyze correlations between assets and provide diversification suggestions

    Args:
        returns_df: DataFrame containing multiple asset returns

    Returns:
        Dict: Dictionary containing correlation analysis and diversification suggestions
    """
    # Calculate correlation coefficient matrix
    correlation = returns_df.corr()

    # Calculate average correlation
    avg_corr = correlation.values[np.triu_indices_from(correlation.values, k=1)].mean()

    # Find high correlation pairs
    high_corr_pairs = []
    for i in range(len(correlation.columns)):
        for j in range(i + 1, len(correlation.columns)):
            if correlation.iloc[i, j] > 0.7:  # Threshold can be adjusted
                high_corr_pairs.append(
                    {
                        "asset1": correlation.columns[i],
                        "asset2": correlation.columns[j],
                        "correlation": float(correlation.iloc[i, j]),
                    }
                )

    # Find low correlation pairs
    low_corr_pairs = []
    for i in range(len(correlation.columns)):
        for j in range(i + 1, len(correlation.columns)):
            if correlation.iloc[i, j] < 0.3:  # Threshold can be adjusted
                low_corr_pairs.append(
                    {
                        "asset1": correlation.columns[i],
                        "asset2": correlation.columns[j],
                        "correlation": float(correlation.iloc[i, j]),
                    }
                )

    # Generate diversification suggestions
    diversification_tips = []

    if avg_corr > 0.6:
        diversification_tips.append(
            "Portfolio has high overall correlation, may need to add low-correlation assets to improve diversification"
        )

    if high_corr_pairs:
        diversification_tips.append(
            "Consider reducing one of the high-correlation asset pairs to avoid redundant risk"
        )

    if len(low_corr_pairs) < len(correlation.columns) / 4:
        diversification_tips.append(
            "Portfolio has few low-correlation assets, consider adding other industries or asset classes"
        )

    # Calculate Principal Component Analysis (PCA) to assess risk sources
    try:
        from sklearn.decomposition import PCA

        pca = PCA()
        pca.fit(returns_df)

        # Calculate proportion of variance explained by first 3 principal components
        explained_variance = pca.explained_variance_ratio_[:3].sum()

        if explained_variance > 0.8:
            diversification_tips.append(
                f"First 3 principal components explain {explained_variance:.1%} of variance, indicating most risk comes from few sources, need better risk diversification"
            )
    except ImportError:
        # Skip PCA analysis if sklearn library is not available
        pass

    return {
        "average_correlation": float(avg_corr),
        "high_correlation_pairs": high_corr_pairs,
        "low_correlation_pairs": low_corr_pairs,
        "diversification_tips": diversification_tips,
    }


def calculate_optimal_weights_for_correlation(returns_df: pd.DataFrame) -> Dict:
    """
    Calculate minimum correlation portfolio weights

    Args:
        returns_df: DataFrame containing multiple asset returns

    Returns:
        Dict: Contains minimum correlation portfolio weights
    """
    try:
        # Calculate covariance matrix
        cov_matrix = returns_df.cov()

        # Use optimization to find minimum variance portfolio
        import scipy.optimize as sco

        # Number of assets
        n_assets = len(returns_df.columns)

        # Objective function: portfolio variance
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))

        # Constraints
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}  # Weights sum to 1
        bounds = tuple((0, 1) for _ in range(n_assets))  # Weights between 0 and 1

        # Initial weight guess
        init_weights = np.array([1.0 / n_assets] * n_assets)

        # Optimization
        result = sco.minimize(
            portfolio_variance,
            init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        # Check if successful
        if result["success"]:
            # Build weights dictionary
            weights_dict = {
                returns_df.columns[i]: float(result["x"][i]) for i in range(n_assets)
            }

            return {
                "success": True,
                "weights": weights_dict,
                "portfolio_variance": float(result["fun"]),
            }
        else:
            return {
                "success": False,
                "error": result["message"],
                "weights": {
                    col: 1.0 / n_assets for col in returns_df.columns
                },  # Default equal weights
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "weights": {
                col: 1.0 / len(returns_df.columns) for col in returns_df.columns
            },  # Default equal weights
        }


def cluster_assets(returns_df: pd.DataFrame, n_clusters: Optional[int] = None) -> Dict:
    """
    Use cluster analysis to group assets

    Args:
        returns_df: DataFrame containing multiple asset returns
        n_clusters: Number of clusters, default is None (auto-determined)

    Returns:
        Dict: Contains clustering results
    """
    try:
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()

        # Convert correlation matrix to distance matrix
        distance_matrix = 1 - np.abs(corr_matrix)

        # If no cluster number specified, auto-determine
        if n_clusters is None:
            # Use square root of number of assets as default cluster number
            n_clusters = max(2, int(np.sqrt(len(returns_df.columns))))

        # Use hierarchical clustering
        from scipy.cluster.hierarchy import fcluster, linkage

        # Perform clustering
        Z = linkage(distance_matrix, method="ward")
        clusters = fcluster(Z, t=n_clusters, criterion="maxclust")

        # Build results
        cluster_dict = {}
        for i, cluster_id in enumerate(clusters):
            cluster_name = f"Cluster_{cluster_id}"
            if cluster_name not in cluster_dict:
                cluster_dict[cluster_name] = []
            cluster_dict[cluster_name].append(returns_df.columns[i])

        return {
            "success": True,
            "n_clusters": n_clusters,
            "clusters": cluster_dict,
            "asset_clusters": {
                returns_df.columns[i]: int(clusters[i])
                for i in range(len(returns_df.columns))
            },
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "n_clusters": 0,
            "clusters": {},
            "asset_clusters": {col: 0 for col in returns_df.columns},
        }
