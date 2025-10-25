import numpy as np
import pandas as pd
import random
import re
import operator
from deap import algorithms, base, creator, tools, gp
import os
from src.utils.logging_config import setup_logger
from typing import Dict, List, Any, Tuple, Callable
import json
from functools import partial
import warnings
from scipy import stats

# Set warning level and random seed
warnings.filterwarnings("ignore", category=RuntimeWarning)
logger = setup_logger('factor_mining')
random.seed(42)
np.random.seed(42)

# Special functions for safety protection
def protected_div(left, right):
    """Safe division, prevents division by zero error"""
    return left / right if abs(right) >= 1e-10 else 1.0

def protected_sqrt(x):
    """Safe square root, handles negative numbers"""
    return np.sqrt(x) if x > 0 else 0.0

def protected_log(x):
    """Safe logarithm, handles negative numbers and zero"""
    return np.log(x) if x > 0 else 0.0

def protected_inv(x):
    """Safe inverse, handles zero"""
    return 1.0 / x if abs(x) >= 1e-10 else 0.0

# Time series functions
def rolling_mean(x, window=10):
    """Rolling mean"""
    return pd.Series(x).rolling(window=window).mean().fillna(0).values

def rolling_std(x, window=10):
    """Rolling standard deviation"""
    return pd.Series(x).rolling(window=window).std().fillna(0).values

def ts_delta(x, period=1):
    """Time series difference"""
    return pd.Series(x).diff(period).fillna(0).values

def ts_delay(x, period=1):
    """Time series lag"""
    return pd.Series(x).shift(period).fillna(0).values

def ts_return(x, period=1):
    """Time series return"""
    return pd.Series(x).pct_change(period).fillna(0).values

def ts_rank(x, window=10):
    """Time series rank"""
    return pd.Series(x).rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]).fillna(0).values


class FactorMiningModule:
    """Main factor mining class, uses genetic programming to automatically generate and screen factors"""
    
    def __init__(self, model_dir: str = 'factors'):
        """Initialize factor mining module"""
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.best_factors = []
        self.pset = None
        self.toolbox = None
        self.logger = setup_logger('factor_mining')
    
    def _setup_primitives(self, input_names: List[str]):
        """Set up primitive set for genetic programming"""
        # Create primitive set
        self.pset = gp.PrimitiveSetTyped("MAIN", [float] * len(input_names), float)
        
        # Rename input variables
        for i, name in enumerate(input_names):
            self.pset.renameArguments(**{f'ARG{i}': name})
        
        # Add basic operators
        self.pset.addPrimitive(operator.add, [float, float], float)
        self.pset.addPrimitive(operator.sub, [float, float], float)
        self.pset.addPrimitive(operator.mul, [float, float], float)
        self.pset.addPrimitive(protected_div, [float, float], float)
        
        # Add mathematical functions
        self.pset.addPrimitive(operator.neg, [float], float)
        self.pset.addPrimitive(abs, [float], float)
        self.pset.addPrimitive(protected_sqrt, [float], float)
        self.pset.addPrimitive(protected_log, [float], float)
        self.pset.addPrimitive(protected_inv, [float], float)
        
        # Create and add time series functions
        ts_functions = {
            'rolling_mean': [5, 10, 20],
            'rolling_std': [5, 10],
            'ts_delta': [1, 5],
            'ts_delay': [1, 5],
            'ts_return': [1, 5],
            'ts_rank': [10]
        }
        
        for func_name, param_values in ts_functions.items():
            base_func = globals()[func_name]
            for param in param_values:
                func = partial(base_func, **({'window': param} if 'roll' in func_name or 'rank' in func_name else {'period': param}))
                func.__name__ = f"{func_name}_{param}"
                self.pset.addPrimitive(func, [float], float)
        
        # Add constants
        self.pset.addEphemeralConstant("rand", lambda: random.uniform(-1, 1), float)
    
    def _evaluate_factor(self, individual, X: np.ndarray, y: np.ndarray, 
                    eval_func: Callable, eval_window: int = 20) -> Tuple[float,]:
        """Evaluate factor quality"""
        # Convert individual to function
        func = self.toolbox.compile(expr=individual)
        
        try:
            # Calculate factor values
            factor_values = np.zeros(len(y))
            
            for i in range(len(y)):
                # Get all features for current sample
                sample = [X[i, j] for j in range(X.shape[1])]
                
                # Calculate factor value
                try:
                    factor_values[i] = func(*sample)
                except:
                    factor_values[i] = np.nan
            
            # Handle infinity and NaN
            factor_values = np.nan_to_num(factor_values, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Check if factor values are valid
            if np.all(factor_values == 0.0) or np.all(np.isclose(factor_values, factor_values[0])):
                # If all values are the same or all zero, give low score
                return (-0.5,)
                
            # Calculate evaluation metrics
            try:
                score = eval_func(factor_values, y)
                
                # Ensure score is a finite number
                if not np.isfinite(score):
                    score = -0.5
                    
                # Calculate factor complexity penalty
                complexity = len(str(individual))
                complexity_penalty = np.clip(complexity / 200, 0, 0.5)
                
                # Final score
                final_score = max(-1.0, min(1.0, score - complexity_penalty))
                
                return (final_score,)
            except Exception as e:
                self.logger.error(f"Error calculating factor score: {str(e)}")
                return (-0.5,)
                
        except Exception:
            # If error occurs, return lowest score
            return (-0.5,)
    
    def _calculate_ic(self, factor_values: np.ndarray, future_returns: np.ndarray, 
                     window: int = 20) -> float:
        """Calculate factor IC value (Information Coefficient)"""
        # Create pandas Series for easier calculation
        factor_series = pd.Series(factor_values)
        returns_series = pd.Series(future_returns)
        
        # Clear invalid values
        valid_indices = ~(np.isnan(factor_values) | np.isnan(future_returns))
        factor_series = factor_series[valid_indices]
        returns_series = returns_series[valid_indices]
        
        if len(factor_series) < window:
            return 0.0
        
        # Calculate rolling correlation coefficient
        ic_series = factor_series.rolling(window).corr(returns_series)
        
        # Return average IC value
        mean_ic = ic_series.mean()
        return 0.0 if np.isnan(mean_ic) else mean_ic
    
    def _calculate_return_spread(self, factor_values: np.ndarray, future_returns: np.ndarray, 
                               quantiles: int = 5) -> float:
        """Calculate factor group return difference"""
        if len(factor_values) < quantiles * 2:
            return 0.0
        
        # Clear invalid values
        valid_indices = ~(np.isnan(factor_values) | np.isnan(future_returns))
        factor_values = factor_values[valid_indices]
        future_returns = future_returns[valid_indices]
        
        if len(factor_values) < quantiles * 2:
            return 0.0
        
        # Group factor values
        df = pd.DataFrame({'factor': factor_values, 'returns': future_returns})
        df['quantile'] = pd.qcut(df['factor'], q=quantiles, labels=False, duplicates='drop')
        
        # Calculate average return for each group
        group_returns = df.groupby('quantile')['returns'].mean()
        
        # Return difference between high and low quantiles
        if len(group_returns) < quantiles:
            return 0.0
            
        return group_returns.iloc[-1] - group_returns.iloc[0]
    
    def _calculate_portfolio_return(self, factor_values: np.ndarray, future_returns: np.ndarray, 
                                  top_percentile: float = 0.2) -> float:
        """Calculate long-short portfolio return based on factor"""
        if len(factor_values) < 10:
            return 0.0
        
        # Clear invalid values
        valid_indices = ~(np.isnan(factor_values) | np.isnan(future_returns))
        factor_values = factor_values[valid_indices]
        future_returns = future_returns[valid_indices]
        
        if len(factor_values) < 10:
            return 0.0
        
        # Sort factor values
        sorted_indices = np.argsort(factor_values)
        
        # Select long and short portfolios
        n_assets = len(sorted_indices)
        n_select = max(1, int(n_assets * top_percentile))
        
        # Long: stocks with highest factor values
        long_indices = sorted_indices[-n_select:]
        long_return = np.mean(future_returns[long_indices])
        
        # Short: stocks with lowest factor values
        short_indices = sorted_indices[:n_select]
        short_return = np.mean(future_returns[short_indices])
        
        # Long-short portfolio return
        return long_return - short_return
    
    def _factor_fitness(self, factor_values: np.ndarray, future_returns: np.ndarray) -> float:
        """Comprehensively calculate factor quality"""
        try:
            # Clear invalid values
            mask = ~(np.isnan(factor_values) | np.isnan(future_returns) | 
                    np.isinf(factor_values) | np.isinf(future_returns))
            
            if np.sum(mask) < 10:  # If too few valid data points
                return 0.0
                
            factor_values_clean = factor_values[mask]
            future_returns_clean = future_returns[mask]
            
            # Calculate multiple metrics
            try:
                ic = self._calculate_ic(factor_values_clean, future_returns_clean)
                if not np.isfinite(ic):
                    ic = 0.0
            except:
                ic = 0.0
                
            try:
                return_spread = self._calculate_return_spread(factor_values_clean, future_returns_clean)
                if not np.isfinite(return_spread):
                    return_spread = 0.0
            except:
                return_spread = 0.0
                
            try:
                portfolio_return = self._calculate_portfolio_return(factor_values_clean, future_returns_clean)
                if not np.isfinite(portfolio_return):
                    portfolio_return = 0.0
            except:
                portfolio_return = 0.0
            
            # Calculate factor autocorrelation (stability)
            try:
                factor_series = pd.Series(factor_values_clean)
                factor_autocorr = factor_series.autocorr(lag=1)
                if np.isnan(factor_autocorr) or not np.isfinite(factor_autocorr):
                    factor_autocorr = 0
            except:
                factor_autocorr = 0
            
            # Calculate factor monotonicity
            try:
                monotonicity = self._calculate_monotonicity(factor_values_clean, future_returns_clean)
                if not np.isfinite(monotonicity):
                    monotonicity = 0.0
            except:
                monotonicity = 0.0
            
            # Comprehensive scoring (weighted by various metrics)
            score = (
                0.4 * abs(ic) +                 
                0.3 * abs(return_spread) +      
                0.2 * abs(portfolio_return) +   
                0.05 * abs(factor_autocorr) +   
                0.05 * abs(monotonicity)        
            )
            
            # Ensure return finite value
            return float(0.0 if not np.isfinite(score) else score)
        
        except Exception as e:
            self.logger.error(f"Error calculating factor fitness: {str(e)}")
            return 0.0
    
    def _calculate_monotonicity(self, factor_values: np.ndarray, future_returns: np.ndarray, 
                           n_groups: int = 5) -> float:
        """Calculate monotonic relationship strength between factor and future returns"""
        try:
            if len(factor_values) < n_groups * 2:
                return 0.0
                
            # Create dataframe
            df = pd.DataFrame({'factor': factor_values, 'returns': future_returns})
            
            # Safely perform grouping
            try:
                df['quantile'] = pd.qcut(df['factor'], q=n_groups, labels=False, duplicates='drop')
            except:
                # If quantile division fails, try reducing number of groups
                try:
                    n_groups = max(2, n_groups - 1)
                    df['quantile'] = pd.qcut(df['factor'], q=n_groups, labels=False, duplicates='drop')
                except:
                    # If still fails, use equal-width grouping
                    df['quantile'] = pd.cut(df['factor'], bins=n_groups, labels=False)
            
            # Calculate average return for each group
            group_returns = df.groupby('quantile')['returns'].mean()
            
            if len(group_returns) < 2:
                return 0.0
                
            # Calculate rank correlation
            try:
                from scipy import stats
                corr, _ = stats.spearmanr(group_returns.index, group_returns.values)
                return 0.0 if np.isnan(corr) else corr
            except:
                # If cannot calculate correlation, calculate simple trend
                trend = (group_returns.iloc[-1] - group_returns.iloc[0]) / (len(group_returns) - 1)
                return np.clip(trend, -1.0, 1.0)
        
        except Exception:
            return 0.0

    def evolve_factors(self, price_data: pd.DataFrame, n_factors: int = 5, 
                      population_size: int = 100, n_generations: int = 50,
                      future_return_periods: int = 5, min_fitness: float = 0.05):
        """Use genetic programming to evolve factors"""
        # Prepare feature data
        X, y, feature_names = self._prepare_data(price_data, future_return_periods)
        
        # Set up primitive set
        self._setup_primitives(feature_names)
        
        # Create fitness function
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        
        # Create individual
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
        
        # Create toolbox
        self.toolbox = base.Toolbox()
        
        # Register expression generator
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=4)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate_factor, X=X, y=y, eval_func=self._factor_fitness)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        
        # Create initial population
        pop = self.toolbox.population(n=population_size)
        hof = tools.HallOfFame(n_factors)
        
        # Statistics object
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        stats.register("std", np.std)
        
        # Start evolution
        try:
            self.logger.info(f"Starting factor evolution, population size: {population_size}, iterations: {n_generations}")
            pop, _ = algorithms.eaSimple(pop, self.toolbox, cxpb=0.5, mutpb=0.1,
                                         ngen=n_generations, stats=stats, halloffame=hof,
                                         verbose=True)
            
            # Extract best factors
            self.best_factors = []
            for i, ind in enumerate(hof):
                if ind.fitness.values[0] > min_fitness:
                    factor_func = self.toolbox.compile(expr=ind)
                    factor_name = f"GP_Factor_{i+1}"
                    
                    # Calculate factor values
                    factor_values = np.zeros(len(y))
                    for j in range(len(y)):
                        sample = [X[j, k] for k in range(X.shape[1])]
                        try:
                            factor_values[j] = factor_func(*sample)
                        except:
                            factor_values[j] = np.nan
                    
                    # Replace invalid values
                    factor_values = np.nan_to_num(factor_values, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Calculate performance metrics
                    ic = self._calculate_ic(factor_values, y)
                    return_spread = self._calculate_return_spread(factor_values, y)
                    portfolio_return = self._calculate_portfolio_return(factor_values, y)
                    
                    # Create factor information
                    factor_info = {
                        'name': factor_name,
                        'expression': str(ind),
                        'fitness': float(ind.fitness.values[0]),
                        'ic': float(ic),
                        'return_spread': float(return_spread),
                        'portfolio_return': float(portfolio_return),
                        'complexity': len(str(ind))
                    }
                    
                    self.best_factors.append(factor_info)
                    
                    self.logger.info(f"Factor {factor_name} generation completed, fitness: {factor_info['fitness']:.4f}")
            
            # Save factors
            self._save_factors()
            
            return self.best_factors
            
        except Exception as e:
            self.logger.error(f"Error during factor evolution: {str(e)}")
            return []
    
    def _prepare_data(self, price_data: pd.DataFrame, future_return_periods: int = 5) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare modeling data"""
        # Copy data to avoid modifying original data
        df = price_data.copy()
        
        # Ensure close price exists
        if 'close' not in df.columns:
            raise ValueError("Price data must contain 'close' column")
        
        # Calculate features
        feature_generators = {
            # Price features
            'open_close': lambda d: d['open'] / d['close'] - 1 if 'open' in d.columns else 0,
            'high_low': lambda d: d['high'] / d['low'] - 1 if 'high' in d.columns and 'low' in d.columns else 0,
            
            # Returns
            'returns_1d': lambda d: d['close'].pct_change(1),
            'returns_5d': lambda d: d['close'].pct_change(5),
            'returns_10d': lambda d: d['close'].pct_change(10),
            'returns_20d': lambda d: d['close'].pct_change(20),
            
            # Volatility
            'volatility_5d': lambda d: d['returns_1d'].rolling(window=5).std(),
            'volatility_20d': lambda d: d['returns_1d'].rolling(window=20).std(),
            
            # Moving averages
            'ma5': lambda d: d['close'].rolling(window=5).mean(),
            'ma10': lambda d: d['close'].rolling(window=10).mean(),
            'ma20': lambda d: d['close'].rolling(window=20).mean(),
            'ma60': lambda d: d['close'].rolling(window=60).mean(),
            
            # Moving average divergence
            'ma5_ma20': lambda d: d['ma5'] / d['ma20'] - 1,
            'ma10_ma60': lambda d: d['ma10'] / d['ma60'] - 1,
        }
        
        # Generate basic features
        for feature, generator in feature_generators.items():
            df[feature] = generator(df)
            
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate volume features
        if 'volume' in df.columns:
            df['volume_ma5'] = df['volume'].rolling(window=5).mean()
            df['volume_ma20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma5']
            df['volume_change'] = df['volume'].pct_change(5)
        
        # Calculate target value: future returns
        df['future_return'] = df['close'].pct_change(future_return_periods).shift(-future_return_periods)
        
        # Remove missing values
        df = df.dropna()
        
        # Extract features and targets
        feature_cols = [
            'open_close', 'high_low', 'returns_1d', 'returns_5d', 'returns_10d', 'returns_20d',
            'volatility_5d', 'volatility_20d', 'ma5_ma20', 'ma10_ma60', 'rsi'
        ]
        
        # Add volume features (if they exist)
        volume_features = ['volume_ratio', 'volume_change']
        for feature in volume_features:
            if feature in df.columns:
                feature_cols.append(feature)
        
        # Extract data
        X = df[feature_cols].values
        y = df['future_return'].values
        
        return X, y, feature_cols
    
    def _save_factors(self):
        """Save generated factors"""
        if not self.best_factors:
            return
        
        # Create save directory
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Save factor configuration
        factors_file = os.path.join(self.model_dir, "best_factors.json")
        with open(factors_file, 'w') as f:
            json.dump(self.best_factors, f, indent=2)
        
        self.logger.info(f"Saved {len(self.best_factors)} factors to {factors_file}")
    
    def load_factors(self):
        """Load saved factors"""
        factors_file = os.path.join(self.model_dir, "best_factors.json")
        
        if not os.path.exists(factors_file):
            self.logger.warning(f"Factor file {factors_file} does not exist")
            return []
        
        try:
            with open(factors_file, 'r') as f:
                self.best_factors = json.load(f)
            
            self.logger.info(f"Successfully loaded {len(self.best_factors)} factors")
            return self.best_factors
        
        except Exception as e:
            self.logger.error(f"Error loading factors: {str(e)}")
            return []
    
    def calculate_factor_values(self, price_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate factor values for latest data"""
        if not self.best_factors:
            self.load_factors()
            
        if not self.best_factors:
            return {}
        
        try:
            # Ensure data has all necessary features
            enhanced_df = self._ensure_all_features(price_data)
            
            # Calculate factor values
            factor_values = {}

            # Import needed functions, ensure they are available in local scope
            # Special functions for safety protection
            def protected_div(left, right):
                """Safe division, prevents division by zero error"""
                if isinstance(right, np.ndarray):
                    return np.where(np.abs(right) >= 1e-10, left / right, 1.0)
                else:
                    return left / right if abs(right) >= 1e-10 else 1.0

            def protected_sqrt(x):
                """Safe square root, handles negative numbers"""
                if isinstance(x, np.ndarray):
                    return np.where(x > 0, np.sqrt(x), 0.0)
                else:
                    return np.sqrt(x) if x > 0 else 0.0

            def protected_log(x):
                """Safe logarithm, handles negative numbers and zero"""
                if isinstance(x, np.ndarray):
                    return np.where(x > 0, np.log(x), 0.0)
                else:
                    return np.log(x) if x > 0 else 0.0

            def protected_inv(x):
                """Safe inverse, handles zero"""
                if isinstance(x, np.ndarray):
                    return np.where(np.abs(x) >= 1e-10, 1.0 / x, 0.0)
                else:
                    return 1.0 / x if abs(x) >= 1e-10 else 0.0
            
            # Time series functions
            def rolling_mean(x, window=10):
                """Rolling mean"""
                return pd.Series(x).rolling(window=window).mean().fillna(0).values

            def rolling_std(x, window=10):
                """Rolling standard deviation"""
                return pd.Series(x).rolling(window=window).std().fillna(0).values

            def ts_delta(x, period=1):
                """Time series difference"""
                return pd.Series(x).diff(period).fillna(0).values

            def ts_delay(x, period=1):
                """Time series lag"""
                return pd.Series(x).shift(period).fillna(0).values

            def ts_return(x, period=1):
                """Time series return"""
                return pd.Series(x).pct_change(period).fillna(0).values

            def ts_rank(x, window=10):
                """Time series rank"""
                return pd.Series(x).rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]).fillna(0).values
            
            # Create function namespace
            local_namespace = {
                # Basic operators
                'add': operator.add, 'sub': operator.sub, 'mul': operator.mul, 'div': protected_div, 'neg': operator.neg,
                
                # Mathematical functions
                'abs': abs, 'sqrt': protected_sqrt, 'log': protected_log, 'inv': protected_inv,
                
                # Specialized safety functions - directly defined in namespace
                'protected_div': protected_div,
                'protected_sqrt': protected_sqrt,
                'protected_log': protected_log,
                'protected_inv': protected_inv,
                
                # Time series functions
                'rolling_mean': rolling_mean, 'rolling_std': rolling_std,
                'ts_delta': ts_delta, 'ts_delay': ts_delay, 
                'ts_return': ts_return, 'ts_rank': ts_rank,
                
                # Time series functions with parameters
                'rolling_mean_5': lambda x: rolling_mean(x, window=5),
                'rolling_mean_10': lambda x: rolling_mean(x, window=10),
                'rolling_mean_20': lambda x: rolling_mean(x, window=20),
                'rolling_std_5': lambda x: rolling_std(x, window=5),
                'rolling_std_10': lambda x: rolling_std(x, window=10),
                'ts_delta_1': lambda x: ts_delta(x, period=1),
                'ts_delta_5': lambda x: ts_delta(x, period=5),
                'ts_delay_1': lambda x: ts_delay(x, period=1),
                'ts_delay_5': lambda x: ts_delay(x, period=5),
                'ts_return_1': lambda x: ts_return(x, period=1),
                'ts_return_5': lambda x: ts_return(x, period=5),
                'ts_rank_10': lambda x: ts_rank(x, window=10),
                
                # NumPy functions
                'mean': np.mean, 'std': np.std, 'min': np.min, 'max': np.max, 'sum': np.sum
            }
            
            # Calculate missing features (may be referenced by factor expressions)
            if 'open' in enhanced_df.columns and 'close' in enhanced_df.columns:
                valid_close = enhanced_df['close'] != 0
                enhanced_df['open_close'] = np.zeros(len(enhanced_df))
                if any(valid_close):
                    enhanced_df.loc[valid_close, 'open_close'] = enhanced_df.loc[valid_close, 'open'] / enhanced_df.loc[valid_close, 'close'] - 1
            
            if 'high' in enhanced_df.columns and 'low' in enhanced_df.columns:
                valid_low = enhanced_df['low'] != 0
                enhanced_df['high_low'] = np.zeros(len(enhanced_df))
                if any(valid_low):
                    enhanced_df.loc[valid_low, 'high_low'] = enhanced_df.loc[valid_low, 'high'] / enhanced_df.loc[valid_low, 'low'] - 1
            
            # Add data features to namespace
            for col in enhanced_df.columns:
                if isinstance(enhanced_df[col], pd.Series):
                    local_namespace[col] = enhanced_df[col].values
            
            # Calculate values for each factor
            for factor_info in self.best_factors:
                try:
                    factor_name = factor_info['name']
                    expr_str = factor_info['expression']
                    
                    # Try to calculate factor values
                    try:
                        # Use safe eval method
                        code = compile(expr_str, '<string>', 'eval')
                        values = eval(code, {"__builtins__": {}}, local_namespace)
                        
                        # If array, use directly; if single value, expand to array
                        if not isinstance(values, np.ndarray) and not isinstance(values, list):
                            values = np.full(len(enhanced_df), values)
                        
                        # Replace invalid values
                        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        # Standardize factor values
                        if len(values) > 0:
                            std_val = np.std(values)
                            if std_val > 1e-8:  # Avoid dividing by near-zero numbers
                                values = (values - np.mean(values)) / std_val
                        
                        # Store factor values
                        factor_values[factor_name] = values
                        
                    except Exception as eval_error:
                        self.logger.error(f"Error compiling factor {factor_name}: {eval_error}")
                    
                except Exception as e:
                    self.logger.error(f"Error calculating factor {factor_info['name']} values: {str(e)}")
            
            return factor_values
        
        except Exception as e:
            self.logger.error(f"Error occurred during factor value calculation: {str(e)}")
            return {}

    def _ensure_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all potentially needed features exist"""
        enhanced_df = df.copy()
        
        # Ensure basic columns exist and are named correctly (case sensitivity issues)
        column_mappings = {
            'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Volume': 'volume'
        }
        
        for original, target in column_mappings.items():
            if target not in enhanced_df.columns and original in enhanced_df.columns:
                enhanced_df[target] = enhanced_df[original]
        
        # Calculate basic features
        if 'close' in enhanced_df.columns:
            # Moving averages
            for window in [5, 10, 20, 60]:
                col_name = f'ma{window}'
                if col_name not in enhanced_df.columns:
                    enhanced_df[col_name] = enhanced_df['close'].rolling(window=window).mean()
            
            # Return indicators
            for period in [1, 5, 10, 20]:
                col_name = f'returns_{period}d'
                if col_name not in enhanced_df.columns:
                    enhanced_df[col_name] = enhanced_df['close'].pct_change(period)
            
            # Volatility indicators
            for window in [5, 10, 20]:
                col_name = f'volatility_{window}d'
                if col_name not in enhanced_df.columns:
                    enhanced_df[col_name] = enhanced_df['close'].pct_change().rolling(window=window).std()
        
            # Moving average crossover indicators
            ma_pairs = [(5, 20), (10, 60)]
            for short_win, long_win in ma_pairs:
                short_ma = f'ma{short_win}'
                long_ma = f'ma{long_win}'
                cross_name = f'{short_ma}_{long_ma}'
                
                if cross_name not in enhanced_df.columns:
                    enhanced_df[cross_name] = enhanced_df[short_ma] / enhanced_df[long_ma] - 1
            
            # RSI indicator
            if 'rsi' not in enhanced_df.columns:
                delta = enhanced_df['close'].diff()
                gain = (delta.where(delta > 0, 0)).fillna(0)
                loss = (-delta.where(delta < 0, 0)).fillna(0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                enhanced_df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD indicator
            if 'macd' not in enhanced_df.columns:
                exp1 = enhanced_df['close'].ewm(span=12, adjust=False).mean()
                exp2 = enhanced_df['close'].ewm(span=26, adjust=False).mean()
                enhanced_df['macd'] = exp1 - exp2
                enhanced_df['macd_signal'] = enhanced_df['macd'].ewm(span=9, adjust=False).mean()
                enhanced_df['macd_hist'] = enhanced_df['macd'] - enhanced_df['macd_signal']
        
        # Fill NaN values
        enhanced_df = enhanced_df.ffill().bfill().fillna(0)
        
        return enhanced_df

    def _create_function_namespace(self) -> Dict[str, Any]:
        """Create namespace containing all necessary functions"""
        namespace = {
            # Basic operators
            'add': operator.add, 'sub': operator.sub, 'mul': operator.mul, 'div': protected_div, 'neg': operator.neg,
            
            # Mathematical functions
            'abs': abs, 'sqrt': protected_sqrt, 'log': protected_log, 'inv': protected_inv,
            
            # Time series functions
            'rolling_mean': rolling_mean, 'rolling_std': rolling_std,
            'ts_delta': ts_delta, 'ts_delay': ts_delay, 
            'ts_return': ts_return, 'ts_rank': ts_rank,
            
            # Time series functions with parameters
            'rolling_mean_5': lambda x: rolling_mean(x, window=5),
            'rolling_mean_10': lambda x: rolling_mean(x, window=10),
            'rolling_mean_20': lambda x: rolling_mean(x, window=20),
            'rolling_std_5': lambda x: rolling_std(x, window=5),
            'rolling_std_10': lambda x: rolling_std(x, window=10),
            'ts_delta_1': lambda x: ts_delta(x, period=1),
            'ts_delta_5': lambda x: ts_delta(x, period=5),
            'ts_delay_1': lambda x: ts_delay(x, period=1),
            'ts_delay_5': lambda x: ts_delay(x, period=5),
            'ts_return_1': lambda x: ts_return(x, period=1),
            'ts_return_5': lambda x: ts_return(x, period=5),
            'ts_rank_10': lambda x: ts_rank(x, window=10),
            
            # NumPy functions
            'mean': np.mean, 'std': np.std, 'min': np.min, 'max': np.max, 'sum': np.sum
        }
        
        return namespace

    def _extract_variable_names(self, expr_str: str) -> List[str]:
        """Extract variable names from expression string"""
        # Basic variable patterns
        var_pattern = r'[a-zA-Z][a-zA-Z0-9_]*'
        
        # Try to extract all possible variable names from expression
        potential_vars = re.findall(var_pattern, expr_str)
        
        # Exclude basic operators, function names and constants
        exclude_list = [
            'add', 'sub', 'mul', 'div', 'neg', 'abs', 
            'sqrt', 'log', 'inv',
            'rolling_mean', 'rolling_std', 'ts_delta', 'ts_delay', 
            'ts_return', 'ts_rank', 'window', 'period',
            'ARG', 'True', 'False', 'None'
        ]
        
        # Filter out non-variable names
        variables = [var for var in potential_vars if var not in exclude_list and not var.startswith('ARG')]
        
        return list(set(variables))  # Remove duplicates
        

class FactorAgent:
    """Factor mining Agent, integrated into existing system"""
    
    def __init__(self, model_dir: str = 'factors'):
        """Initialize factor Agent"""
        self.factor_module = FactorMiningModule(model_dir=model_dir)
        self.logger = setup_logger('factor_agent')
        self.factors_generated = False
    
    def generate_factors(self, price_data, n_factors=5, **kwargs):
        """Generate factors"""
        try:
            # Extract and prepare parameters
            evolve_params = {
                'price_data': price_data,
                'n_factors': n_factors
            }
            
            # Add other possible parameters
            allowed_params = ['population_size', 'n_generations', 'future_return_periods', 'min_fitness']
            for param in allowed_params:
                if param in kwargs:
                    evolve_params[param] = kwargs[param]
            
            # Generate factors
            factors = self.factor_module.evolve_factors(**evolve_params)
            
            self.factors_generated = len(factors) > 0
            return factors
            
        except Exception as e:
            self.logger.error(f"Error generating factors: {str(e)}")
            return []
    
    def load_factors(self):
        """Load saved factors"""
        try:
            factors = self.factor_module.load_factors()
            self.factors_generated = len(factors) > 0
            return factors
            
        except Exception as e:
            self.logger.error(f"Error loading factors: {str(e)}")
            return []
    
    def get_factor_values(self, price_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get factor values"""
        try:
            # If no factors generated, try to load
            if not self.factors_generated:
                self.load_factors()
            
            # If still no factors, return empty dictionary
            if not self.factors_generated:
                return {}
            
            # Calculate factor values
            factor_values = self.factor_module.calculate_factor_values(price_data)
            return factor_values
            
        except Exception as e:
            self.logger.error(f"Error getting factor values: {str(e)}")
            return {}
    
    def generate_signals(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signals"""
        signals = {}
        
        try:
            # Get factor values
            factor_values = self.get_factor_values(price_data)
            
            if not factor_values:
                return {'signal': 'neutral', 'confidence': 0.5}
            
            # Get latest factor values
            latest_values = {name: values[-1] for name, values in factor_values.items()}
            
            # Record all factor signals
            factor_signals = []
            
            # Analyze each factor's signal
            for factor_name, factor_value in latest_values.items():
                # Factor trend (compared to previous 5 days)
                factor_series = factor_values[factor_name]
                factor_trend = factor_value - np.mean(factor_series[-6:-1]) if len(factor_series) > 5 else 0
                
                # Factor strength (standardized)
                factor_std = np.std(factor_series[-20:]) if len(factor_series) > 20 else 1
                factor_strength = factor_value / (factor_std + 1e-8)
                
                # Determine signal
                if factor_value > 0.5 or factor_trend > 0.2:
                    signal = 'bullish'
                    confidence = min(0.5 + abs(factor_value) * 0.3, 0.9)
                elif factor_value < -0.5 or factor_trend < -0.2:
                    signal = 'bearish'
                    confidence = min(0.5 + abs(factor_value) * 0.3, 0.9)
                else:
                    signal = 'neutral'
                    confidence = 0.5
                
                factor_signals.append({
                    'name': factor_name,
                    'value': float(factor_value),
                    'trend': float(factor_trend),
                    'strength': float(factor_strength),
                    'signal': signal,
                    'confidence': float(confidence)
                })
            
            # Calculate comprehensive signal
            bullish_count = sum(1 for s in factor_signals if s['signal'] == 'bullish')
            bearish_count = sum(1 for s in factor_signals if s['signal'] == 'bearish')
            
            # Weighted average confidence
            avg_confidence = sum(s['confidence'] for s in factor_signals) / len(factor_signals)
            
            # Determine final signal
            if bullish_count > bearish_count:
                final_signal = 'bullish'
                final_confidence = min(0.5 + 0.1 * bullish_count, avg_confidence)
            elif bearish_count > bullish_count:
                final_signal = 'bearish'
                final_confidence = min(0.5 + 0.1 * bearish_count, avg_confidence)
            else:
                final_signal = 'neutral'
                final_confidence = 0.5
            
            # Return signal
            signals = {
                'signal': final_signal,
                'confidence': final_confidence,
                'factor_signals': factor_signals,
                'reasoning': self._generate_reasoning(factor_signals, final_signal, final_confidence)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating trading signals: {str(e)}")
            signals = {'signal': 'neutral', 'confidence': 0.5}
        
        return signals
    
    def _generate_reasoning(self, factor_signals: List[Dict[str, Any]], 
                           final_signal: str, final_confidence: float) -> str:
        """Generate signal explanation"""
        reasoning_parts = []
        
        # Count signals
        bullish_count = sum(1 for s in factor_signals if s['signal'] == 'bullish')
        bearish_count = sum(1 for s in factor_signals if s['signal'] == 'bearish')
        neutral_count = sum(1 for s in factor_signals if s['signal'] == 'neutral')
        
        # Add comprehensive analysis
        reasoning_parts.append(f"Genetic programming factor signals: {len(factor_signals)} factors, {bullish_count} bullish, {bearish_count} bearish, {neutral_count} neutral")
        
        # Add strongest factor analysis
        strongest_bullish = [s for s in factor_signals if s['signal'] == 'bullish']
        strongest_bearish = [s for s in factor_signals if s['signal'] == 'bearish']
        
        if strongest_bullish:
            top_bullish = max(strongest_bullish, key=lambda s: s['confidence'])
            reasoning_parts.append(f"Strongest bullish factor: {top_bullish['name']}, factor value: {top_bullish['value']:.2f}, confidence: {top_bullish['confidence']:.2f}")
        
        if strongest_bearish:
            top_bearish = max(strongest_bearish, key=lambda s: s['confidence'])
            reasoning_parts.append(f"Strongest bearish factor: {top_bearish['name']}, factor value: {top_bearish['value']:.2f}, confidence: {top_bearish['confidence']:.2f}")
        
        # Add final conclusion
        reasoning_parts.append(f"Comprehensive genetic programming factor analysis yields {final_signal} signal, confidence: {final_confidence:.2f}")
        
        return "; ".join(reasoning_parts)