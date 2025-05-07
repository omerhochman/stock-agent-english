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

# 设置警告级别和随机种子
warnings.filterwarnings("ignore", category=RuntimeWarning)
logger = setup_logger('factor_mining')
random.seed(42)
np.random.seed(42)

# 安全保护的特殊函数
def protected_div(left, right):
    """安全除法，防止除以0错误"""
    return left / right if abs(right) >= 1e-10 else 1.0

def protected_sqrt(x):
    """安全平方根，处理负数"""
    return np.sqrt(x) if x > 0 else 0.0

def protected_log(x):
    """安全对数，处理负数和0"""
    return np.log(x) if x > 0 else 0.0

def protected_inv(x):
    """安全倒数，处理0"""
    return 1.0 / x if abs(x) >= 1e-10 else 0.0

# 时间序列函数
def rolling_mean(x, window=10):
    """滚动平均"""
    return pd.Series(x).rolling(window=window).mean().fillna(0).values

def rolling_std(x, window=10):
    """滚动标准差"""
    return pd.Series(x).rolling(window=window).std().fillna(0).values

def ts_delta(x, period=1):
    """时间序列差分"""
    return pd.Series(x).diff(period).fillna(0).values

def ts_delay(x, period=1):
    """时间序列滞后"""
    return pd.Series(x).shift(period).fillna(0).values

def ts_return(x, period=1):
    """时间序列收益率"""
    return pd.Series(x).pct_change(period).fillna(0).values

def ts_rank(x, window=10):
    """时间序列排名"""
    return pd.Series(x).rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]).fillna(0).values


class FactorMiningModule:
    """因子挖掘主类，使用遗传编程自动生成和筛选因子"""
    
    def __init__(self, model_dir: str = 'factors'):
        """初始化因子挖掘模块"""
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.best_factors = []
        self.pset = None
        self.toolbox = None
        self.logger = setup_logger('factor_mining')
    
    def _setup_primitives(self, input_names: List[str]):
        """设置遗传编程的原语集"""
        # 创建原语集
        self.pset = gp.PrimitiveSetTyped("MAIN", [float] * len(input_names), float)
        
        # 重命名输入变量
        for i, name in enumerate(input_names):
            self.pset.renameArguments(**{f'ARG{i}': name})
        
        # 添加基本运算符
        self.pset.addPrimitive(operator.add, [float, float], float)
        self.pset.addPrimitive(operator.sub, [float, float], float)
        self.pset.addPrimitive(operator.mul, [float, float], float)
        self.pset.addPrimitive(protected_div, [float, float], float)
        
        # 添加数学函数
        self.pset.addPrimitive(operator.neg, [float], float)
        self.pset.addPrimitive(abs, [float], float)
        self.pset.addPrimitive(protected_sqrt, [float], float)
        self.pset.addPrimitive(protected_log, [float], float)
        self.pset.addPrimitive(protected_inv, [float], float)
        
        # 创建和添加时间序列函数
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
        
        # 添加常数
        self.pset.addEphemeralConstant("rand", lambda: random.uniform(-1, 1), float)
    
    def _evaluate_factor(self, individual, X: np.ndarray, y: np.ndarray, 
                    eval_func: Callable, eval_window: int = 20) -> Tuple[float,]:
        """评估因子质量"""
        # 将个体转换为函数
        func = self.toolbox.compile(expr=individual)
        
        try:
            # 计算因子值
            factor_values = np.zeros(len(y))
            
            for i in range(len(y)):
                # 获取当前样本的所有特征
                sample = [X[i, j] for j in range(X.shape[1])]
                
                # 计算因子值
                try:
                    factor_values[i] = func(*sample)
                except:
                    factor_values[i] = np.nan
            
            # 处理无穷大和NaN
            factor_values = np.nan_to_num(factor_values, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 检查因子值是否有效
            if np.all(factor_values == 0.0) or np.all(np.isclose(factor_values, factor_values[0])):
                # 如果所有值都相同或全为零，给予低分
                return (-0.5,)
                
            # 计算评估指标
            try:
                score = eval_func(factor_values, y)
                
                # 确保分数是一个有限数
                if not np.isfinite(score):
                    score = -0.5
                    
                # 计算因子复杂度惩罚
                complexity = len(str(individual))
                complexity_penalty = np.clip(complexity / 200, 0, 0.5)
                
                # 最终分数
                final_score = max(-1.0, min(1.0, score - complexity_penalty))
                
                return (final_score,)
            except Exception as e:
                self.logger.error(f"计算因子得分时出错: {str(e)}")
                return (-0.5,)
                
        except Exception:
            # 如果出错，返回最低分
            return (-0.5,)
    
    def _calculate_ic(self, factor_values: np.ndarray, future_returns: np.ndarray, 
                     window: int = 20) -> float:
        """计算因子的IC值（信息系数）"""
        # 创建pandas Series便于计算
        factor_series = pd.Series(factor_values)
        returns_series = pd.Series(future_returns)
        
        # 清除无效值
        valid_indices = ~(np.isnan(factor_values) | np.isnan(future_returns))
        factor_series = factor_series[valid_indices]
        returns_series = returns_series[valid_indices]
        
        if len(factor_series) < window:
            return 0.0
        
        # 计算滚动相关系数
        ic_series = factor_series.rolling(window).corr(returns_series)
        
        # 返回平均IC值
        mean_ic = ic_series.mean()
        return 0.0 if np.isnan(mean_ic) else mean_ic
    
    def _calculate_return_spread(self, factor_values: np.ndarray, future_returns: np.ndarray, 
                               quantiles: int = 5) -> float:
        """计算因子分组回报差异"""
        if len(factor_values) < quantiles * 2:
            return 0.0
        
        # 清除无效值
        valid_indices = ~(np.isnan(factor_values) | np.isnan(future_returns))
        factor_values = factor_values[valid_indices]
        future_returns = future_returns[valid_indices]
        
        if len(factor_values) < quantiles * 2:
            return 0.0
        
        # 对因子值进行分组
        df = pd.DataFrame({'factor': factor_values, 'returns': future_returns})
        df['quantile'] = pd.qcut(df['factor'], q=quantiles, labels=False, duplicates='drop')
        
        # 计算每组平均回报
        group_returns = df.groupby('quantile')['returns'].mean()
        
        # 高分位与低分位的回报差异
        if len(group_returns) < quantiles:
            return 0.0
            
        return group_returns.iloc[-1] - group_returns.iloc[0]
    
    def _calculate_portfolio_return(self, factor_values: np.ndarray, future_returns: np.ndarray, 
                                  top_percentile: float = 0.2) -> float:
        """计算基于因子的多空组合收益"""
        if len(factor_values) < 10:
            return 0.0
        
        # 清除无效值
        valid_indices = ~(np.isnan(factor_values) | np.isnan(future_returns))
        factor_values = factor_values[valid_indices]
        future_returns = future_returns[valid_indices]
        
        if len(factor_values) < 10:
            return 0.0
        
        # 对因子值进行排序
        sorted_indices = np.argsort(factor_values)
        
        # 选择多头和空头组合
        n_assets = len(sorted_indices)
        n_select = max(1, int(n_assets * top_percentile))
        
        # 多头：因子值最高的股票
        long_indices = sorted_indices[-n_select:]
        long_return = np.mean(future_returns[long_indices])
        
        # 空头：因子值最低的股票
        short_indices = sorted_indices[:n_select]
        short_return = np.mean(future_returns[short_indices])
        
        # 多空组合收益
        return long_return - short_return
    
    def _factor_fitness(self, factor_values: np.ndarray, future_returns: np.ndarray) -> float:
        """综合计算因子质量"""
        try:
            # 清除无效值
            mask = ~(np.isnan(factor_values) | np.isnan(future_returns) | 
                    np.isinf(factor_values) | np.isinf(future_returns))
            
            if np.sum(mask) < 10:  # 如果有效数据点太少
                return 0.0
                
            factor_values_clean = factor_values[mask]
            future_returns_clean = future_returns[mask]
            
            # 计算多个指标
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
            
            # 计算因子自相关性（稳定性）
            try:
                factor_series = pd.Series(factor_values_clean)
                factor_autocorr = factor_series.autocorr(lag=1)
                if np.isnan(factor_autocorr) or not np.isfinite(factor_autocorr):
                    factor_autocorr = 0
            except:
                factor_autocorr = 0
            
            # 计算因子单调性
            try:
                monotonicity = self._calculate_monotonicity(factor_values_clean, future_returns_clean)
                if not np.isfinite(monotonicity):
                    monotonicity = 0.0
            except:
                monotonicity = 0.0
            
            # 综合评分（各项指标加权）
            score = (
                0.4 * abs(ic) +                 
                0.3 * abs(return_spread) +      
                0.2 * abs(portfolio_return) +   
                0.05 * abs(factor_autocorr) +   
                0.05 * abs(monotonicity)        
            )
            
            # 确保返回有限值
            return float(0.0 if not np.isfinite(score) else score)
        
        except Exception as e:
            self.logger.error(f"计算因子适应度时出错: {str(e)}")
            return 0.0
    
    def _calculate_monotonicity(self, factor_values: np.ndarray, future_returns: np.ndarray, 
                           n_groups: int = 5) -> float:
        """计算因子与未来收益之间的单调关系强度"""
        try:
            if len(factor_values) < n_groups * 2:
                return 0.0
                
            # 创建数据框
            df = pd.DataFrame({'factor': factor_values, 'returns': future_returns})
            
            # 安全地进行分组
            try:
                df['quantile'] = pd.qcut(df['factor'], q=n_groups, labels=False, duplicates='drop')
            except:
                # 如果分位数划分失败，尝试减少组数
                try:
                    n_groups = max(2, n_groups - 1)
                    df['quantile'] = pd.qcut(df['factor'], q=n_groups, labels=False, duplicates='drop')
                except:
                    # 如果仍然失败，使用等宽分组
                    df['quantile'] = pd.cut(df['factor'], bins=n_groups, labels=False)
            
            # 计算每组平均回报
            group_returns = df.groupby('quantile')['returns'].mean()
            
            if len(group_returns) < 2:
                return 0.0
                
            # 计算排序相关性
            try:
                from scipy import stats
                corr, _ = stats.spearmanr(group_returns.index, group_returns.values)
                return 0.0 if np.isnan(corr) else corr
            except:
                # 如果无法计算相关性，计算简单的趋势
                trend = (group_returns.iloc[-1] - group_returns.iloc[0]) / (len(group_returns) - 1)
                return np.clip(trend, -1.0, 1.0)
        
        except Exception:
            return 0.0

    def evolve_factors(self, price_data: pd.DataFrame, n_factors: int = 5, 
                      population_size: int = 100, n_generations: int = 50,
                      future_return_periods: int = 5, min_fitness: float = 0.05):
        """使用遗传编程进化因子"""
        # 准备特征数据
        X, y, feature_names = self._prepare_data(price_data, future_return_periods)
        
        # 设置原语集
        self._setup_primitives(feature_names)
        
        # 创建适应度函数
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        
        # 创建个体
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
        
        # 创建工具箱
        self.toolbox = base.Toolbox()
        
        # 注册表达式生成器
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=4)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate_factor, X=X, y=y, eval_func=self._factor_fitness)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        
        # 创建初始种群
        pop = self.toolbox.population(n=population_size)
        hof = tools.HallOfFame(n_factors)
        
        # 统计对象
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        stats.register("std", np.std)
        
        # 启动进化
        try:
            self.logger.info(f"开始因子进化，种群大小: {population_size}，迭代次数: {n_generations}")
            pop, _ = algorithms.eaSimple(pop, self.toolbox, cxpb=0.5, mutpb=0.1,
                                         ngen=n_generations, stats=stats, halloffame=hof,
                                         verbose=True)
            
            # 提取最佳因子
            self.best_factors = []
            for i, ind in enumerate(hof):
                if ind.fitness.values[0] > min_fitness:
                    factor_func = self.toolbox.compile(expr=ind)
                    factor_name = f"GP_Factor_{i+1}"
                    
                    # 计算因子值
                    factor_values = np.zeros(len(y))
                    for j in range(len(y)):
                        sample = [X[j, k] for k in range(X.shape[1])]
                        try:
                            factor_values[j] = factor_func(*sample)
                        except:
                            factor_values[j] = np.nan
                    
                    # 替换无效值
                    factor_values = np.nan_to_num(factor_values, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # 计算性能指标
                    ic = self._calculate_ic(factor_values, y)
                    return_spread = self._calculate_return_spread(factor_values, y)
                    portfolio_return = self._calculate_portfolio_return(factor_values, y)
                    
                    # 创建因子信息
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
                    
                    self.logger.info(f"因子 {factor_name} 生成完成，适应度: {factor_info['fitness']:.4f}")
            
            # 保存因子
            self._save_factors()
            
            return self.best_factors
            
        except Exception as e:
            self.logger.error(f"因子进化过程中出错: {str(e)}")
            return []
    
    def _prepare_data(self, price_data: pd.DataFrame, future_return_periods: int = 5) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """准备建模数据"""
        # 复制数据，避免修改原始数据
        df = price_data.copy()
        
        # 确保有收盘价
        if 'close' not in df.columns:
            raise ValueError("价格数据必须包含'close'列")
        
        # 计算特征
        feature_generators = {
            # 价格特征
            'open_close': lambda d: d['open'] / d['close'] - 1 if 'open' in d.columns else 0,
            'high_low': lambda d: d['high'] / d['low'] - 1 if 'high' in d.columns and 'low' in d.columns else 0,
            
            # 收益率
            'returns_1d': lambda d: d['close'].pct_change(1),
            'returns_5d': lambda d: d['close'].pct_change(5),
            'returns_10d': lambda d: d['close'].pct_change(10),
            'returns_20d': lambda d: d['close'].pct_change(20),
            
            # 波动率
            'volatility_5d': lambda d: d['returns_1d'].rolling(window=5).std(),
            'volatility_20d': lambda d: d['returns_1d'].rolling(window=20).std(),
            
            # 移动平均
            'ma5': lambda d: d['close'].rolling(window=5).mean(),
            'ma10': lambda d: d['close'].rolling(window=10).mean(),
            'ma20': lambda d: d['close'].rolling(window=20).mean(),
            'ma60': lambda d: d['close'].rolling(window=60).mean(),
            
            # 均线差离
            'ma5_ma20': lambda d: d['ma5'] / d['ma20'] - 1,
            'ma10_ma60': lambda d: d['ma10'] / d['ma60'] - 1,
        }
        
        # 生成基本特征
        for feature, generator in feature_generators.items():
            df[feature] = generator(df)
            
        # 计算RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 计算成交量特征
        if 'volume' in df.columns:
            df['volume_ma5'] = df['volume'].rolling(window=5).mean()
            df['volume_ma20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma5']
            df['volume_change'] = df['volume'].pct_change(5)
        
        # 计算目标值：未来收益率
        df['future_return'] = df['close'].pct_change(future_return_periods).shift(-future_return_periods)
        
        # 删除缺失值
        df = df.dropna()
        
        # 提取特征和目标
        feature_cols = [
            'open_close', 'high_low', 'returns_1d', 'returns_5d', 'returns_10d', 'returns_20d',
            'volatility_5d', 'volatility_20d', 'ma5_ma20', 'ma10_ma60', 'rsi'
        ]
        
        # 添加成交量特征（如果存在）
        volume_features = ['volume_ratio', 'volume_change']
        for feature in volume_features:
            if feature in df.columns:
                feature_cols.append(feature)
        
        # 提取数据
        X = df[feature_cols].values
        y = df['future_return'].values
        
        return X, y, feature_cols
    
    def _save_factors(self):
        """保存生成的因子"""
        if not self.best_factors:
            return
        
        # 创建保存目录
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 保存因子配置
        factors_file = os.path.join(self.model_dir, "best_factors.json")
        with open(factors_file, 'w') as f:
            json.dump(self.best_factors, f, indent=2)
        
        self.logger.info(f"已将{len(self.best_factors)}个因子保存至 {factors_file}")
    
    def load_factors(self):
        """加载已保存的因子"""
        factors_file = os.path.join(self.model_dir, "best_factors.json")
        
        if not os.path.exists(factors_file):
            self.logger.warning(f"因子文件 {factors_file} 不存在")
            return []
        
        try:
            with open(factors_file, 'r') as f:
                self.best_factors = json.load(f)
            
            self.logger.info(f"成功加载{len(self.best_factors)}个因子")
            return self.best_factors
        
        except Exception as e:
            self.logger.error(f"加载因子时出错: {str(e)}")
            return []
    
    def calculate_factor_values(self, price_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """计算最新数据的因子值"""
        if not self.best_factors:
            self.load_factors()
            
        if not self.best_factors:
            return {}
        
        try:
            # 确保数据中有所有必要的特征
            enhanced_df = self._ensure_all_features(price_data)
            
            # 计算因子值
            factor_values = {}

            # 导入需要的函数，确保在局部作用域可用
            # 安全保护的特殊函数
            def protected_div(left, right):
                """安全除法，防止除以0错误"""
                if isinstance(right, np.ndarray):
                    return np.where(np.abs(right) >= 1e-10, left / right, 1.0)
                else:
                    return left / right if abs(right) >= 1e-10 else 1.0

            def protected_sqrt(x):
                """安全平方根，处理负数"""
                if isinstance(x, np.ndarray):
                    return np.where(x > 0, np.sqrt(x), 0.0)
                else:
                    return np.sqrt(x) if x > 0 else 0.0

            def protected_log(x):
                """安全对数，处理负数和0"""
                if isinstance(x, np.ndarray):
                    return np.where(x > 0, np.log(x), 0.0)
                else:
                    return np.log(x) if x > 0 else 0.0

            def protected_inv(x):
                """安全倒数，处理0"""
                if isinstance(x, np.ndarray):
                    return np.where(np.abs(x) >= 1e-10, 1.0 / x, 0.0)
                else:
                    return 1.0 / x if abs(x) >= 1e-10 else 0.0
            
            # 时间序列函数
            def rolling_mean(x, window=10):
                """滚动平均"""
                return pd.Series(x).rolling(window=window).mean().fillna(0).values

            def rolling_std(x, window=10):
                """滚动标准差"""
                return pd.Series(x).rolling(window=window).std().fillna(0).values

            def ts_delta(x, period=1):
                """时间序列差分"""
                return pd.Series(x).diff(period).fillna(0).values

            def ts_delay(x, period=1):
                """时间序列滞后"""
                return pd.Series(x).shift(period).fillna(0).values

            def ts_return(x, period=1):
                """时间序列收益率"""
                return pd.Series(x).pct_change(period).fillna(0).values

            def ts_rank(x, window=10):
                """时间序列排名"""
                return pd.Series(x).rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]).fillna(0).values
            
            # 创建函数命名空间
            local_namespace = {
                # 基本运算符
                'add': operator.add, 'sub': operator.sub, 'mul': operator.mul, 'div': protected_div, 'neg': operator.neg,
                
                # 数学函数
                'abs': abs, 'sqrt': protected_sqrt, 'log': protected_log, 'inv': protected_inv,
                
                # 专用安全函数 - 直接定义在命名空间中
                'protected_div': protected_div,
                'protected_sqrt': protected_sqrt,
                'protected_log': protected_log,
                'protected_inv': protected_inv,
                
                # 时间序列函数
                'rolling_mean': rolling_mean, 'rolling_std': rolling_std,
                'ts_delta': ts_delta, 'ts_delay': ts_delay, 
                'ts_return': ts_return, 'ts_rank': ts_rank,
                
                # 带参数的时间序列函数
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
                
                # NumPy函数
                'mean': np.mean, 'std': np.std, 'min': np.min, 'max': np.max, 'sum': np.sum
            }
            
            # 计算缺失的特征（可能被因子表达式引用）
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
            
            # 将数据特征添加到命名空间
            for col in enhanced_df.columns:
                if isinstance(enhanced_df[col], pd.Series):
                    local_namespace[col] = enhanced_df[col].values
            
            # 计算每个因子的值
            for factor_info in self.best_factors:
                try:
                    factor_name = factor_info['name']
                    expr_str = factor_info['expression']
                    
                    # 尝试计算因子值
                    try:
                        # 使用安全的eval方式
                        code = compile(expr_str, '<string>', 'eval')
                        values = eval(code, {"__builtins__": {}}, local_namespace)
                        
                        # 如果是数组，直接使用；如果是单个值，扩展为数组
                        if not isinstance(values, np.ndarray) and not isinstance(values, list):
                            values = np.full(len(enhanced_df), values)
                        
                        # 替换无效值
                        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        # 标准化因子值
                        if len(values) > 0:
                            std_val = np.std(values)
                            if std_val > 1e-8:  # 避免除以接近零的数
                                values = (values - np.mean(values)) / std_val
                        
                        # 存储因子值
                        factor_values[factor_name] = values
                        
                    except Exception as eval_error:
                        self.logger.error(f"编译因子 {factor_name} 出错: {eval_error}")
                    
                except Exception as e:
                    self.logger.error(f"计算因子 {factor_info['name']} 值时出错: {str(e)}")
            
            return factor_values
        
        except Exception as e:
            self.logger.error(f"计算因子值过程中发生错误: {str(e)}")
            return {}

    def _ensure_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """确保所有可能需要的特征都存在"""
        enhanced_df = df.copy()
        
        # 确保基本列存在并命名正确（大小写问题）
        column_mappings = {
            'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Volume': 'volume'
        }
        
        for original, target in column_mappings.items():
            if target not in enhanced_df.columns and original in enhanced_df.columns:
                enhanced_df[target] = enhanced_df[original]
        
        # 计算基本特征
        if 'close' in enhanced_df.columns:
            # 移动平均线
            for window in [5, 10, 20, 60]:
                col_name = f'ma{window}'
                if col_name not in enhanced_df.columns:
                    enhanced_df[col_name] = enhanced_df['close'].rolling(window=window).mean()
            
            # 收益率指标
            for period in [1, 5, 10, 20]:
                col_name = f'returns_{period}d'
                if col_name not in enhanced_df.columns:
                    enhanced_df[col_name] = enhanced_df['close'].pct_change(period)
            
            # 波动率指标
            for window in [5, 10, 20]:
                col_name = f'volatility_{window}d'
                if col_name not in enhanced_df.columns:
                    enhanced_df[col_name] = enhanced_df['close'].pct_change().rolling(window=window).std()
        
            # 均线交叉指标
            ma_pairs = [(5, 20), (10, 60)]
            for short_win, long_win in ma_pairs:
                short_ma = f'ma{short_win}'
                long_ma = f'ma{long_win}'
                cross_name = f'{short_ma}_{long_ma}'
                
                if cross_name not in enhanced_df.columns:
                    enhanced_df[cross_name] = enhanced_df[short_ma] / enhanced_df[long_ma] - 1
            
            # RSI指标
            if 'rsi' not in enhanced_df.columns:
                delta = enhanced_df['close'].diff()
                gain = (delta.where(delta > 0, 0)).fillna(0)
                loss = (-delta.where(delta < 0, 0)).fillna(0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                enhanced_df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD指标
            if 'macd' not in enhanced_df.columns:
                exp1 = enhanced_df['close'].ewm(span=12, adjust=False).mean()
                exp2 = enhanced_df['close'].ewm(span=26, adjust=False).mean()
                enhanced_df['macd'] = exp1 - exp2
                enhanced_df['macd_signal'] = enhanced_df['macd'].ewm(span=9, adjust=False).mean()
                enhanced_df['macd_hist'] = enhanced_df['macd'] - enhanced_df['macd_signal']
        
        # 填充NaN值
        enhanced_df = enhanced_df.ffill().bfill().fillna(0)
        
        return enhanced_df

    def _create_function_namespace(self) -> Dict[str, Any]:
        """创建包含所有必要函数的命名空间"""
        namespace = {
            # 基本运算符
            'add': operator.add, 'sub': operator.sub, 'mul': operator.mul, 'div': protected_div, 'neg': operator.neg,
            
            # 数学函数
            'abs': abs, 'sqrt': protected_sqrt, 'log': protected_log, 'inv': protected_inv,
            
            # 时间序列函数
            'rolling_mean': rolling_mean, 'rolling_std': rolling_std,
            'ts_delta': ts_delta, 'ts_delay': ts_delay, 
            'ts_return': ts_return, 'ts_rank': ts_rank,
            
            # 带参数的时间序列函数
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
            
            # NumPy函数
            'mean': np.mean, 'std': np.std, 'min': np.min, 'max': np.max, 'sum': np.sum
        }
        
        return namespace

    def _extract_variable_names(self, expr_str: str) -> List[str]:
        """从表达式字符串中提取变量名"""
        # 基本变量模式
        var_pattern = r'[a-zA-Z][a-zA-Z0-9_]*'
        
        # 尝试从表达式中提取所有可能的变量名
        potential_vars = re.findall(var_pattern, expr_str)
        
        # 排除基本运算符、函数名和常量
        exclude_list = [
            'add', 'sub', 'mul', 'div', 'neg', 'abs', 
            'sqrt', 'log', 'inv',
            'rolling_mean', 'rolling_std', 'ts_delta', 'ts_delay', 
            'ts_return', 'ts_rank', 'window', 'period',
            'ARG', 'True', 'False', 'None'
        ]
        
        # 过滤掉不是变量名的内容
        variables = [var for var in potential_vars if var not in exclude_list and not var.startswith('ARG')]
        
        return list(set(variables))  # 去重
        

class FactorAgent:
    """因子挖掘Agent，集成到现有系统中"""
    
    def __init__(self, model_dir: str = 'factors'):
        """初始化因子Agent"""
        self.factor_module = FactorMiningModule(model_dir=model_dir)
        self.logger = setup_logger('factor_agent')
        self.factors_generated = False
    
    def generate_factors(self, price_data, n_factors=5, **kwargs):
        """生成因子"""
        try:
            # 提取和准备参数
            evolve_params = {
                'price_data': price_data,
                'n_factors': n_factors
            }
            
            # 添加其他可能的参数
            allowed_params = ['population_size', 'n_generations', 'future_return_periods', 'min_fitness']
            for param in allowed_params:
                if param in kwargs:
                    evolve_params[param] = kwargs[param]
            
            # 生成因子
            factors = self.factor_module.evolve_factors(**evolve_params)
            
            self.factors_generated = len(factors) > 0
            return factors
            
        except Exception as e:
            self.logger.error(f"生成因子时出错: {str(e)}")
            return []
    
    def load_factors(self):
        """加载已保存的因子"""
        try:
            factors = self.factor_module.load_factors()
            self.factors_generated = len(factors) > 0
            return factors
            
        except Exception as e:
            self.logger.error(f"加载因子时出错: {str(e)}")
            return []
    
    def get_factor_values(self, price_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """获取因子值"""
        try:
            # 如果没有生成因子，则尝试加载
            if not self.factors_generated:
                self.load_factors()
            
            # 如果仍然没有因子，返回空字典
            if not self.factors_generated:
                return {}
            
            # 计算因子值
            factor_values = self.factor_module.calculate_factor_values(price_data)
            return factor_values
            
        except Exception as e:
            self.logger.error(f"获取因子值时出错: {str(e)}")
            return {}
    
    def generate_signals(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """生成交易信号"""
        signals = {}
        
        try:
            # 获取因子值
            factor_values = self.get_factor_values(price_data)
            
            if not factor_values:
                return {'signal': 'neutral', 'confidence': 0.5}
            
            # 获取最新因子值
            latest_values = {name: values[-1] for name, values in factor_values.items()}
            
            # 记录所有因子信号
            factor_signals = []
            
            # 分析每个因子的信号
            for factor_name, factor_value in latest_values.items():
                # 因子趋势（与前5日比较）
                factor_series = factor_values[factor_name]
                factor_trend = factor_value - np.mean(factor_series[-6:-1]) if len(factor_series) > 5 else 0
                
                # 因子强度（标准化）
                factor_std = np.std(factor_series[-20:]) if len(factor_series) > 20 else 1
                factor_strength = factor_value / (factor_std + 1e-8)
                
                # 确定信号
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
            
            # 计算综合信号
            bullish_count = sum(1 for s in factor_signals if s['signal'] == 'bullish')
            bearish_count = sum(1 for s in factor_signals if s['signal'] == 'bearish')
            
            # 加权平均置信度
            avg_confidence = sum(s['confidence'] for s in factor_signals) / len(factor_signals)
            
            # 确定最终信号
            if bullish_count > bearish_count:
                final_signal = 'bullish'
                final_confidence = min(0.5 + 0.1 * bullish_count, avg_confidence)
            elif bearish_count > bullish_count:
                final_signal = 'bearish'
                final_confidence = min(0.5 + 0.1 * bearish_count, avg_confidence)
            else:
                final_signal = 'neutral'
                final_confidence = 0.5
            
            # 返回信号
            signals = {
                'signal': final_signal,
                'confidence': final_confidence,
                'factor_signals': factor_signals,
                'reasoning': self._generate_reasoning(factor_signals, final_signal, final_confidence)
            }
            
        except Exception as e:
            self.logger.error(f"生成交易信号时出错: {str(e)}")
            signals = {'signal': 'neutral', 'confidence': 0.5}
        
        return signals
    
    def _generate_reasoning(self, factor_signals: List[Dict[str, Any]], 
                           final_signal: str, final_confidence: float) -> str:
        """生成信号的解释"""
        reasoning_parts = []
        
        # 统计信号
        bullish_count = sum(1 for s in factor_signals if s['signal'] == 'bullish')
        bearish_count = sum(1 for s in factor_signals if s['signal'] == 'bearish')
        neutral_count = sum(1 for s in factor_signals if s['signal'] == 'neutral')
        
        # 添加综合分析
        reasoning_parts.append(f"遗传编程因子信号: {len(factor_signals)}个因子中，{bullish_count}个看多, {bearish_count}个看空, {neutral_count}个中性")
        
        # 添加最强因子分析
        strongest_bullish = [s for s in factor_signals if s['signal'] == 'bullish']
        strongest_bearish = [s for s in factor_signals if s['signal'] == 'bearish']
        
        if strongest_bullish:
            top_bullish = max(strongest_bullish, key=lambda s: s['confidence'])
            reasoning_parts.append(f"最强看多因子: {top_bullish['name']}, 因子值: {top_bullish['value']:.2f}, 置信度: {top_bullish['confidence']:.2f}")
        
        if strongest_bearish:
            top_bearish = max(strongest_bearish, key=lambda s: s['confidence'])
            reasoning_parts.append(f"最强看空因子: {top_bearish['name']}, 因子值: {top_bearish['value']:.2f}, 置信度: {top_bearish['confidence']:.2f}")
        
        # 添加最终结论
        reasoning_parts.append(f"综合遗传编程因子分析得出{final_signal}信号，置信度: {final_confidence:.2f}")
        
        return "; ".join(reasoning_parts)