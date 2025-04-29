import numpy as np
import pandas as pd
import random
import operator
from deap import algorithms, base, creator, tools, gp
import os
import logging
from typing import Dict, List, Any, Tuple, Callable
import json
from functools import partial
import warnings
from scipy import stats

# 设置警告级别
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 设置日志
logger = logging.getLogger('factor_mining')

# 设置随机种子以确保结果可重现
random.seed(42)
np.random.seed(42)


# 安全保护的特殊函数
def protected_div(left, right):
    """
    安全除法，防止除以0错误
    """
    if abs(right) < 1e-10:
        return 1.0
    return left / right

def protected_sqrt(x):
    """
    安全平方根，处理负数
    """
    if x <= 0:
        return 0.0
    return np.sqrt(x)

def protected_log(x):
    """
    安全对数，处理负数和0
    """
    if x <= 0:
        return 0.0
    return np.log(x)

def protected_inv(x):
    """
    安全倒数，处理0
    """
    if abs(x) < 1e-10:
        return 0.0
    return 1.0 / x

def rolling_mean(x, window=10):
    """
    滚动平均
    """
    return pd.Series(x).rolling(window=window).mean().fillna(0).values

def rolling_std(x, window=10):
    """
    滚动标准差
    """
    return pd.Series(x).rolling(window=window).std().fillna(0).values

def ts_delta(x, period=1):
    """
    时间序列差分
    """
    return pd.Series(x).diff(period).fillna(0).values

def ts_delay(x, period=1):
    """
    时间序列滞后
    """
    return pd.Series(x).shift(period).fillna(0).values

def ts_return(x, period=1):
    """
    时间序列收益率
    """
    return pd.Series(x).pct_change(period).fillna(0).values

def ts_rank(x, window=10):
    """
    时间序列排名
    """
    return pd.Series(x).rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]).fillna(0).values


class FactorMiningModule:
    """因子挖掘主类，使用遗传编程自动生成和筛选因子"""
    
    def __init__(self, model_dir: str = 'factors'):
        """
        初始化因子挖掘模块
        
        Args:
            model_dir: 因子保存目录
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # 存储最佳因子
        self.best_factors = []
        self.pset = None
        self.toolbox = None
        
        self.logger = logging.getLogger('factor_mining')
    
    def _setup_primitives(self, input_names: List[str]):
        """
        设置遗传编程的原语集
        
        Args:
            input_names: 输入特征名称列表
        """
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
        
        # 添加时间序列函数
        self.pset.addPrimitive(partial(rolling_mean, window=5), [float], float)
        self.pset.addPrimitive(partial(rolling_mean, window=10), [float], float)
        self.pset.addPrimitive(partial(rolling_mean, window=20), [float], float)
        self.pset.addPrimitive(partial(rolling_std, window=5), [float], float)
        self.pset.addPrimitive(partial(rolling_std, window=10), [float], float)
        self.pset.addPrimitive(partial(ts_delta, period=1), [float], float)
        self.pset.addPrimitive(partial(ts_delta, period=5), [float], float)
        self.pset.addPrimitive(partial(ts_delay, period=1), [float], float)
        self.pset.addPrimitive(partial(ts_delay, period=5), [float], float)
        self.pset.addPrimitive(partial(ts_return, period=1), [float], float)
        self.pset.addPrimitive(partial(ts_return, period=5), [float], float)
        self.pset.addPrimitive(partial(ts_rank, window=10), [float], float)
        
        # 添加常数
        self.pset.addEphemeralConstant("rand", lambda: random.uniform(-1, 1), float)
    
    def _evaluate_factor(self, individual, X: np.ndarray, y: np.ndarray, 
                        eval_func: Callable, eval_window: int = 20) -> Tuple[float,]:
        """
        评估因子质量
        
        Args:
            individual: 个体（因子表达式）
            X: 输入特征矩阵
            y: 目标值（未来收益率）
            eval_func: 评估函数（例如 IC / 回测收益等）
            eval_window: 评估窗口
            
        Returns:
            适应度值（越高越好）
        """
        # 将個体转换为函数
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
            
            # 计算评估指标
            score = eval_func(factor_values, y)
            
            # 计算因子复杂度惩罚
            complexity = len(str(individual))
            complexity_penalty = np.clip(complexity / 200, 0, 0.5)  # 将复杂度惩罚限制在0-0.5
            
            # 最终分数为评估指标减去复杂度惩罚
            final_score = score - complexity_penalty
            
            return (final_score,)
            
        except Exception as e:
            # 如果出错，返回最低分
            return (-np.inf,)
    
    def _calculate_ic(self, factor_values: np.ndarray, future_returns: np.ndarray, 
                     window: int = 20) -> float:
        """
        计算因子的IC值（信息系数）
        
        Args:
            factor_values: 因子值
            future_returns: 未来收益率
            window: 滚动窗口大小
            
        Returns:
            平均IC值
        """
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
        """
        计算因子分组回报差异
        
        Args:
            factor_values: 因子值
            future_returns: 未来收益率
            quantiles: 分组数量
            
        Returns:
            最高分位组与最低分位组回报差异
        """
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
        """
        计算基于因子的多空组合收益
        
        Args:
            factor_values: 因子值
            future_returns: 未来收益率
            top_percentile: 纳入多空组合的百分比
            
        Returns:
            多空组合收益
        """
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
        long_short_return = long_return - short_return
        
        return long_short_return
    
    def _factor_fitness(self, factor_values: np.ndarray, future_returns: np.ndarray) -> float:
        """
        综合计算因子质量
        
        Args:
            factor_values: 因子值
            future_returns: 未来收益率
            
        Returns:
            综合适应度分数
        """
        # 计算多个指标
        ic = self._calculate_ic(factor_values, future_returns)
        return_spread = self._calculate_return_spread(factor_values, future_returns)
        portfolio_return = self._calculate_portfolio_return(factor_values, future_returns)
        
        # 计算因子自相关性（稳定性）
        factor_series = pd.Series(factor_values)
        factor_autocorr = factor_series.autocorr(lag=1)
        if np.isnan(factor_autocorr):
            factor_autocorr = 0
        
        # 计算因子单调性（按分组测试）
        monotonicity = 0.0
        try:
            df = pd.DataFrame({'factor': factor_values, 'returns': future_returns})
            df['quantile'] = pd.qcut(df['factor'], q=5, labels=False, duplicates='drop')
            group_returns = df.groupby('quantile')['returns'].mean()
            # 使用秩相关系数判断单调性
            if len(group_returns) >= 3:
                monotonicity = stats.spearmanr(group_returns.index, group_returns.values)[0]
                if np.isnan(monotonicity):
                    monotonicity = 0.0
        except:
            pass
        
        # 综合评分（各项指标加权）
        fitness = (
            0.4 * abs(ic) +                  # IC值重要性
            0.3 * abs(return_spread) +       # 分组收益差异
            0.2 * abs(portfolio_return) +    # 多空组合收益
            0.05 * abs(factor_autocorr) +    # 自相关性（稳定性）
            0.05 * abs(monotonicity)         # 单调性
        )
        
        return fitness
    
    def evolve_factors(self, price_data: pd.DataFrame, n_factors: int = 5, 
                      population_size: int = 100, n_generations: int = 50,
                      future_return_periods: int = 5, min_fitness: float = 0.05):
        """
        使用遗传编程进化因子
        
        Args:
            price_data: 股票价格数据DataFrame
            n_factors: 生成的因子数量
            population_size: 种群大小
            n_generations: 进化代数
            future_return_periods: 未来收益周期（天）
            min_fitness: 最小适应度要求
            
        Returns:
            生成的因子列表
        """
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
        
        # 注册个体创建器
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        
        # 注册种群创建器
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # 注册评估函数
        self.toolbox.register("evaluate", self._evaluate_factor, X=X, y=y, 
                             eval_func=self._factor_fitness)
        
        # 注册遗传操作
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)
        
        # 注册选择操作
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        
        # 注册编译器
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
            pop, logbook = algorithms.eaSimple(pop, self.toolbox, cxpb=0.5, mutpb=0.1,
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
        """
        准备建模数据
        
        Args:
            price_data: 价格数据
            future_return_periods: 未来收益周期
            
        Returns:
            X: 特征矩阵
            y: 目标值
            feature_names: 特征名称列表
        """
        # 复制数据，避免修改原始数据
        df = price_data.copy()
        
        # 确保有收盘价
        if 'close' not in df.columns:
            raise ValueError("价格数据必须包含'close'列")
        
        # 计算特征
        # 价格特征
        df['open_close'] = df['open'] / df['close'] - 1 if 'open' in df.columns else 0
        df['high_low'] = df['high'] / df['low'] - 1 if 'high' in df.columns and 'low' in df.columns else 0
        
        # 计算收益率
        df['returns_1d'] = df['close'].pct_change(1)
        df['returns_5d'] = df['close'].pct_change(5)
        df['returns_10d'] = df['close'].pct_change(10)
        df['returns_20d'] = df['close'].pct_change(20)
        
        # 计算成交量特征
        if 'volume' in df.columns:
            df['volume_ma5'] = df['volume'].rolling(window=5).mean()
            df['volume_ma20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma5']
            df['volume_change'] = df['volume'].pct_change(5)
        
        # 计算波动率
        df['volatility_5d'] = df['returns_1d'].rolling(window=5).std()
        df['volatility_20d'] = df['returns_1d'].rolling(window=20).std()
        
        # 计算技术指标
        # 移动平均
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma60'] = df['close'].rolling(window=60).mean()
        
        # 均线差离
        df['ma5_ma20'] = df['ma5'] / df['ma20'] - 1
        df['ma10_ma60'] = df['ma10'] / df['ma60'] - 1
        
        # 相对强弱指标（RSI）
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
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
        """
        保存生成的因子
        """
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
        """
        加载已保存的因子
        
        Returns:
            加载的因子列表
        """
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
        """
        计算最新数据的因子值
        
        Args:
            price_data: 价格数据
            
        Returns:
            因子值字典
        """
        if not self.best_factors:
            self.load_factors()
            
        if not self.best_factors:
            return {}
        
        # 准备特征数据
        X, _, feature_names = self._prepare_data(price_data)
        
        # 确保已设置原语集和工具箱
        if self.pset is None:
            self._setup_primitives(feature_names)
            
        if self.toolbox is None:
            # 简单初始化工具箱
            self.toolbox = base.Toolbox()
            self.toolbox.register("compile", gp.compile, pset=self.pset)
        
        # 计算因子值
        factor_values = {}
        
        for factor_info in self.best_factors:
            try:
                # 解析表达式
                expr = eval(factor_info['expression'])
                # 编译为函数
                func = self.toolbox.compile(expr=expr)
                
                # 计算因子值
                values = np.zeros(X.shape[0])
                
                for i in range(X.shape[0]):
                    sample = [X[i, j] for j in range(X.shape[1])]
                    try:
                        values[i] = func(*sample)
                    except:
                        values[i] = np.nan
                
                # 替换无效值
                values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
                
                # 标准化因子值
                values = (values - np.mean(values)) / (np.std(values) + 1e-8)
                
                # 存储因子值
                factor_values[factor_info['name']] = values
                
            except Exception as e:
                self.logger.error(f"计算因子 {factor_info['name']} 值时出错: {str(e)}")
        
        return factor_values


class FactorAgent:
    """因子挖掘Agent，集成到现有系统中"""
    
    def __init__(self, model_dir: str = 'factors'):
        """
        初始化因子Agent
        
        Args:
            model_dir: 因子保存目录
        """
        self.factor_module = FactorMiningModule(model_dir=model_dir)
        self.logger = logging.getLogger('factor_agent')
        self.factors_generated = False
    
    def generate_factors(self, price_data: pd.DataFrame, n_factors: int = 5):
        """
        生成因子
        
        Args:
            price_data: 价格数据
            n_factors: 生成的因子数量
            
        Returns:
            生成的因子列表
        """
        try:
            # 生成因子
            factors = self.factor_module.evolve_factors(
                price_data=price_data,
                n_factors=n_factors,
                population_size=100,
                n_generations=20  # 减少代数以加快训练
            )
            
            self.factors_generated = len(factors) > 0
            return factors
            
        except Exception as e:
            self.logger.error(f"生成因子时出错: {str(e)}")
            return []
    
    def load_factors(self):
        """
        加载已保存的因子
        
        Returns:
            加载的因子列表
        """
        try:
            factors = self.factor_module.load_factors()
            self.factors_generated = len(factors) > 0
            return factors
            
        except Exception as e:
            self.logger.error(f"加载因子时出错: {str(e)}")
            return []
    
    def get_factor_values(self, price_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        获取因子值
        
        Args:
            price_data: 价格数据
            
        Returns:
            因子值字典
        """
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
        """
        生成交易信号
        
        Args:
            price_data: 价格数据
            
        Returns:
            包含交易信号的字典
        """
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
            
            # 最近的收盘价
            latest_close = price_data['close'].iloc[-1]
            
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
        """
        生成信号的解释
        
        Args:
            factor_signals: 因子信号列表
            final_signal: 最终信号
            final_confidence: 最终置信度
            
        Returns:
            信号解释
        """
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