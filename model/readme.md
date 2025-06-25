# Model 模块

## 模块结构

```
model/
├── __init__.py              # 模块初始化，公开API
├── dl.py                    # 深度学习模型（LSTM + 随机森林）
├── rl.py                    # 强化学习模型（PPO算法）
├── deap_factors.py          # 遗传编程因子挖掘
├── evaluate.py              # 模型评估框架
├── split_evaluate.py        # 数据划分与评估工具
└── train/
    ├── train.py            # 统一训练接口
    └── readme.md           # 训练模块详细说明
```

## 核心模型介绍

### 1. 深度学习模型 (dl.py)

深度学习模型结合 LSTM 神经网络和随机森林分类器，从时间序列和横截面两个维度进行预测。

**核心组件**：

- **LSTMModel**: 基于 PyTorch 的 LSTM 网络，用于价格序列预测
- **RandomForestClassifier**: 基于 sklearn 的分类器，用于涨跌方向预测
- **MLAgent**: 统一的模型管理接口

**主要功能**：

```python
from model.dl import MLAgent

# 初始化并训练模型
ml_agent = MLAgent(model_dir='my_model')
ml_agent.train_models(price_data, epochs=100, hidden_dim=128)

# 生成交易信号
signals = ml_agent.generate_signals(price_data)
print(f"信号: {signals['signal']}, 置信度: {signals['confidence']}")
```

**技术特点**：

- 支持多特征输入（价格、技术指标）
- 自动数据预处理和归一化
- 多步预测和概率输出
- 模型持久化和加载

### 2. 强化学习模型 (rl.py)

强化学习模型使用 PPO（Proximal Policy Optimization）算法训练智能交易代理。

**核心组件**：

- **StockTradingEnv**: OpenAI Gym 兼容的交易环境
- **ActorCritic**: PPO 的策略-价值网络
- **PPOAgent**: PPO 算法实现
- **RLTradingAgent**: 统一的强化学习接口

**主要功能**：

```python
from model.rl import RLTradingAgent

# 训练强化学习模型
rl_agent = RLTradingAgent(model_dir='rl_model')
training_history = rl_agent.train(price_data, n_episodes=500)

# 生成交易信号
signals = rl_agent.generate_signals(price_data)
print(f"RL信号: {signals['signal']}, 行为概率: {signals['action_probabilities']}")
```

**技术特点**：

- 基于真实市场数据的模拟交易环境
- 支持买入、卖出、持有三种动作
- 考虑交易成本和滑点
- GPU 加速训练

### 3. 遗传编程因子模型 (deap_factors.py)

遗传编程模型通过进化算法自动发现和优化交易因子。

**核心组件**：

- **FactorMiningModule**: 遗传编程核心引擎
- **FactorAgent**: 因子挖掘和信号生成接口
- **进化算子**: 交叉、变异、选择等遗传操作

**主要功能**：

```python
from model.deap_factors import FactorAgent

# 生成交易因子
factor_agent = FactorAgent(model_dir='my_factors')
factors = factor_agent.generate_factors(price_data, n_factors=5)

# 查看生成的因子
for factor in factors:
    print(f"因子: {factor['name']}, 表达式: {factor['expression']}")

# 生成交易信号
signals = factor_agent.generate_signals(price_data)
```

**技术特点**：

- 自动因子发现和表达式生成
- 多目标优化（IC、收益、夏普比等）
- 因子复杂度控制
- 安全的数学运算保护

## 模型评估框架 (evaluate.py)

提供完整的模型性能评估和可视化功能。

**核心功能**：

```python
from model.evaluate import ModelEvaluator

evaluator = ModelEvaluator(output_dir='evaluation')

# 回归模型评估
metrics = evaluator.evaluate_regression_model(y_true, y_pred, 'LSTM', 'test')

# 分类模型评估
metrics = evaluator.evaluate_classification_model(y_true, y_pred, 'RF', 'test')

# 预测结果可视化
evaluator.visualize_predictions(y_true, y_pred, date_index, 'model', 'test')
```

**评估指标**：

- **回归**: MSE, RMSE, MAE, R²
- **分类**: 准确率, 精确率, 召回率, F1 分数
- **可视化**: 预测对比图, 误差分析, 混淆矩阵

## 统一训练接口

### 命令行训练工具

```bash
# 训练深度学习模型
python -m model.train.train --ticker 600519 --model dl

# 训练所有模型
python -m model.train.train --ticker 600519 --model all

# 数据划分评估
python -m model.train.train --ticker 600519 --model dl --action evaluate

# 自定义参数训练
python -m model.train.train --ticker 600519 --model dl --params '{"epochs": 200, "hidden_dim": 256}'
```

### 编程接口

```python
# 导入所有核心组件
from model import MLAgent, RLTradingAgent, FactorAgent

# 分别训练三种模型
ml_agent = MLAgent()
ml_agent.train_models(price_data)

rl_agent = RLTradingAgent()
rl_agent.train(price_data)

factor_agent = FactorAgent()
factor_agent.generate_factors(price_data)

# 获取综合信号
ml_signals = ml_agent.generate_signals(price_data)
rl_signals = rl_agent.generate_signals(price_data)
factor_signals = factor_agent.generate_signals(price_data)
```

## 信号生成与融合

### 信号标准格式

所有模型生成的信号都遵循统一格式：

```python
{
    'signal': 'bullish',        # 信号类型: bullish/bearish/neutral
    'confidence': 0.78,         # 置信度: 0-1
    'reasoning': '...',         # 决策理由
    'model_specific': {...}     # 模型特定信息
}
```

### 多模型信号融合

```python
# 简单投票法
def combine_signals(ml_signals, rl_signals, factor_signals):
    signals = [ml_signals['signal'], rl_signals['signal'], factor_signals['signal']]

    bullish_count = signals.count('bullish')
    bearish_count = signals.count('bearish')

    if bullish_count > bearish_count:
        return 'bullish'
    elif bearish_count > bullish_count:
        return 'bearish'
    else:
        return 'neutral'

# 加权平均法
def weighted_combine(signals_list, weights):
    weighted_sum = sum(w * signal_to_score(s) for w, s in zip(weights, signals_list))
    return score_to_signal(weighted_sum)
```

## 技术指标与特征工程

### 自动技术指标计算

模块自动计算常用技术指标：

```python
# 价格指标
- 移动平均线 (MA5, MA10, MA20, MA60)
- 价格变化率 (1日、5日、10日、20日)
- 波动率 (5日、10日、20日)

# 技术指标
- RSI (相对强弱指标)
- MACD (移动平均收敛发散)
- 布林带
- 成交量指标

# 因子工程
- 滞后特征
- 滚动统计
- 价量关系
- 市场微观结构
```

### 数据预处理流程

```python
def preprocess_stock_data(price_df, technical_indicators=None):
    # 1. 数据清洗和格式化
    df = price_df.copy()
    df['returns'] = df['close'].pct_change()

    # 2. 技术指标计算
    df['ma5'] = df['close'].rolling(5).mean()
    df['rsi'] = calculate_rsi(df['close'])
    df['macd'] = calculate_macd(df['close'])

    # 3. 滞后特征
    for lag in [1, 2, 3, 5, 10]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)

    # 4. 滚动统计
    for window in [5, 10, 20]:
        df[f'volatility_{window}d'] = df['returns'].rolling(window).std()

    return df.dropna()
```

## 使用注意事项

### 数据要求

1. **最小数据量**: 建议至少 252 个交易日（1 年）的历史数据
2. **数据质量**: 确保价格数据完整，无异常值
3. **特征对齐**: 确保所有特征在时间上对齐

### 训练建议

1. **深度学习**:

   - 使用较长的序列长度（10-20 天）
   - 适当的隐藏层维度（64-256）
   - 早停机制防止过拟合

2. **强化学习**:

   - 足够的训练 episode（500-2000）
   - 合理的奖励函数设计
   - 适当的探索-利用平衡

3. **遗传编程**:
   - 控制因子复杂度
   - 设置合适的适应度阈值
   - 避免过度拟合
