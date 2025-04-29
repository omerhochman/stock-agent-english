"""
深度学习/强化学习/遗传编程模块，提供了一系列机器学习和人工智能算法，用于量化交易、因子挖掘和市场预测。

主要功能
-------
1. 深度学习模型
   - LSTM模型用于时间序列预测
   - 随机森林分类器用于股票筛选

2. 强化学习模型
   - PPO算法用于交易策略优化
   - 自定义交易环境

3. 遗传编程
   - 自动化因子挖掘
   - 投资决策支持

使用示例
-------

1. 使用深度学习预测股票价格

```python
from model.dl import MLAgent, DeepLearningModule
import pandas as pd

# 准备股票价格数据
price_data = pd.DataFrame({
    'open': [100, 101, 102, 103, 104],
    'high': [105, 106, 107, 108, 109],
    'low': [95, 96, 97, 98, 99],
    'close': [102, 103, 104, 105, 106],
    'volume': [1000, 1100, 1200, 1300, 1400]
}, index=pd.date_range('2023-01-01', periods=5))

# 创建MLAgent实例
# 参数:
#   model_dir: 模型保存目录，默认为'model'
# 返回值:
#   MLAgent对象
ml_agent = MLAgent(model_dir='my_model')

# 训练模型
# 参数:
#   price_data: 价格数据DataFrame，需要包含'open', 'high', 'low', 'close', 'volume'列
#   technical_indicators: 可选的额外技术指标字典，键为指标名称，值为pd.Series
# 返回值:
#   无直接返回值，训练结果保存在MLAgent实例中
ml_agent.train_model(price_data)

# 生成预测信号
# 参数:
#   price_data: 价格数据DataFrame
#   technical_indicators: 可选的额外技术指标字典
# 返回值:
#   Dict[str, Any]: 包含以下键值对的字典:
#     - signal: 交易信号，'bullish'(看多), 'bearish'(看空), 'neutral'(中性)
#     - confidence: 置信度，0到1之间的浮点数
#     - lstm_predictions: LSTM模型预测，包含'future_prices'和'expected_returns'
#     - rf_prediction: 随机森林模型预测，包含'prediction'和'probability'
#     - reasoning: 预测理由
signals = ml_agent.generate_signals(price_data)

print(f"交易信号: {signals['signal']}")
print(f"置信度: {signals['confidence']:.2f}")
print(f"预测的未来价格: {signals.get('lstm_predictions', {}).get('future_prices', [])}")
print(f"决策依据: {signals['reasoning']}")
```

2. 使用强化学习训练交易策略

```python
from model.rl import RLTradingAgent
import pandas as pd
import numpy as np

# 准备股票价格数据（需要更多历史数据用于RL训练）
# 生成模拟数据
dates = pd.date_range('2022-01-01', periods=500)
prices = np.random.randn(500).cumsum() + 100  # 随机游走价格
price_data = pd.DataFrame({
    'open': prices * 0.99,
    'high': prices * 1.02,
    'low': prices * 0.98,
    'close': prices,
    'volume': np.random.randint(1000, 10000, 500)
}, index=dates)

# 创建RLTradingAgent实例
# 参数:
#   model_dir: 模型保存目录，默认为'model'
# 返回值:
#   RLTradingAgent对象
rl_agent = RLTradingAgent(model_dir='rl_model')

# 训练模型
# 参数:
#   price_data: 价格数据DataFrame
#   tech_indicators: 可选的技术指标字典
# 返回值:
#   Dict: 包含训练历史的字典，包括收益、损失等指标
training_history = rl_agent.train(price_data)

# 生成交易信号
# 参数:
#   price_data: 价格数据DataFrame
#   tech_indicators: 可选的技术指标字典
# 返回值:
#   Dict[str, Any]: 包含以下键值对的字典:
#     - signal: 交易信号，'bullish', 'bearish', 'neutral'
#     - confidence: 置信度，0到1之间的浮点数
#     - action_probabilities: 各动作的概率
#     - reasoning: 预测理由
signals = rl_agent.generate_signals(price_data)

print(f"RL交易信号: {signals['signal']}")
print(f"置信度: {signals['confidence']:.2f}")
print(f"行为概率: {signals.get('action_probabilities', {})}")
print(f"决策依据: {signals['reasoning']}")
```

3. 使用遗传编程挖掘交易因子

```python
from model.deap_factors import FactorAgent
import pandas as pd
import numpy as np

# 准备股票价格数据
dates = pd.date_range('2022-01-01', periods=500)
prices = np.random.randn(500).cumsum() + 100
price_data = pd.DataFrame({
    'open': prices * 0.99,
    'high': prices * 1.02,
    'low': prices * 0.98,
    'close': prices,
    'volume': np.random.randint(1000, 10000, 500)
}, index=dates)

# 创建FactorAgent实例
# 参数:
#   model_dir: 因子保存目录，默认为'factors'
# 返回值:
#   FactorAgent对象
factor_agent = FactorAgent(model_dir='my_factors')

# 生成因子
# 参数:
#   price_data: 价格数据DataFrame
#   n_factors: 生成的因子数量，默认为5
# 返回值:
#   List[Dict]: 生成的因子列表，每个因子包含名称、表达式、适应度等信息
factors = factor_agent.generate_factors(price_data, n_factors=3)

# 查看生成的因子
for factor in factors:
    print(f"因子名称: {factor['name']}")
    print(f"因子表达式: {factor['expression']}")
    print(f"因子适应度: {factor['fitness']:.4f}")
    print(f"信息系数(IC): {factor['ic']:.4f}")
    print("-------------------")

# 使用因子生成交易信号
# 参数:
#   price_data: 价格数据DataFrame
# 返回值:
#   Dict[str, Any]: 包含以下键值对的字典:
#     - signal: 交易信号，'bullish', 'bearish', 'neutral'
#     - confidence: 置信度，0到1之间的浮点数
#     - factor_signals: 各因子的信号
#     - reasoning: 预测理由
signals = factor_agent.generate_signals(price_data)

print(f"因子综合信号: {signals['signal']}")
print(f"置信度: {signals['confidence']:.2f}")
print(f"决策依据: {signals['reasoning']}")
```

4. 组合多种模型的交易信号

```python
from model.dl import MLAgent
from model.rl import RLTradingAgent
from model.deap_factors import FactorAgent
import pandas as pd
import numpy as np

# 加载模型
ml_agent = MLAgent()
ml_agent.load_model()

rl_agent = RLTradingAgent()
rl_agent.load_model()

factor_agent = FactorAgent()
factor_agent.load_factors()

# 获取最新价格数据
price_data = pd.DataFrame(...)  # 假设这里有实时数据获取逻辑

# 获取各个模型的信号
ml_signals = ml_agent.generate_signals(price_data)
rl_signals = rl_agent.generate_signals(price_data)
factor_signals = factor_agent.generate_signals(price_data)

# 组合信号
# 简单投票法
signals = [
    ml_signals['signal'],
    rl_signals['signal'],
    factor_signals['signal']
]

# 计算每种信号的数量
bullish_count = signals.count('bullish')
bearish_count = signals.count('bearish')
neutral_count = signals.count('neutral')

# 综合信号
if bullish_count > bearish_count and bullish_count > neutral_count:
    final_signal = 'bullish'
    # 计算综合置信度（加权平均）
    confidence = (
        ml_signals['confidence'] * (1 if ml_signals['signal'] == 'bullish' else 0) +
        rl_signals['confidence'] * (1 if rl_signals['signal'] == 'bullish' else 0) +
        factor_signals['confidence'] * (1 if factor_signals['signal'] == 'bullish' else 0)
    ) / bullish_count
elif bearish_count > bullish_count and bearish_count > neutral_count:
    final_signal = 'bearish'
    confidence = (
        ml_signals['confidence'] * (1 if ml_signals['signal'] == 'bearish' else 0) +
        rl_signals['confidence'] * (1 if rl_signals['signal'] == 'bearish' else 0) +
        factor_signals['confidence'] * (1 if factor_signals['signal'] == 'bearish' else 0)
    ) / bearish_count
else:
    final_signal = 'neutral'
    confidence = 0.5

print(f"最终交易信号: {final_signal}")
print(f"置信度: {confidence:.2f}")
print("各模型信号:")
print(f"  机器学习模型: {ml_signals['signal']} ({ml_signals['confidence']:.2f})")
print(f"  强化学习模型: {rl_signals['signal']} ({rl_signals['confidence']:.2f})")
print(f"  因子模型: {factor_signals['signal']} ({factor_signals['confidence']:.2f})")
```
"""

from model.dl import MLAgent, DeepLearningModule, preprocess_stock_data
from model.rl import RLTradingAgent, RLTrader, StockTradingEnv
from model.deap_factors import FactorAgent, FactorMiningModule

# 公开API
__all__ = [
    # 深度学习模块
    'MLAgent',
    'DeepLearningModule',
    'preprocess_stock_data',
    
    # 强化学习模块
    'RLTradingAgent',
    'RLTrader',
    'StockTradingEnv',
    
    # 遗传编程模块
    'FactorAgent',
    'FactorMiningModule'
]