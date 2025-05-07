# A股投资系统 - 模型训练模块

本模块提供了三种不同类型的AI模型训练和测试功能，包括深度学习模型(dl)、强化学习模型(rl)和遗传编程因子模型(factor)。这些模型可以协同工作，从不同角度分析股票市场，提供综合的交易信号和决策支持。

## 快速开始

```bash
# 基本用法
python -m model.train.train --ticker 600054 --model dl

# 训练所有模型
python -m model.train.train --ticker 600054 --model all

# 指定日期范围
python -m model.train.train --ticker 600054 --model dl --start-date 2023-01-01 --end-date 2023-12-31

# 评估模型性能
python -m model.train.train --ticker 600054 --model dl --action evaluate
```

## 基本用法

日期范围使用`--start-date 2023-01-01 --end-date 2023-12-31`指定，如未指定，默认为2年前到昨天。

```bash
# 使用深度学习模型训练和测试
python -m model.train.train --ticker 600054 --model dl

# 仅训练强化学习模型
python -m model.train.train --ticker 600054 --model rl --action train

# 仅测试之前训练好的因子模型
python -m model.train.train --ticker 600054 --model factor --action test

# 训练所有模型
python -m model.train.train --ticker 600054 --model all

# 指定自定义参数（注意windows需要加\来转义双引号）
python -m model.train.train --ticker 600054 --model dl --params '{"hidden_dim": 128, "epochs": 100}'
```

## 数据划分与模型评估功能

系统支持按照指定比例划分训练集、验证集和测试集，并对模型性能进行评估和可视化：

```bash
# 使用默认比例(70%/20%/10%)划分数据，并评估深度学习模型
python -m model.train.train --ticker 600054 --model dl --action evaluate

# 自定义数据划分比例
python -m model.train.train --ticker 600054 --model dl --action evaluate --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1

# 评估所有模型（深度学习、强化学习和因子模型）
python -m model.train.train --ticker 600054 --model all --action evaluate

# 指定评估结果保存目录
python -m model.train.train --ticker 600054 --model dl --action evaluate --eval-dir ./evaluation_results

# 打乱数据（默认按时间顺序划分）
python -m model.train.train --ticker 600054 --model dl --action evaluate --shuffle
```

## 模型类型详解

### 1. 深度学习模型 (dl)

深度学习模型结合LSTM神经网络和随机森林分类器进行价格预测和趋势判断。

**主要特点**:
- LSTM用于预测未来10天的价格走势
- 随机森林分类器用于预测涨跌方向
- 综合两个模型的结果生成最终信号

**默认参数**:
```
seq_length: 15           # 序列长度
forecast_days: 10        # 预测天数
hidden_dim: 128          # 隐藏层维度
num_layers: 3            # LSTM层数
epochs: 100              # 训练轮数
batch_size: 32           # 批次大小
learning_rate: 0.0005    # 学习率
```

**使用的特征**:
- 收盘价 (close)
- 5日移动平均线 (ma5)
- 10日移动平均线 (ma10)
- 20日移动平均线 (ma20)
- 相对强弱指标 (rsi)
- 移动平均收敛发散指标 (macd)

### 2. 强化学习模型 (rl)

强化学习模型使用PPO (Proximal Policy Optimization) 算法训练智能体，通过在模拟环境中交易来学习最优策略。

**主要特点**:
- 基于PPO算法的交易策略
- 智能体可以选择买入、卖出或持有动作
- 模型通过最大化累积收益来优化策略

**默认参数**:
```
n_episodes: 100              # 训练轮数
batch_size: 32               # 批次大小
reward_scaling: 1.0          # 奖励缩放
initial_balance: 100000      # 初始资金
transaction_fee_percent: 0.001 # 交易费率
window_size: 10              # 观察窗口大小
max_steps: 236               # 最大步数
```

### 3. 遗传编程因子模型 (factor)

遗传编程因子模型通过进化算法自动发现有预测能力的交易因子，并使用这些因子生成交易信号。

**主要特点**:
- 自动发现预测性较强的交易因子
- 多因子综合评分
- 适应市场变化的进化能力

**默认参数**:
```
n_factors: 3                 # 生成因子数量
population_size: 50          # 种群大小
n_generations: 20            # 迭代次数
future_return_periods: 5     # 未来收益期
min_fitness: 0.03            # 最小适应度
```

## 命令行参数

### 基本参数
- `--ticker`: 股票代码 (必需)
- `--model`: 模型类型，可选值: dl, rl, factor, all
- `--action`: 操作类型，可选值: train, test, evaluate (默认执行train和test)
- `--start-date`: 开始日期 (默认为结束日期前2年)
- `--end-date`: 结束日期 (默认为昨天)

### 高级参数
- `--params`: 自定义模型参数 (JSON格式)
  
  示例: `--params '{"hidden_dim": 256, "epochs": 200}'`

### 评估参数
- `--train-ratio`: 训练集比例 (默认: 0.7)
- `--val-ratio`: 验证集比例 (默认: 0.2)
- `--test-ratio`: 测试集比例 (默认: 0.1)
- `--shuffle`: 是否打乱数据 (默认按时间顺序划分)
- `--eval-dir`: 评估结果保存目录 (默认: "models/evaluation")

## 输出信号含义

模型训练和测试完成后会生成交易信号:

1. **信号类型**:
   - `bullish`: 看多信号，预期市场将上涨
   - `bearish`: 看空信号，预期市场将下跌
   - `neutral`: 中性信号，市场方向不明

2. **置信度**:
   - 范围从0到1，值越高表示模型对预测越有信心
   - 例: `bearish, 置信度: 0.73` 表示73%的把握认为市场将下跌

## 详细决策理由解析

以深度学习模型输出为例：
```
决策理由: LSTM模型预测未来10天: 短期收益率(-0.31%), 中期收益率(-17.68%), 长期收益率(-18.54%). 预期正收益天数占比: 0.00%; 随机森林模型预测下跌概率: 55.86%, 上涨概率: 44.14%; LSTM技术分析结果: bearish, 置信度: 90.00%; 随机森林技术分析结果: bearish, 置信度: 55.86%; 综合分析产生看空信号，置信度: 72.93%. 请注意，即使在看空信号下，市场仍有可能反弹，建议适时止盈。
```

## 模型投票机制

当使用`--model all`时，系统会综合三种模型的信号:

1. 统计各模型信号 (看多、看空、中性)
2. 计算加权评分 (范围从-1到1)
3. 生成最终综合信号

## 实际运行效果示例

```
==================================================
综合结果:
==================================================
dl     模型: bearish  (置信度: 0.73)
rl     模型: bearish  (置信度: 1.00)
factor 模型: bearish  (置信度: 0.70)

模型投票结果:

看多: 0，看空: 3，中性: 0

加权评分: -1.00 (范围: -1到1)

综合信号: bearish
```

## 评估结果输出

评估模式 (`--action evaluate`) 会生成以下内容:

1. 训练集与测试集数据分布比较图
2. 模型性能指标 (MSE、RMSE、MAE、R² 等)
3. 预测结果可视化图表
4. 模型预测误差分析
5. 特征重要性分析 (适用于随机森林模型)
6. 未来价格预测展示

所有评估结果默认保存在 `models/evaluation` 目录下。

## 模型训练日志示例

系统会生成详细的训练日志，包括训练进度、模型性能和最终结果：

```
2025-05-07 19:33:18 - deep_learning - INFO - 开始训练LSTM模型...
2025-05-07 19:33:18 - deep_learning - INFO - LSTM训练使用特征列: ['close', 'ma5', 'ma10', 'ma20', 'rsi', 'macd']
2025-05-07 19:33:18 - deep_learning - INFO - 初始化LSTM模型，输入维度: 6, 隐藏层维度: 128, 层数: 3
2025-05-07 19:33:20 - deep_learning - INFO - Epoch 1/100, Training Loss: 0.1068, Validation Loss: 0.0213
...
2025-05-07 19:33:32 - deep_learning - INFO - LSTM模型训练完成
2025-05-07 19:33:32 - deep_learning - INFO - 标签分布 - 上涨: 40.17%, 下跌: 59.83%
```

```
2025-05-07 19:33:53 - factor_mining - INFO - 开始因子进化，种群大小: 100，迭代次数: 50
gen     nevals  avg             min     max             std
0       100     -0.376797       -0.5    0.210664        0.205371
...
50      56      0.15184         -0.5    0.236512        0.222616
2025-05-07 19:34:55 - factor_mining - INFO - 因子 GP_Factor_1 生成完成，适应度: 0.2365
```

## 注意事项

1. **数据量要求**: 强化学习模型需要足够的历史数据，建议至少使用一年以上的数据
2. **模型存储**: 训练好的模型会保存在 `models/` 目录下
3. **Windows路径**: Windows系统使用自定义参数时需要转义双引号，例如:
   ```
   python -m model.train.train --ticker 600054 --model dl --params "{\"hidden_dim\": 128}"
   ```

## 已知问题

在某些情况下，强化学习模型训练可能会出现以下警告:
```
警告: 数据量不足。window_size=20, max_steps=252, 数据长度=100
已调整 max_steps 为 75
```

这是因为数据长度不足以满足默认的window_size和max_steps参数，系统会自动调整参数继续训练。
