# 训练脚本基本用法：

日期用`--start-date 2023-01-01 --end-date 2023-12-31`指定，如未指定，默认为 2 年前到昨天

```python
# 使用深度学习模型训练和测试
python train.py --ticker 600054 --model dl

# 仅训练强化学习模型
python train.py --ticker 600054 --model rl --action train

# 仅测试之前训练好的因子模型
python train.py --ticker 600054 --model factor --action test

# 训练所有模型
python train.py --ticker 600054 --model all

# 指定自定义参数（注意windows需要加\来转义双引号）
python train.py --ticker 600054 --model dl --params '{"hidden_dim": 128, "epochs": 100}'
```

# 数据划分与模型评估功能

新增功能支持按照指定比例划分训练集、验证集和测试集，并对模型性能进行评估和可视化。

```python
# 使用默认比例(70%/20%/10%)划分数据，并评估深度学习模型
python train.py --ticker 600054 --model dl --action evaluate

# 自定义数据划分比例
python train.py --ticker 600054 --model dl --action evaluate --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1

# 评估所有模型（深度学习、强化学习和因子模型）
python train.py --ticker 600054 --model all --action evaluate

# 指定评估结果保存目录
python train.py --ticker 600054 --model dl --action evaluate --eval-dir ./evaluation_results

# 打乱数据（默认按时间顺序划分）
python train.py --ticker 600054 --model dl --action evaluate --shuffle
```

## 评估结果

评估模式会生成以下内容：

1. 训练集与测试集数据分布比较图
2. 模型性能指标（MSE、RMSE、MAE、R² 等）
3. 预测结果可视化图表
4. 模型预测误差分析
5. 特征重要性分析（适用于随机森林模型）
6. 未来价格预测展示

所有评估结果默认保存在 `models/evaluation` 目录下。

## 参数说明：

- `--action evaluate`: 使用数据划分和评估模式
- `--train-ratio`: 训练集比例，默认 0.7 (70%)
- `--val-ratio`: 验证集比例，默认 0.2 (20%)
- `--test-ratio`: 测试集比例，默认 0.1 (10%)
- `--shuffle`: 是否打乱数据，默认不打乱（按时间顺序划分）
- `--eval-dir`: 评估结果保存目录，默认为 "models/evaluation"
