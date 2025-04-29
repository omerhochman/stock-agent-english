# 训练脚本基本用法：

日期用`--start-date 2023-01-01 --end-date 2023-12-31`指定，如未指定，默认为一年前到现在

```python
# 使用深度学习模型训练和测试
python train.py --ticker 600054 --model dl

# 仅训练强化学习模型
python train.py --ticker 600054 --model rl --action train

# 仅测试之前训练好的因子模型
python train.py --ticker 600054 --model factor --action test

# 训练所有模型
python train.py --ticker 600054 --model all

# 指定自定义参数
python train.py --ticker 600054 --model dl --params '{"hidden_dim": 128, "epochs": 100}'
```
