# 单次投资决策示例
from src.main import run_hedge_fund
import uuid

# 生成唯一运行ID
run_id = str(uuid.uuid4())

# 构建投资组合
portfolio = {"cash": 100000, "stock": 0}

# 执行决策
result = run_hedge_fund(
    run_id=run_id,
    ticker="600519",  # 贵州茅台
    start_date="2023-01-01",
    end_date="2023-12-31",
    portfolio=portfolio,
    show_reasoning=True,
    num_of_news=5
)

# 打印结果
print(f"投资决策结果: {result}")