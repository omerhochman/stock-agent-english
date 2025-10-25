# Single investment decision example
import uuid

from src.main import run_hedge_fund

# Generate unique run ID
run_id = str(uuid.uuid4())

# Build portfolio
portfolio = {"cash": 100000, "stock": 0}

# Execute decision
result = run_hedge_fund(
    run_id=run_id,
    ticker="600519",  # Kweichow Moutai
    start_date="2023-01-01",
    end_date="2023-12-31",
    portfolio=portfolio,
    show_reasoning=True,
    num_of_news=5,
)

# Print results
print(f"Investment decision result: {result}")
