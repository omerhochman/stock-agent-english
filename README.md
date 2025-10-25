# A-Share Investment Agent System

Forked From：https://github.com/24mlight/A_Share_investment_Agent.git

Course Project Version：https://github.com/1517005260/AShareAgent

![System Architecture V2](assets/img/structure.svg)

## System Overview

This is an Agent-based A-share investment decision system that achieves full-process automation of data collection, analysis, decision-making, and risk management through collaborative work of multiple specialized agents. The system adopts a modular design where each agent is responsible for specific analysis tasks, and the Portfolio Manager ultimately synthesizes analysis results from all parties to make trading decisions.

This project also includes an embedded MCP analysis module, see [this directory](./src/mcp_api/) for details.

## System Components

The system consists of the following collaborative agents:

1. **Market Data Analyst** - Responsible for collecting and preprocessing market data
2. **Technical Analyst** - Analyzes technical indicators and generates trading signals
3. **Fundamentals Analyst** - Analyzes fundamental data and generates trading signals
4. **Sentiment Analyst** - Analyzes market sentiment and generates trading signals
5. **Valuation Analyst** - Calculates intrinsic value of stocks and generates trading signals
6. **AI Model Analyst** - Runs AI model predictions and generates trading signals
7. **Macro Analyst** - Analyzes macroeconomic environment and generates trading signals
8. **Researcher Bull** - Analyzes comprehensive research results from a bullish perspective
9. **Researcher Bear** - Analyzes comprehensive research results from a bearish perspective
10. **Debate Room** - Synthesizes bullish and bearish views to form balanced analysis
11. **Risk Manager** - Calculates risk indicators and sets position limits
12. **Portfolio Manager** - Makes final trading decisions and generates orders

For detailed agent descriptions, please see [src/agents/README.md](src/agents/README.md).

## Environment Setup

### Clone Repository

```bash
git clone https://github.com/1517005260/stock-agent.git
cd stock-agent
```

### Using Conda to Configure Environment

1. Create and activate Conda environment:

```bash
conda create -n stock python=3.10
conda activate stock
```

2. Install dependencies:

```bash
cd stock-agent/
pip install -r requirements.txt
pip install -e .
```

3. Set environment variables:

```bash
# Create .env file to store API keys
cp .env.example .env
```

**Directly modify .env file**

Open .env file and fill in your API key:

```
OPENAI_COMPATIBLE_API_KEY=your_openai_compatible_api_key
OPENAI_COMPATIBLE_BASE_URL=https://api.example.com/v1
OPENAI_COMPATIBLE_MODEL=your_model_name

TUSHARE_TOKEN=your_tushare_api_key
```

## Usage

### Running the System

Main program:

```bash
# Basic usage
python -m src.main --ticker 600054 --show-reasoning

# Multi-asset
python src/main.py --ticker 600519 --tickers "600519,000858,601398" --start-date 2023-01-01 --end-date 2023-12-31

# Specify date range
python -m src.main --ticker 600054 --start-date 2023-01-01 --end-date 2023-12-31 --show-reasoning

# Specify initial capital and news count
python -m src.main --ticker 600054 --initial-capital 200000 --num-of-news 10

# Show detailed summary report
python -m src.main --ticker 600054 --summary
```

### Backtesting System

#### Basic Backtesting Commands

```bash
# Basic backtesting
python -m src.backtester --ticker 600054

# Specify backtesting time range
python -m src.backtester --ticker 600054 --start-date 2022-01-01 --end-date 2022-12-31

# Custom initial capital
python -m src.backtester --ticker 600054 --initial-capital 500000

# Show detailed analysis process
python -m src.backtester --ticker 600054 --show-reasoning

# Generate summary report
python -m src.backtester --ticker 600054 --summary
```

#### Backtesting Parameter Description

- `--ticker`: Stock code (required)
- `--start-date`: Backtesting start date (optional, format YYYY-MM-DD, default is one year before end date)
- `--end-date`: Backtesting end date (optional, format YYYY-MM-DD, default is yesterday)
- `--initial-capital`: Initial capital (optional, default is 100,000)
- `--initial-position`: Initial position quantity (optional, default is 0)
- `--show-reasoning`: Show daily analysis reasoning process (optional)
- `--summary`: Show summary report after backtesting ends (optional)

#### Backtesting Output Description

The backtesting system will output the following information:

1. **Daily Trading Records**: Including date, price, trading action, quantity, cash balance, position value, etc.
2. **Performance Metrics**: Total return, annualized return, Sharpe ratio, maximum drawdown, volatility, etc.
3. **Risk Metrics**: VaR, CVaR, Sortino ratio, Calmar ratio, etc.
4. **Trading Statistics**: Number of trades, win rate, profit/loss ratio, etc.

### Backtesting Test Tool (test_backtest.py)

#### Overview

`test/test_backtest.py` is a comprehensive testing tool specifically designed for testing and validating backtesting framework functionality. It provides multiple testing modes that can compare AI Agent strategies with traditional benchmark strategies and generate detailed performance analysis reports.

#### Main Features

1. **Benchmark Strategy Testing**: Tests various traditional investment strategies

   - Buy & Hold
   - Momentum
   - Mean Reversion
   - Moving Average
   - Random Walk

2. **AI Agent Strategy Testing**: Tests the intelligent investment agent strategies of this project

3. **Performance Comparison Analysis**: Generates detailed strategy comparison reports and rankings

4. **Statistical Significance Testing**: Validates statistical significance differences between strategies

#### Usage Methods

##### Basic Commands

```bash
# Run complete test suite (default 3 months)
python test/test_backtest.py

# Specify stock code
python test/test_backtest.py --ticker 600519

# Quick test mode (3 months)
python test/test_backtest.py --quick --ticker 600054

# Medium test mode (8 months)
python test/test_backtest.py --medium --ticker 600054

# Full test mode (2 years)
python test/test_backtest.py --full --ticker 600054

# Custom time range
python test/test_backtest.py --start-date 2023-01-01 --end-date 2023-12-31 --ticker 600054
```

##### Specialized Testing

```bash
# Test only AI Agent strategies
python test/test_backtest.py --ai-only --ticker 600054

# Test only benchmark strategies
python test/test_backtest.py --baseline-only --ticker 600054

# Run comprehensive comparison analysis
python test/test_backtest.py --comparison --ticker 600054
```

#### Parameter Description

- `--quick`: Quick test mode (3 months time range)
- `--medium`: Medium test mode (8 months time range)
- `--full`: Full test mode (2 years time range)
- `--start-date`: Custom start date (YYYY-MM-DD format)
- `--end-date`: Custom end date (YYYY-MM-DD format)
- `--ticker`: Stock code (default is 000001)
- `--ai-only`: Test only AI Agent strategies
- `--baseline-only`: Test only benchmark strategies
- `--comparison`: Run comprehensive comparison test

#### Test Output

The testing tool will generate the following types of reports:

1. **Performance Comparison Table**:

   - Strategy rankings
   - Total return, annualized return
   - Sharpe ratio, Sortino ratio, Calmar ratio
   - Maximum drawdown, annualized volatility
   - Win rate, profit/loss ratio, number of trades
   - VaR risk indicators

2. **AI Agent Specialized Analysis**:

   - Ranking among all strategies
   - Comparison with average level
   - Comprehensive performance rating (Excellent ⭐⭐⭐ / Good ⭐⭐ / Average ⭐)

3. **Statistical Significance Testing**:
   - Pairwise comparison between strategies
   - Diebold-Mariano test
   - Sharpe ratio test
   - Statistical power analysis

#### Test Strategy Description

##### Benchmark Strategies

1. **Buy & Hold**: Buy and hold strategy, suitable for long-term investment
2. **Momentum**: Price momentum-based strategy, buy high sell low
3. **Mean Reversion**: Mean reversion strategy, buy low sell high
4. **Moving Average**: Moving average strategy, based on moving average crossover signals
5. **Random Walk**: Random trading strategy, used as benchmark control

##### AI Agent Strategies

Integrates comprehensive decision strategies from all intelligent analysts in this project, including:

- Technical analysis, fundamental analysis, sentiment analysis
- Valuation analysis, macro analysis, AI model prediction
- Bull-bear debate, risk management, portfolio optimization

#### Example Output

```
==========================================
Backtesting Test Results Summary
==========================================
Test Configuration:
  Time Range: 2023-01-01 to 2023-03-31
  Test Duration: 89 days
  Stock Code: 600054

📊 Strategy Performance Ranking:
Rank  Strategy Name              Return      Sharpe Ratio   Max Drawdown
1     AI_Agent                  15.23%      1.245         8.45%
2     Momentum                  12.67%      0.987         12.34%
3     Buy_Hold                  8.91%       0.756         15.67%
4     Mean_Reversion           6.45%       0.543         18.23%
5     Moving_Average           4.32%       0.321         22.11%
6     Random_Walk              -2.15%      -0.123        25.67%

🎯 AI Agent Performance Analysis:
  📊 Ranking: 1st place (out of 6 strategies)
  💰 Return: 15.23%
  📈 Sharpe Ratio: 1.245
  📉 Max Drawdown: 8.45%
  🔄 Number of Trades: 23

📊 Comparison with Average Level:
  Return Difference: +7.57%
  Sharpe Ratio Difference: +0.621
  Drawdown Difference: -6.89%
  Overall Rating: Excellent ⭐⭐⭐
```

#### Notes

1. **Test Time Selection**:

   - Quick mode is suitable for functionality verification
   - Medium mode is suitable for strategy optimization
   - Full mode is suitable for formal evaluation

2. **Stock Selection**:

   - Recommend selecting large-cap stocks with good liquidity for testing
   - Avoid selecting stocks that are suspended or have incomplete data

3. **Result Interpretation**:
   - Focus on risk-adjusted return metrics (such as Sharpe ratio)
   - Pay attention to risk control indicators such as maximum drawdown
   - Consider trading frequency and actual execution costs

Model Training and Evaluation:

```bash
# Train deep learning model
python -m model.train.train --ticker 600054 --model dl

# Train all models
python -m model.train.train --ticker 600054 --model all

# Model evaluation (split training, validation, test sets)
python -m model.train.train --ticker 600054 --model dl --action evaluate

# Custom data split ratios
python -m model.train.train --ticker 600054 --model dl --action evaluate --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```

Data Processing Tools:

```bash
# Data analysis and technical indicator calculation
python -m src.tools.data_analyzer --ticker 600054

# News acquisition test
python -m src.tools.test_news_crawler
```

### Parameter Description

- `--ticker`: Stock code (required)
- `--tickers`: Multiple stock codes, comma-separated (optional, for multi-asset analysis)
- `--show-reasoning`: Show analysis reasoning process (optional, default is false)
- `--summary`: Show summary report (optional, default is false)
- `--initial-capital`: Initial cash amount (optional, default is 100,000)
- `--initial-position`: Initial position quantity (optional, default is 0)
- `--num-of-news`: Number of news articles used for sentiment analysis (optional, default is 5)
- `--start-date`: Analysis start date (optional, format YYYY-MM-DD, default is one year before end date)
- `--end-date`: Analysis end date (optional, format YYYY-MM-DD, default is yesterday)

### Command Line Mode Output Description

The system will output the following information:

1. Fundamental analysis results
2. Valuation analysis results
3. Technical analysis results
4. Sentiment analysis results
5. Risk management assessment
6. Final trading decision

If the `--show-reasoning` parameter is used, it will also display the detailed analysis process of each agent.
If the `--summary` parameter is used, it will display a formatted summary report after the analysis is complete.

**Example Output:**

```
--- Finished Workflow Run ID: c94a353c-8d28-486e-b5e7-9e7f92a1b7c4 ---
2025-05-07 19:56:56 - structured_terminal - INFO -
════════════════════════════════════════════════════════════════════════════════
                               Stock Code 600054 Investment Analysis Report
════════════════════════════════════════════════════════════════════════════════
                         Analysis Period: 2023-01-01 to 2025-05-06

╔═══════════════════════════════════ 📈 Technical Analysis ═══════════════════════════════════╗
║ Signal: 📈 bullish
║ Confidence: 34%
║ ├─ signal: bullish
║ ├─ confidence: 0.3369
║ ├─ market_regime: mean_reverting
║ ├─ regime_confidence: 0.5000
║ ├─ strategy_weights:
║   ├─ trend: 0.2000
║   ├─ mean_reversion: 0.4500
║   ├─ momentum: 0.1500
║   ├─ volatility: 0.1500
║   └─ stat_arb: 0.0500
║ └─ strategy_signals:
║   ├─ trend_following:
║     ├─ signal: neutral
║     ├─ confidence: 0.5000
║     └─ metrics:
║       ├─ adx: 17.4486
║       └─ trend_strength: 0.1745
║   ├─ mean_reversion:
║     ├─ signal: neutral
║     ├─ confidence: 0.2400
║     └─ metrics:
║       ├─ z_score: -0.6314
║       ├─ price_vs_bb: 0.2563
║       ├─ rsi_14: 39.8467
║       ├─ rsi_28: 48.0707
║       ├─ avg_deviation: -0.0200
║       ├─ k_percent: 21.0145
║       ├─ d_percent: 17.7575
║       └─ signal_score: 0
║   ├─ momentum:
║     ├─ signal: neutral
║     ├─ confidence: 0.2000
║     └─ metrics:
║       ├─ momentum_1m: -0.0260
║       ├─ momentum_3m: 0.0782
║       ├─ momentum_6m: 0.0280
║       ├─ relative_strength: 0.0983
║       ├─ volume_trend: 0.8827
║       └─ divergence: -0.1343
║   ├─ volatility:
║     ├─ signal: bullish
║     ├─ confidence: 0.7000
║     └─ metrics:
║       ├─ historical_volatility: 0.4362
║       ├─ volatility_regime: 1.5622
║       ├─ volatility_z_score: 0.5622
║       ├─ atr_ratio: 0.0304
║       ├─ garch_vol_trend: -0.2795
║       ├─ garch_forecast_quality: 0.8000
║       └─ garch_results:
║         ├─ model_type: GARCH(1,1)
║         ├─ parameters:
║           ├─ omega: 0.0000
║           ├─ alpha: 0.1484
║           ├─ beta: 0.7570
║           └─ persistence: 0.9054
║         ├─ log_likelihood: 1424.2592
║         ├─ forecast:
║           ├─ 0.01934439715669238
║           ├─ 0.01947384175695497
║           ├─ 0.019590300231429235
║           ├─ 0.01969514513510902
║           ├─ 0.01978959022705738
║           ├─ 0.019874711562961323
║           ├─ 0.019951465249916377
║           ├─ 0.020020702495460313
║           ├─ 0.020083182443127238
║           └─ 0.02013958318202366
║         └─ forecast_annualized:
║           ├─ 0.307082784834438
║           ├─ 0.3091376541595679
║           ├─ 0.31098637512871713
║           ├─ 0.31265073637693247
║           ├─ 0.3141500057320084
║           ├─ 0.315501265048413
║           ├─ 0.3167196920557553
║           ├─ 0.31781879925478945
║           ├─ 0.31881063767551954
║           └─ 0.3197059716488009
║   └─ statistical_arbitrage:
║     ├─ signal: neutral
║     ├─ confidence: 0.5000
║     └─ metrics:
║       ├─ hurst_exponent: 0.00000
║       ├─ skewness: -0.8531
║       └─ kurtosis: 4.0486
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════ 📝 Fundamental Analysis ══════════════════════════════════╗
║ Signal: 📈 bullish
║ Confidence: 50%
║ ├─ signal: bullish
║ ├─ confidence: 50%
║ └─ reasoning:
║   ├─ profitability_signal:
║     ├─ signal: neutral
║     └─ details: ROE: 12.00%, Net Margin: 15.00%, Op Margin: 18.00%
║   ├─ growth_signal:
║     ├─ signal: bearish
║     └─ details: Revenue Growth: 10.00%, Earnings Growth: 8.00%
║   ├─ financial_health_signal:
║     ├─ signal: bullish
║     └─ details: Current Ratio: 1.50, D/E: 0.40
║   └─ price_ratios_signal:
║     ├─ signal: bullish
║     └─ details: P/E: 57.18, P/B: 1.80, P/S: 3.00
╚══════════════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════ 🔍 Sentiment Analysis ═══════════════════════════════════╗
║ Signal: 📈 bullish
║ Confidence: 90%
║ ├─ signal: bullish
║ ├─ confidence: 90%
║ └─ reasoning: Based on 5 recent news articles, sentiment score: 0.90
╚══════════════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════ 💰 Valuation Analysis ═══════════════════════════════════╗
║ Signal: 📈 bullish
║ Confidence: 62%
║ ├─ signal: bullish
║ ├─ confidence: 0.6250
║ ├─ valuation_gap: 9.5668
║ ├─ all_valuations:
║   ├─ Agent 1:
║       ├─ method: DCF
║       ├─ value: $156.95B
║       └─ weight: 0.3500
║   ├─ Agent 2:
║       ├─ method: Owner Earnings
║       ├─ value: $97.82B
║       └─ weight: 0.3500
║   ├─ Agent 3:
║       ├─ method: Relative Valuation
║       ├─ value: 18.3600
║       └─ weight: 0.1500
║   └─ Agent 4:
║       ├─ method: Residual Income
║       ├─ value: 0
║       └─ weight: 0.1500
║ ├─ reasoning:
║   ├─ dcf_analysis:
║     ├─ signal: bullish
║     ├─ details: Intrinsic Value: $156,954,655,682.63, Market Cap: $8,438,920,121.00, Difference: 1759.9%
║     └─ model_details:
║       ├─ stages: Multi-stage DCF
║       ├─ wacc: 5.0%
║       ├─ beta: 0.78
║       └─ terminal_growth: 3.0%
║   ├─ owner_earnings_analysis:
║     ├─ signal: bullish
║     ├─ details: Owner Earnings Value: $97,823,398,513.58, Market Cap: $8,438,920,121.00, Difference: 1059.2%
║     └─ model_details:
║       ├─ required_return: 5.0%
║       ├─ margin_of_safety: 25%
║       └─ growth_rate: 8.0%
║   ├─ relative_valuation:
║     ├─ signal: bearish
║     ├─ details: Relative Valuation: $18.36, Market Cap: $8,438,920,121.00, Difference: -100.0%
║     └─ model_details:
║       ├─ pe_ratio: 57.18 (Industry Average Adjusted: 15.30)
║       ├─ pb_ratio: 1.80 (Industry Average: 1.50)
║       └─ growth_premium: 0.3
║   ├─ residual_income_valuation:
║     ├─ signal: bearish
║     ├─ details: Residual Income Value: $0.00, Market Cap: $8,438,920,121.00, Difference: -100.0%
║     └─ model_details:
║       ├─ book_value: $0.00
║       ├─ roe: 12.0%
║       └─ excess_return: 7.0%
║   └─ weighted_valuation:
║     ├─ signal: bullish
║     ├─ details: Weighted Valuation: $89,172,318,971.42, Market Cap: $8,438,920,121.00, Difference: 956.7%
║     ├─ weights:
║       ├─ DCF: 35%
║       ├─ Owner Earnings: 35%
║       ├─ Relative Valuation: 15%
║       └─ Residual Income: 15%
║     └─ consistency: 0.50
║ └─ capm_data:
║   ├─ beta: 0.7848
║   ├─ required_return: 0.0500
║   ├─ risk_free_rate: 0.0001
║   ├─ market_return: 0.0068
║   └─ market_volatility: 0.1798
╚══════════════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════ 🐂 Bullish Research Analysis ═══════════════════════════════════╗
║ Confidence: 35%
║ ├─ perspective: bullish
║ ├─ confidence: 0.3524
║ ├─ thesis_points:
║   ├─ Technical indicators show bullish momentum with 0.3368983957219251 confidence
║   ├─ Strong fundamentals with 50% confidence
║   ├─ Positive market sentiment with 90% confidence
║   └─ Stock appears undervalued with 0.625 confidence
║ └─ reasoning: Bullish thesis based on comprehensive analysis of technical, fundamental, sentiment, and valuation factors
╚══════════════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════ 🐻 Bearish Research Analysis ═══════════════════════════════════╗
║ Confidence: 30%
║ ├─ perspective: bearish
║ ├─ confidence: 0.3000
║ ├─ thesis_points:
║   ├─ Technical rally may be temporary, suggesting potential reversal
║   ├─ Current fundamental strength may not be sustainable
║   ├─ Market sentiment may be overly optimistic, indicating potential risks
║   └─ Current valuation may not fully reflect downside risks
║ └─ reasoning: Bearish thesis based on comprehensive analysis of technical, fundamental, sentiment, and valuation factors
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════ 🗣️ Debate Room Analysis ══════════════════════════════════╗
║ Signal: 📉 bearish
║ Confidence: 30%
║ ├─ signal: bearish
║ ├─ confidence: 0.3000
║ ├─ bull_confidence: 0.3524
║ ├─ bear_confidence: 0.3000
║ ├─ confidence_diff: 0.0524
║ ├─ llm_score: -0.6000
║ ├─ llm_analysis: The bullish perspective highlights several key factors such as technical indicators showing bullish momentum, strong fundamentals, positive market sentiment, and an undervalued stock. However, these points have varying levels of confidence, some of which are relatively low (e.g., technical indicators at ~0.34 confidence). Conversely, the bearish view, supported by the AI model analysis, suggests that the technical rally might be short-lived, fundamentals may not be sustainable, market sentiment ...
║ ├─ llm_reasoning: The bearish arguments are supported by a high level of confidence from AI models, indicating a stronger likelihood of a downturn. Additionally, potential over-optimism in market sentiment and risks of unsustainable fundamentals further support a cautious approach. The bullish arguments, while notable, have lower confidence levels, reducing their persuasiveness.
║ ├─ mixed_confidence_diff: -0.2536
║ ├─ debate_summary:
║   ├─ Bullish Arguments:
║   ├─ + Technical indicators show bullish momentum with 0.3368983957219251 confidence
║   ├─ + Strong fundamentals with 50% confidence
║   ├─ + Positive market sentiment with 90% confidence
║   ├─ + Stock appears undervalued with 0.625 confidence
║   ├─
Bearish Arguments:
║   ├─ - Technical rally may be temporary, suggesting potential reversal
║   ├─ - Current fundamental strength may not be sustainable
║   ├─ - Market sentiment may be overly optimistic, indicating potential risks
║   └─ - Current valuation may not fully reflect downside risks
║ ├─ reasoning: Bearish arguments more convincing
║ └─ ai_model_contribution:
║   ├─ included: ✅
║   ├─ signal: bearish
║   ├─ confidence: 0.9000
║   └─ weight: 0.1500
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════ ⚠️ Risk Management Analysis ══════════════════════════════════╗
║ ├─ max_position_size: 2000.0000
║ ├─ risk_score: 4
║ ├─ trading_action: hold
║ ├─ risk_metrics:
║   ├─ volatility: 0.3464
║   ├─ value_at_risk_95: 0.0275
║   ├─ conditional_var_95: 0.0455
║   ├─ max_drawdown: -0.3268
║   ├─ skewness: 0.0188
║   ├─ kurtosis: 3.3005
║   ├─ sortino_ratio: 0.1112
║   ├─ market_risk_score: 4
║   ├─ stress_test_results:
║     └─ no_position: ✅
║   └─ macro_environment_assessment:
║     ├─ global_risks: ❌
║     ├─ liquidity_concerns: ❌
║     └─ volatility_regime: high
║ ├─ position_sizing:
║   ├─ kelly_fraction: 0.0500
║   ├─ win_rate: 0.4024
║   ├─ win_loss_ratio: 1.0476
║   ├─ risk_adjustment: 0.7000
║   └─ total_portfolio_value: 100000.0000
║ ├─ debate_analysis:
║   ├─ bull_confidence: 0.3524
║   ├─ bear_confidence: 0.3000
║   ├─ debate_confidence: 0.3000
║   └─ debate_signal: bearish
║ ├─ volatility_model:
║   ├─ model_type: GARCH(1,1)
║   ├─ parameters:
║     ├─ omega: 0.0000
║     ├─ alpha: 0.1484
║     ├─ beta: 0.7570
║     └─ persistence: 0.9054
║   ├─ log_likelihood: 1424.2592
║   ├─ forecast:
║     ├─ 0.01934439715669238
║     ├─ 0.01947384175695497
║     ├─ 0.019590300231429235
║     ├─ 0.01969514513510902
║     ├─ 0.01978959022705738
║     ├─ 0.019874711562961323
║     ├─ 0.019951465249916377
║     ├─ 0.020020702495460313
║     ├─ 0.020083182443127238
║     └─ 0.02013958318202366
║   └─ forecast_annualized:
║     ├─ 0.307082784834438
║     ├─ 0.3091376541595679
║     ├─ 0.31098637512871713
║     ├─ 0.31265073637693247
║     ├─ 0.3141500057320084
║     ├─ 0.315501265048413
║     ├─ 0.3167196920557553
║     ├─ 0.31781879925478945
║     ├─ 0.31881063767551954
║     └─ 0.3197059716488009
║ └─ reasoning: Risk Score 4/10: Market Risk=4, Volatility=34.64%, VaR=2.75%, CVaR=4.55%, Max Drawdown=-32.68%, Skewness=0.02, Debate Signal=bearish, Kelly Recommended Ratio=0.05
╚══════════════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════ 🌍 Macro Analysis ═══════════════════════════════════╗
║ Macro Environment: 📈 positive
║ Impact on Stock: 📈 positive
║ ● Key Factors:
║   • Consumer market recovery
║   • Fiscal policy support
║   • Market sentiment improvement
║   • International travel restrictions easing
║   • Regional economic development
║ ● Analysis Summary:
║   The current macroeconomic environment has a positive impact on the A-share market, especially the cultural tourism industry. First, reports show that Huangshan Tourism and other tourism companies have significant performance growth, with multiple mentions of record-high tourist traffic, indicating domestic tourism market recovery. This reflects strong recovery in consumer domestic demand, positively affecting the entire...
║ ├─ signal: positive
║ ├─ confidence: 0.7000
║ ├─ macro_environment: positive
║ ├─ impact_on_stock: positive
║ ├─ key_factors:
║   ├─ Consumer market recovery
║   ├─ Fiscal policy support
║   ├─ Market sentiment improvement
║   ├─ International travel restrictions easing
║   └─ Regional economic development
║ ├─ reasoning: The current macroeconomic environment has a positive impact on the A-share market, especially the cultural tourism industry. First, reports show that Huangshan Tourism and other tourism companies have significant performance growth, with multiple mentions of record-high tourist traffic, indicating domestic tourism market recovery. This reflects strong recovery in consumer domestic demand, positively affecting the entire cultural tourism industry. Second, fiscal policy may provide support for tourism and related industries, such as tax cuts and investment subsidies, to promote cultural tourism industry growth, directly benefiting corporate performance. Third, in terms of market sentiment, due to tourism activity and consumption recovery, investor confidence has increased, stock market liquidity has increased, risk appetite has risen, thus driving up related stock prices. Additionally, the easing of international travel restrictions may expand market space, benefiting the industry, thus enhancing the profitability of companies like Huangshan Tourism. Finally, regional economic development such as the strong growth in the Yangtze River Delta region provides Huangshan Tourism with further expansion opportunities. Therefore, overall, the current macroeconomic environment and various important factors are favorable for Huangshan Tourism and other heavily weighted stocks.
║ └─ summary: Macro Environment: positive
Impact on Stock: positive
Key Factors:
- Consumer market recovery
- Fiscal policy support
- Market sentiment improvement
- International travel restrictions easing
- Regional economic development
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════ 📂 Portfolio Management Analysis ══════════════════════════════════╗
║ Trading Action: ⏸️ HOLD
║ Trading Quantity: 171
║ Decision Confidence: 80%
║ ● Analyst Opinions:
║ ● Decision Reasoning:
║   The decision to hold is primarily dictated by the risk management constraint
║   s which recommend holding. Despite positive signals from valuation, fundamen
║   tal, technical, macro, and sentiment analyses, AI models overwhelmingly indi
║   cate a bearish outlook with high confidence. This, combined with a risk mana
║   gement action of hold, means no position is initiated. The high bullish sent
║   iment suggests potential future opportunities, but current AI signals provid
║   e caution.

AI model analysis gives bearish signal, although different from the decision direction, it has been taken into consideration and position has been appropriately adjusted.
║ ├─ action: hold
║ ├─ quantity: 171
║ ├─ confidence: 0.8000
║ ├─ agent_signals:
║   ├─ Agent 1:
║       ├─ agent_name: AI Models
║       ├─ signal: bearish
║       └─ confidence: 0.9000
║   ├─ Agent 2:
║       ├─ agent_name: Valuation Analysis
║       ├─ signal: bullish
║       └─ confidence: 0.6250
║   ├─ Agent 3:
║       ├─ agent_name: Fundamental Analysis
║       ├─ signal: bullish
║       └─ confidence: 0.5000
║   ├─ Agent 4:
║       ├─ agent_name: Technical Analysis
║       ├─ signal: bullish
║       └─ confidence: 0.3369
║   ├─ Agent 5:
║       ├─ agent_name: Macro Analysis
║       ├─ signal: positive
║       └─ confidence: 0.7000
║   └─ Agent 6:
║       ├─ agent_name: Sentiment Analysis
║       ├─ signal: bullish
║       └─ confidence: 0.9000
║ ├─ reasoning: The decision to hold is primarily dictated by the risk management constraints which recommend holding. Despite positive signals from valuation, fundamental, technical, macro, and sentiment analyses, AI models overwhelmingly indicate a bearish outlook with high confidence. This, combined with a risk management action of hold, means no position is initiated. The high bullish sentiment suggests potential future opportunities, but current AI signals provide caution.

AI model analysis gives bearish signal, although different from the decision direction, it has been taken into consideration...
║ ├─ portfolio_optimization:
║   ├─ risk_score: 4
║   ├─ kelly_fraction: 0.6000
║   ├─ risk_factor: 0.6000
║   ├─ max_position_size: 2000.0000
║   ├─ suggested_position_value: 2000.0000
║   ├─ total_portfolio_value: 100000.0000
║   ├─ position_profit_pct: 0
║   ├─ macro_adjustment: 1.0000
║   ├─ analytics:
║     ├─ multi_asset: ❌
║     ├─ expected_annual_return: 0.0201
║     ├─ expected_annual_volatility: 0.3143
║     ├─ beta_adjusted_return: 0.0201
║     ├─ sharpe_ratio: 0.0636
║     ├─ volatility_adjustment: 1.0198
║     ├─ return_multiplier: 0.8200
║     ├─ beta: 1.0000
║     ├─ market_volatility: 0.1798
║     └─ risk_free_rate: 0.0001
║   └─ market_data:
║     ├─ market_returns_mean: 0.0000
║     ├─ market_returns_std: 0.0113
║     ├─ stock_returns_mean: 0.0001
║     ├─ stock_returns_std: 0.0202
║     ├─ market_volatility: 0.1798
║     └─ stock_volatility: 0.3205
║ └─ ai_model_integration:
║   ├─ used: ✅
║   ├─ signal: bearish
║   ├─ confidence: 0.9000
║   └─ impact_on_position: 1.0000
╚══════════════════════════════════════════════════════════════════════════════╝

════════════════════════════════════════════════════════════════════════════════

Final Result:
{"action": "hold", "quantity": 171, "confidence": 0.8, "agent_signals": [{"agent_name": "AI Models", "signal": "bearish", "confidence": 0.9}, {"agent_name": "Valuation Analysis", "signal": "bullish", "confidence": 0.625}, {"agent_name": "Fundamental Analysis", "signal": "bullish", "confidence": 0.5}, {"agent_name": "Technical Analysis", "signal": "bullish", "confidence": 0.3368983957219251}, {"agent_name": "Macro Analysis", "signal": "positive", "confidence": 0.7}, {"agent_name": "Sentiment Analysis", "signal": "bullish", "confidence": 0.9}], "reasoning": "The decision to hold is primarily dictated by the risk management constraints which recommend holding. Despite positive signals from valuation, fundamental, technical, macro, and sentiment analyses, AI models overwhelmingly indicate a bearish outlook with high confidence. This, combined with a risk management action of hold, means no position is initiated. The high bullish sentiment suggests potential future opportunities, but current AI signals provide caution.\n\nAI model analysis gives bearish signal, although different from the decision direction, it has been taken into consideration and position has been appropriately adjusted.", "portfolio_optimization": {"risk_score": 4, "kelly_fraction": 0.6000000000000001, "risk_factor": 0.6, "max_position_size": 2000.0, "suggested_position_value": 2000.0, "total_portfolio_value": 100000.0, "position_profit_pct": 0, "macro_adjustment": 1.0, "analytics": {"multi_asset": false, "expected_annual_return": 0.020056438875632077, "expected_annual_volatility": 0.31425639219149415, "beta_adjusted_return": 0.020056438875632077, "sharpe_ratio": 0.06357019014466853, "volatility_adjustment": 1.0197810629914392, "return_multiplier": 0.82, "beta": 1.0, "market_volatility": 0.17975368423841376, "risk_free_rate": 7.910026984126984e-05}, "market_data": {"market_returns_mean": 2.7084239960171616e-05, "market_returns_std": 0.01132341775577318, "stock_returns_mean": 9.705980872837823e-05, "stock_returns_std": 0.02018788364201582, "market_volatility": 0.17975368423841376, "stock_volatility": 0.3204727176808965}}, "ai_model_integration": {"used": true, "signal": "bearish", "confidence": 0.9, "impact_on_position": 1.0}}
```

### Log File Description

The system will generate the following types of log files in the `logs/` directory:

1. **Backtesting Logs**

   - File name format: `backtest_{stock_code}_{current_date}_{backtest_start_date}_{backtest_end_date}.log`
   - Example: `backtest_301157_20250107_20241201_20241230.log`
   - Contains: Analysis results, trading decisions, and portfolio status for each trading day

2. **API Call Logs**
   - File name format: `api_calls_{current_date}.log`
   - Example: `api_calls_20250107.log`
   - Contains: Detailed information and responses for all API calls

All date formats are YYYY-MM-DD. If the `--show-reasoning` parameter is used, detailed analysis processes will also be recorded in the log files.

## Project Structure

```
stock-agent/
├── src/                         # Agent core logic and tools
│   ├── agents/                  # Agent definitions and workflows
│   │   ├── __init__.py
│   │   ├── debate_room.py
│   │   ├── fundamentals.py
│   │   ├── market_data.py
│   │   ├── portfolio_manager.py
│   │   ├── researcher_bear.py
│   │   ├── researcher_bull.py
│   │   ├── risk_manager.py
│   │   ├── sentiment.py
│   │   ├── state.py
│   │   ├── technicals.py
│   │   ├── valuation.py
│   │   ├── ai_model_analyst.py
│   │   ├── macro_analyst.py
│   │   └── README.md           # Detailed agent documentation
│   ├── data/                   # Data storage directory (local cache, etc.)
│   │   ├── img/                # Project images
│   │   ├── sentiment_cache.json
│   │   └── stock_news/
│   ├── tools/                  # Tools and functional modules (LLM, data acquisition)
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data_analyzer.py
│   │   ├── news_crawler.py
│   │   └── factor_data_api.py
│   ├── utils/                  # Common utility functions (logging, LLM clients, serialization)
│   │   ├── __init__.py
│   │   ├── api_utils.py        # API tools shared by agents
│   │   ├── llm_clients.py
│   │   ├── llm_interaction_logger.py
│   │   ├── logging_config.py
│   │   ├── output_logger.py
│   │   └── serialization.py
│   ├── backtester.py          # Backtesting system
│   └── main.py                # Agent workflow definition and command line entry
├── model/                     # Machine learning and deep learning models
│   ├── train/                 # Model training scripts
│   └── predict/               # Model prediction scripts
├── logs/                      # Log file directory
├── factors/                   # Factor definitions and calculations
├── .env                       # Environment variable configuration
├── .env.example               # Environment variable example
└── README.md                  # Project documentation
```

## Architecture Design

This project is a multi-agent AI investment system that adopts a modular design where each agent has its specific responsibilities. The system architecture is as follows:

```
Market Data → [Technical/Fundamentals/Sentiment/Valuation/AI Model/Macro] → [Bull/Bear Researchers] → Debate Room → Risk Manager → Portfolio Manager → Trading Decision
```

## Data Flow and Processing

### Data Types

The main data types processed by the system include market data, financial indicator data, financial statement data, and trading signals. Each data type has standardized structures and processing workflows.

### System Features

1. **Multi-LLM Support**

   - Supports OpenAI API
   - Supports any LLM service compatible with OpenAI API format (such as Huawei Cloud Ark, OpenRouter, etc.)
   - Intelligent switching functionality: automatically selects available LLM services

2. **Modular Design**

   - Each agent is an independent module
   - Easy to maintain and upgrade
   - Can be tested and optimized individually

3. **Extensibility**

   - Can easily add new analysts
   - Supports adding new data sources
   - Can extend decision strategies

4. **Risk Management**

   - Multi-level risk control
   - Real-time risk assessment
   - Automatic stop-loss mechanisms

5. **Multi-Asset Analysis**
   - Supports analyzing multiple stocks
   - Provides portfolio optimization recommendations
   - Calculates correlations and risk indicators

## Future Development Directions

1. Optimize the backtesting system to solve the over-holding (hold) problem
2. Add more machine learning models and factors
3. Add richer technical indicators and analysis methods
4. Strengthen macroeconomic analysis capabilities
5. Enhance multi-asset allocation and portfolio optimization functionality
