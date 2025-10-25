# Investment Analysis System Agent Module (src/agents)

## Overview

This module is the core component of an AI investment analysis framework based on a multi-agent system, implementing a complete investment research and decision pipeline. The system adopts modern portfolio theory, behavioral finance, and machine learning technologies to provide comprehensive stock analysis and investment recommendations.

## System Architecture

The system adopts a distributed agent architecture where each agent focuses on specific analysis areas, ultimately forming investment decisions through intelligent aggregation:

```
Data Collection → Multi-dimensional Analysis → AI Model Prediction → Debate Evaluation → Risk Management → Portfolio Decision
```

## Core Agent Modules

### 1. Market Data Agent (market_data.py)

**Function**: Responsible for collecting and preprocessing all market-related data

- **Core Features**:
  - Supports single and multi-asset data collection
  - Obtains stock price history, financial indicators, market data
  - Data validation and cleaning
  - Unified data formatting

**Main Methods**:

```python
@agent_endpoint("market_data", "Market data collection, responsible for obtaining stock price history, financial indicators and market information")
def market_data_agent(state: AgentState):
    # Collect complete market data for multiple stocks
    # Return standardized data structure
```

### 2. Technical Analysis Agent (technicals.py)

**Function**: Regime-aware technical analysis system based on 2024-2025 research

- **Core Algorithms**:
  - Trend following strategies (EMA, ADX, Ichimoku)
  - Mean reversion strategies (RSI, Bollinger Bands, KD indicators)
  - Momentum strategies (multi-period momentum, OBV)
  - GARCH volatility prediction model
  - Statistical arbitrage signals

**Advanced Features**:

- **Regime Detection**: Uses Gaussian mixture models to automatically identify market states
- **Dynamic Weights**: Adjusts strategy weights based on market regime
- **Signal Enhancement**: Filters and enhances signals based on regime characteristics

```python
# Core signal combination logic
regime_adjusted_weights = _calculate_regime_adjusted_weights(current_regime)
combined_signal = weighted_signal_combination({
    'trend': trend_signals,
    'mean_reversion': mean_reversion_signals,
    'momentum': momentum_signals,
    'volatility': volatility_signals
}, regime_adjusted_weights)
```

### 3. Fundamental Analysis Agent (fundamentals.py)

**Function**: Comprehensive financial indicator analysis

- **Analysis Dimensions**:
  - Profitability analysis (ROE, net profit margin, operating margin)
  - Growth analysis (revenue growth, profit growth, book value growth)
  - Financial health (current ratio, debt ratio, cash flow)
  - Valuation ratios (PE, PB, PS ratios)

**Decision Logic**:

```python
# Multi-dimensional signal aggregation
signals = [profitability_signal, growth_signal, health_signal, valuation_signal]
overall_signal = determine_signal_by_majority(signals)
confidence = calculate_weighted_confidence(signals)
```

### 4. Valuation Analysis Agent (valuation.py)

**Function**: Evaluates intrinsic value using multiple valuation models

- **Valuation Methods**:
  - **DCF Model**: Three-stage discounted cash flow
  - **Owner Earnings Method**: Improved Buffett valuation method
  - **Relative Valuation**: Industry comparison analysis
  - **Residual Income Model**: ROE-based valuation

**Core Algorithm**:

```python
# Weighted valuation combination
all_valuations = [
    {"method": "DCF", "value": dcf_value, "weight": 0.35},
    {"method": "Owner Earnings", "value": owner_earnings_value, "weight": 0.35},
    {"method": "Relative Valuation", "value": relative_value, "weight": 0.15},
    {"method": "Residual Income", "value": residual_income_value, "weight": 0.15}
]
```

### 5. Sentiment Analysis Agent (sentiment.py)

**Function**: Analysis based on news and market sentiment

- **Data Source**: News data within 7 days
- **Analysis Method**: NLP sentiment analysis
- **Signal Generation**: Conversion from sentiment score to trading signal

### 6. Macro Analysis Agent (macro_analyst.py)

**Function**: Analysis of macroeconomic environment impact on individual stocks

- **Analysis Factors**:
  - Monetary policy (interest rates, reserve requirement ratio)
  - Fiscal policy (government spending, taxation)
  - Industrial policy (industry planning, regulation)
  - International environment (global economy, trade relations)

### 7. AI Model Analysis Agent (ai_model_analyst.py)

**Function**: Integrates deep learning, reinforcement learning, and genetic programming models

- **Model Types**:
  - **Deep Learning**: LSTM + Random Forest combination
  - **Reinforcement Learning**: PPO algorithm for trading strategy optimization
  - **Genetic Programming**: Automated factor mining

**Signal Aggregation**:

```python
# Multi-model signal combination
weights = {
    'deep_learning': 0.35,
    'reinforcement_learning': 0.35,
    'genetic_programming': 0.30
}
combined_signal = combine_ai_signals(ml_signals, rl_signals, factor_signals)
```

### 8. Debate Room Agent (debate_room.py)

**Function**: Adaptive signal aggregation system based on 2024-2025 research

- **Core Technologies**:
  - FLAG-Trader framework's regime-aware aggregation
  - FINSABER weight optimization
  - Lopez-Lira dynamic thresholds

**Aggregation Algorithm**:

```python
def adaptive_signal_aggregation(signals: Dict, regime_info: Dict, confidence_threshold: float = 0.6):
    # Dynamically adjust signal weights based on market regime
    regime_adjustments = {
        "low_volatility_trending": {'technical': 1.3, 'ai_model': 1.2},
        "high_volatility_mean_reverting": {'fundamental': 1.4, 'valuation': 1.3},
        "crisis_regime": {'macro': 1.5, 'sentiment': 1.2}
    }
```

### 9. Risk Management Agent (risk_manager.py)

**Function**: Comprehensive risk assessment based on modern portfolio theory

- **Risk Indicators**:
  - VaR and CVaR calculation
  - Maximum drawdown analysis
  - GARCH volatility prediction
  - Stress testing
  - Regime risk assessment

**Position Sizing Optimization**:

```python
# Regime-aware Kelly criterion
kelly_fraction = win_rate - ((1 - win_rate) / win_loss_ratio)
# Adjust conservative factor based on market regime
conservative_factor = 0.3 if regime_name == "crisis_regime" else 0.5
```

### 10. Portfolio Analysis Agent (portfolio_analyzer.py)

**Function**: Multi-asset portfolio optimization and risk assessment

- **Analysis Functions**:
  - Efficient frontier generation
  - Sharpe ratio optimization
  - Correlation analysis
  - Beta coefficient calculation
  - Tail risk measurement

### 11. Portfolio Management Agent (portfolio_manager.py)

**Function**: Final trading decisions and portfolio management

- **Decision Integration**: Synthesizes all analyst recommendations
- **Modern Portfolio Theory**: Uses MPT to optimize decisions
- **LLM Enhancement**: Uses large language models for final decisions

**Decision Weights**:

```python
# Signal weight allocation
weights = {
    'ai_models': 0.15,      # AI model predictions
    'valuation': 0.35,      # Valuation analysis (main driver)
    'fundamental': 0.30,    # Fundamental analysis
    'technical': 0.25,      # Technical analysis
    'macro': 0.15,          # Macro analysis
    'sentiment': 0.10       # Sentiment analysis
}
```

### 12. Bull/Bear Researchers (researcher_bull.py / researcher_bear.py)

**Function**: Analyze markets from different perspectives to provide diverse viewpoints

- **Bull Researcher**: Seeks investment opportunities and positive factors
- **Bear Researcher**: Identifies risks and negative factors
- **Risk Adjustment**: Adjusts confidence based on overall market environment

## Advanced Features

### Regime Detection System (regime_detector.py)

Advanced market regime detection based on 2024-2025 research:

```python
class AdvancedRegimeDetector:
    def __init__(self, n_regimes: int = 3):
        self.regime_names = {
            0: "low_volatility_trending",
            1: "high_volatility_mean_reverting",
            2: "crisis_regime"
        }
```

**Feature Engineering**:

- Multi-timeframe volatility
- Trend strength indicators
- Momentum features
- Market microstructure
- Hurst exponent (long memory)

### State Management System (state.py)

Unified agent state management:

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[Dict[str, Any], merge_dicts]
    metadata: Annotated[Dict[str, Any], merge_dicts]
```

## Data Flow Architecture

1. **Data Collection Stage**: market_data_agent collects raw data
2. **Analysis Stage**: Professional agents perform parallel analysis
3. **AI Prediction Stage**: ai_model_analyst generates machine learning predictions
4. **Integration Stage**: debate_room_agent intelligently aggregates signals
5. **Risk Assessment**: risk_manager evaluates risks and sets constraints
6. **Decision Stage**: portfolio_manager generates final trading decisions

## Performance Optimization

- **Caching Mechanism**: Time-consuming operations like macro analysis use caching
- **Parallel Processing**: Multiple agents can execute in parallel
- **Dynamic Thresholds**: Adaptive adjustment based on market conditions
- **Error Handling**: Complete exception handling and fallback mechanisms

## Configuration and Extension

### Adding New Agents

```python
@agent_endpoint("new_agent", "New agent description")
def new_agent(state: AgentState):
    # Implement agent logic
    return {
        "messages": [message],
        "data": updated_data,
        "metadata": metadata
    }
```

### Custom Weights

The system supports dynamic adjustment of agent weights based on market conditions, which can be adjusted through configuration files or runtime parameters.

## Research Foundation

This system is based on the latest financial AI research from 2024-2025:

- **FLAG-Trader**: Regime-aware signal aggregation
- **FINSABER**: Multi-factor signal integration framework
- **Lopez-Lira**: Dynamic thresholds and weight adjustment
- **RLMF**: Reinforcement learning in finance applications

## Usage Examples

```python
# Initialize system state
state = {
    "messages": [],
    "data": {"ticker": "000001", "tickers": ["000001", "000002"]},
    "metadata": {"show_reasoning": True}
}

# Execute analysis pipeline
result = market_data_agent(state)
result = technical_analyst_agent(result)
result = fundamentals_agent(result)
# ... other agents
final_decision = portfolio_management_agent(result)
```

## Output Format

Each agent returns standardized JSON format results containing:

- `signal`: Investment signal (bullish/bearish/neutral)
- `confidence`: Confidence level (0-1)
- `reasoning`: Detailed analysis reasoning
- `metrics`: Related calculation indicators

The final portfolio management agent outputs complete trading decisions, including specific buy/sell recommendations and position sizes.
