# Model Module

## Module Structure

```
model/
├── __init__.py              # Module initialization, public API
├── dl.py                    # Deep learning models (LSTM + Random Forest)
├── rl.py                    # Reinforcement learning models (PPO algorithm)
├── deap_factors.py          # Genetic programming factor mining
├── evaluate.py              # Model evaluation framework
├── split_evaluate.py        # Data splitting and evaluation tools
└── train/
    ├── train.py            # Unified training interface
    └── readme.md           # Detailed training module documentation
```

## Core Model Introduction

### 1. Deep Learning Models (dl.py)

Deep learning models combine LSTM neural networks and Random Forest classifiers to make predictions from both time series and cross-sectional dimensions.

**Core Components**:

- **LSTMModel**: PyTorch-based LSTM network for price sequence prediction
- **RandomForestClassifier**: sklearn-based classifier for price direction prediction
- **MLAgent**: Unified model management interface

**Main Functions**:

```python
from model.dl import MLAgent

# Initialize and train models
ml_agent = MLAgent(model_dir='my_model')
ml_agent.train_models(price_data, epochs=100, hidden_dim=128)

# Generate trading signals
signals = ml_agent.generate_signals(price_data)
print(f"Signal: {signals['signal']}, Confidence: {signals['confidence']}")
```

**Technical Features**:

- Support for multi-feature input (prices, technical indicators)
- Automatic data preprocessing and normalization
- Multi-step prediction and probability output
- Model persistence and loading

### 2. Reinforcement Learning Models (rl.py)

Reinforcement learning models use PPO (Proximal Policy Optimization) algorithm to train intelligent trading agents.

**Core Components**:

- **StockTradingEnv**: OpenAI Gym compatible trading environment
- **ActorCritic**: PPO policy-value network
- **PPOAgent**: PPO algorithm implementation
- **RLTradingAgent**: Unified reinforcement learning interface

**Main Functions**:

```python
from model.rl import RLTradingAgent

# Train reinforcement learning model
rl_agent = RLTradingAgent(model_dir='rl_model')
training_history = rl_agent.train(price_data, n_episodes=500)

# Generate trading signals
signals = rl_agent.generate_signals(price_data)
print(f"RL Signal: {signals['signal']}, Action Probabilities: {signals['action_probabilities']}")
```

**Technical Features**:

- Simulation trading environment based on real market data
- Support for buy, sell, hold actions
- Consider transaction costs and slippage
- GPU accelerated training

### 3. Genetic Programming Factor Models (deap_factors.py)

Genetic programming models automatically discover and optimize trading factors through evolutionary algorithms.

**Core Components**:

- **FactorMiningModule**: Genetic programming core engine
- **FactorAgent**: Factor mining and signal generation interface
- **Evolution Operators**: Crossover, mutation, selection and other genetic operations

**Main Functions**:

```python
from model.deap_factors import FactorAgent

# Generate trading factors
factor_agent = FactorAgent(model_dir='my_factors')
factors = factor_agent.generate_factors(price_data, n_factors=5)

# View generated factors
for factor in factors:
    print(f"Factor: {factor['name']}, Expression: {factor['expression']}")

# Generate trading signals
signals = factor_agent.generate_signals(price_data)
```

**Technical Features**:

- Automatic factor discovery and expression generation
- Multi-objective optimization (IC, return, Sharpe ratio, etc.)
- Factor complexity control
- Safe mathematical operation protection

## Model Evaluation Framework (evaluate.py)

Provides complete model performance evaluation and visualization functionality.

**Core Functions**:

```python
from model.evaluate import ModelEvaluator

evaluator = ModelEvaluator(output_dir='evaluation')

# Regression model evaluation
metrics = evaluator.evaluate_regression_model(y_true, y_pred, 'LSTM', 'test')

# Classification model evaluation
metrics = evaluator.evaluate_classification_model(y_true, y_pred, 'RF', 'test')

# Prediction result visualization
evaluator.visualize_predictions(y_true, y_pred, date_index, 'model', 'test')
```

**Evaluation Metrics**:

- **Regression**: MSE, RMSE, MAE, R²
- **Classification**: Accuracy, Precision, Recall, F1 Score
- **Visualization**: Prediction comparison charts, error analysis, confusion matrix

## Unified Training Interface

### Command Line Training Tools

```bash
# Train deep learning model
python -m model.train.train --ticker 600519 --model dl

# Train all models
python -m model.train.train --ticker 600519 --model all

# Data split evaluation
python -m model.train.train --ticker 600519 --model dl --action evaluate

# Custom parameter training
python -m model.train.train --ticker 600519 --model dl --params '{"epochs": 200, "hidden_dim": 256}'
```

### Programming Interface

```python
# Import all core components
from model import MLAgent, RLTradingAgent, FactorAgent

# Train three types of models separately
ml_agent = MLAgent()
ml_agent.train_models(price_data)

rl_agent = RLTradingAgent()
rl_agent.train(price_data)

factor_agent = FactorAgent()
factor_agent.generate_factors(price_data)

# Get comprehensive signals
ml_signals = ml_agent.generate_signals(price_data)
rl_signals = rl_agent.generate_signals(price_data)
factor_signals = factor_agent.generate_signals(price_data)
```

## Signal Generation and Fusion

### Signal Standard Format

All models generate signals following a unified format:

```python
{
    'signal': 'bullish',        # Signal type: bullish/bearish/neutral
    'confidence': 0.78,         # Confidence: 0-1
    'reasoning': '...',         # Decision reasoning
    'model_specific': {...}     # Model-specific information
}
```

### Multi-Model Signal Fusion

```python
# Simple voting method
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

# Weighted average method
def weighted_combine(signals_list, weights):
    weighted_sum = sum(w * signal_to_score(s) for w, s in zip(weights, signals_list))
    return score_to_signal(weighted_sum)
```

## Technical Indicators and Feature Engineering

### Automatic Technical Indicator Calculation

The module automatically calculates common technical indicators:

```python
# Price indicators
- Moving averages (MA5, MA10, MA20, MA60)
- Price change rates (1-day, 5-day, 10-day, 20-day)
- Volatility (5-day, 10-day, 20-day)

# Technical indicators
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Volume indicators

# Feature engineering
- Lag features
- Rolling statistics
- Price-volume relationships
- Market microstructure
```

### Data Preprocessing Pipeline

```python
def preprocess_stock_data(price_df, technical_indicators=None):
    # 1. Data cleaning and formatting
    df = price_df.copy()
    df['returns'] = df['close'].pct_change()

    # 2. Technical indicator calculation
    df['ma5'] = df['close'].rolling(5).mean()
    df['rsi'] = calculate_rsi(df['close'])
    df['macd'] = calculate_macd(df['close'])

    # 3. Lag features
    for lag in [1, 2, 3, 5, 10]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)

    # 4. Rolling statistics
    for window in [5, 10, 20]:
        df[f'volatility_{window}d'] = df['returns'].rolling(window).std()

    return df.dropna()
```

## Usage Notes

### Data Requirements

1. **Minimum Data Volume**: Recommend at least 252 trading days (1 year) of historical data
2. **Data Quality**: Ensure price data is complete with no outliers
3. **Feature Alignment**: Ensure all features are aligned in time

### Training Recommendations

1. **Deep Learning**:

   - Use longer sequence lengths (10-20 days)
   - Appropriate hidden layer dimensions (64-256)
   - Early stopping mechanism to prevent overfitting

2. **Reinforcement Learning**:

   - Sufficient training episodes (500-2000)
   - Reasonable reward function design
   - Appropriate exploration-exploitation balance

3. **Genetic Programming**:
   - Control factor complexity
   - Set appropriate fitness thresholds
   - Avoid overfitting
