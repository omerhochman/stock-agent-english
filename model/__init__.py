"""
Deep Learning/Reinforcement Learning/Genetic Programming module providing a series of machine learning and artificial intelligence algorithms for quantitative trading, factor mining and market prediction.

Main Features
-------
1. Deep Learning Models
   - LSTM models for time series prediction
   - Random Forest classifier for stock screening

2. Reinforcement Learning Models
   - PPO algorithm for trading strategy optimization
   - Custom trading environment

3. Genetic Programming
   - Automated factor mining
   - Investment decision support

Usage Examples
-------

1. Using Deep Learning to Predict Stock Prices

```python
from model.dl import MLAgent, DeepLearningModule
import pandas as pd

# Prepare stock price data
price_data = pd.DataFrame({
    'open': [100, 101, 102, 103, 104],
    'high': [105, 106, 107, 108, 109],
    'low': [95, 96, 97, 98, 99],
    'close': [102, 103, 104, 105, 106],
    'volume': [1000, 1100, 1200, 1300, 1400]
}, index=pd.date_range('2023-01-01', periods=5))

# Create MLAgent instance
# Parameters:
#   model_dir: Model save directory, default is 'model'
# Returns:
#   MLAgent object
ml_agent = MLAgent(model_dir='my_model')

# Train model
# Parameters:
#   price_data: Price data DataFrame, needs to contain 'open', 'high', 'low', 'close', 'volume' columns
#   technical_indicators: Optional additional technical indicators dictionary, keys are indicator names, values are pd.Series
# Returns:
#   No direct return value, training results saved in MLAgent instance
ml_agent.train_model(price_data)

# Generate prediction signals
# Parameters:
#   price_data: Price data DataFrame
#   technical_indicators: Optional additional technical indicators dictionary
# Returns:
#   Dict[str, Any]: Dictionary containing the following key-value pairs:
#     - signal: Trading signal, 'bullish', 'bearish', 'neutral'
#     - confidence: Confidence level, float between 0 and 1
#     - lstm_predictions: LSTM model predictions, containing 'future_prices' and 'expected_returns'
#     - rf_prediction: Random forest model predictions, containing 'prediction' and 'probability'
#     - reasoning: Prediction reasoning
signals = ml_agent.generate_signals(price_data)

print(f"Trading signal: {signals['signal']}")
print(f"Confidence: {signals['confidence']:.2f}")
print(f"Predicted future prices: {signals.get('lstm_predictions', {}).get('future_prices', [])}")
print(f"Decision reasoning: {signals['reasoning']}")
```

2. Using reinforcement learning to train trading strategies

```python
from model.rl import RLTradingAgent
import pandas as pd
import numpy as np

# Prepare stock price data (more historical data needed for RL training)
# Generate simulated data
dates = pd.date_range('2022-01-01', periods=500)
prices = np.random.randn(500).cumsum() + 100  # Random walk prices
price_data = pd.DataFrame({
    'open': prices * 0.99,
    'high': prices * 1.02,
    'low': prices * 0.98,
    'close': prices,
    'volume': np.random.randint(1000, 10000, 500)
}, index=dates)

# Create RLTradingAgent instance
# Parameters:
#   model_dir: Model save directory, default 'model'
# Returns:
#   RLTradingAgent object
rl_agent = RLTradingAgent(model_dir='rl_model')

# Train model
# Parameters:
#   price_data: Price data DataFrame
#   tech_indicators: Optional technical indicators dictionary
# Returns:
#   Dict: Dictionary containing training history, including returns, losses, etc.
training_history = rl_agent.train(price_data)

# Generate trading signals
# Parameters:
#   price_data: Price data DataFrame
#   tech_indicators: Optional technical indicators dictionary
# Returns:
#   Dict[str, Any]: Dictionary containing the following key-value pairs:
#     - signal: Trading signal, 'bullish', 'bearish', 'neutral'
#     - confidence: Confidence level, float between 0 and 1
#     - action_probabilities: Probabilities for each action
#     - reasoning: Prediction reasoning
signals = rl_agent.generate_signals(price_data)

print(f"RL trading signal: {signals['signal']}")
print(f"Confidence: {signals['confidence']:.2f}")
print(f"Action probabilities: {signals.get('action_probabilities', {})}")
print(f"Decision reasoning: {signals['reasoning']}")
```

3. Using genetic programming to mine trading factors

```python
from model.deap_factors import FactorAgent
import pandas as pd
import numpy as np

# Prepare stock price data
dates = pd.date_range('2022-01-01', periods=500)
prices = np.random.randn(500).cumsum() + 100
price_data = pd.DataFrame({
    'open': prices * 0.99,
    'high': prices * 1.02,
    'low': prices * 0.98,
    'close': prices,
    'volume': np.random.randint(1000, 10000, 500)
}, index=dates)

# Create FactorAgent instance
# Parameters:
#   model_dir: Factor save directory, default 'factors'
# Returns:
#   FactorAgent object
factor_agent = FactorAgent(model_dir='my_factors')

# Generate factors
# Parameters:
#   price_data: Price data DataFrame
#   n_factors: Number of factors to generate, default 5
# Returns:
#   List[Dict]: List of generated factors, each containing name, expression, fitness, etc.
factors = factor_agent.generate_factors(price_data, n_factors=3)

# View generated factors
for factor in factors:
    print(f"Factor name: {factor['name']}")
    print(f"Factor expression: {factor['expression']}")
    print(f"Factor fitness: {factor['fitness']:.4f}")
    print(f"Information coefficient (IC): {factor['ic']:.4f}")
    print("-------------------")

# Use factors to generate trading signals
# Parameters:
#   price_data: Price data DataFrame
# Returns:
#   Dict[str, Any]: Dictionary containing the following key-value pairs:
#     - signal: Trading signal, 'bullish', 'bearish', 'neutral'
#     - confidence: Confidence level, float between 0 and 1
#     - factor_signals: Signals from each factor
#     - reasoning: Prediction reasoning
signals = factor_agent.generate_signals(price_data)

print(f"Factor composite signal: {signals['signal']}")
print(f"Confidence: {signals['confidence']:.2f}")
print(f"Decision reasoning: {signals['reasoning']}")
```

4. Combining trading signals from multiple models

```python
from model.dl import MLAgent
from model.rl import RLTradingAgent
from model.deap_factors import FactorAgent
import pandas as pd
import numpy as np

# Load models
ml_agent = MLAgent()
ml_agent.load_model()

rl_agent = RLTradingAgent()
rl_agent.load_model()

factor_agent = FactorAgent()
factor_agent.load_factors()

# Get latest price data
price_data = pd.DataFrame(...)  # Assume real-time data fetching logic here

# Get signals from each model
ml_signals = ml_agent.generate_signals(price_data)
rl_signals = rl_agent.generate_signals(price_data)
factor_signals = factor_agent.generate_signals(price_data)

# Combine signals
# Simple voting method
signals = [
    ml_signals['signal'],
    rl_signals['signal'],
    factor_signals['signal']
]

# Count each type of signal
bullish_count = signals.count('bullish')
bearish_count = signals.count('bearish')
neutral_count = signals.count('neutral')

# Composite signal
if bullish_count > bearish_count and bullish_count > neutral_count:
    final_signal = 'bullish'
    # Calculate composite confidence (weighted average)
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

print(f"Final trading signal: {final_signal}")
print(f"Confidence: {confidence:.2f}")
print("Model signals:")
print(f"  Machine learning model: {ml_signals['signal']} ({ml_signals['confidence']:.2f})")
print(f"  Reinforcement learning model: {rl_signals['signal']} ({rl_signals['confidence']:.2f})")
print(f"  Factor model: {factor_signals['signal']} ({factor_signals['confidence']:.2f})")
```
"""

from model.dl import MLAgent, DeepLearningModule, preprocess_stock_data
from model.rl import RLTradingAgent, RLTrader, StockTradingEnv
from model.deap_factors import FactorAgent, FactorMiningModule
from . import evaluate
from . import split_evaluate

# Public API
__all__ = [
    # Deep learning module
    'MLAgent',
    'DeepLearningModule',
    'preprocess_stock_data',
    
    # Reinforcement learning module
    'RLTradingAgent',
    'RLTrader',
    'StockTradingEnv',
    
    # Genetic programming module
    'FactorAgent',
    'FactorMiningModule',
    'evaluate',
    'split_evaluate'
]