# A-Share Investment System - Model Training Module

This module provides three different types of AI model training and testing functionality, including deep learning models (dl), reinforcement learning models (rl) and genetic programming factor models (factor). These models can work together to analyze the stock market from different angles, providing comprehensive trading signals and decision support.

## Quick Start

```bash
# Basic usage
python -m model.train.train --ticker 600054 --model dl

# Train all models
python -m model.train.train --ticker 600054 --model all

# Specify date range
python -m model.train.train --ticker 600054 --model dl --start-date 2023-01-01 --end-date 2023-12-31

# Evaluate model performance
python -m model.train.train --ticker 600054 --model dl --action evaluate
```

## Basic Usage

Date range is specified using `--start-date 2023-01-01 --end-date 2023-12-31`, if not specified, defaults to 2 years ago to yesterday.

```bash
# Train and test using deep learning model
python -m model.train.train --ticker 600054 --model dl

# Train only reinforcement learning model
python -m model.train.train --ticker 600054 --model rl --action train

# Test only previously trained factor model
python -m model.train.train --ticker 600054 --model factor --action test

# Train all models
python -m model.train.train --ticker 600054 --model all

# Specify custom parameters (note: Windows needs \ to escape double quotes)
python -m model.train.train --ticker 600054 --model dl --params '{"hidden_dim": 128, "epochs": 100}'
```

## Data Splitting and Model Evaluation Functionality

The system supports splitting training, validation and test sets according to specified proportions, and evaluates and visualizes model performance:

```bash
# Use default proportions (70%/20%/10%) to split data and evaluate deep learning model
python -m model.train.train --ticker 600054 --model dl --action evaluate

# Custom data splitting proportions
python -m model.train.train --ticker 600054 --model dl --action evaluate --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1

# Evaluate all models (deep learning, reinforcement learning and factor models)
python -m model.train.train --ticker 600054 --model all --action evaluate

# Specify evaluation results save directory
python -m model.train.train --ticker 600054 --model dl --action evaluate --eval-dir ./evaluation_results

# Shuffle data (default is time-ordered split)
python -m model.train.train --ticker 600054 --model dl --action evaluate --shuffle
```

## Model Type Details

### 1. Deep Learning Models (dl)

Deep learning models combine LSTM neural networks and random forest classifiers for price prediction and trend analysis.

**Main Features**:
- LSTM for predicting future 10-day price trends
- Random forest classifier for predicting price direction
- Combine results from both models to generate final signal

**Default Parameters**:
```
seq_length: 15           # Sequence length
forecast_days: 10        # Forecast days
hidden_dim: 128          # Hidden layer dimension
num_layers: 3            # LSTM layers
epochs: 100              # Training epochs
batch_size: 32           # Batch size
learning_rate: 0.0005    # Learning rate
```

**Features Used**:
- Closing price (close)
- 5-day moving average (ma5)
- 10-day moving average (ma10)
- 20-day moving average (ma20)
- Relative Strength Index (rsi)
- Moving Average Convergence Divergence (macd)

### 2. Reinforcement Learning Model (rl)

The reinforcement learning model uses PPO (Proximal Policy Optimization) algorithm to train agents, learning optimal strategies through trading in simulated environments.

**Main Features**:
- Trading strategy based on PPO algorithm
- Agents can choose buy, sell, or hold actions
- Model optimizes strategy by maximizing cumulative returns

**Default Parameters**:
```
n_episodes: 100              # Training episodes
batch_size: 32               # Batch size
reward_scaling: 1.0          # Reward scaling
initial_balance: 100000      # Initial capital
transaction_fee_percent: 0.001 # Transaction fee rate
window_size: 10              # Observation window size
max_steps: 236               # Maximum steps
```

### 3. Genetic Programming Factor Model (factor)

The genetic programming factor model automatically discovers predictive trading factors through evolutionary algorithms and uses these factors to generate trading signals.

**Main Features**:
- Automatically discovers highly predictive trading factors
- Multi-factor comprehensive scoring
- Evolutionary capability to adapt to market changes

**Default Parameters**:
```
n_factors: 3                 # Number of factors generated
population_size: 50          # Population size
n_generations: 20            # Number of iterations
future_return_periods: 5     # Future return periods
min_fitness: 0.03            # Minimum fitness
```

## Command Line Arguments

### Basic Parameters
- `--ticker`: Stock code (required)
- `--model`: Model type, options: dl, rl, factor, all
- `--action`: Operation type, options: train, test, evaluate (default: execute train and test)
- `--start-date`: Start date (default: 2 years before end date)
- `--end-date`: End date (default: yesterday)

### Advanced Parameters
- `--params`: Custom model parameters (JSON format)
  
  Example: `--params '{"hidden_dim": 256, "epochs": 200}'`

### Evaluation Parameters
- `--train-ratio`: Training set ratio (default: 0.7)
- `--val-ratio`: Validation set ratio (default: 0.2)
- `--test-ratio`: Test set ratio (default: 0.1)
- `--shuffle`: Whether to shuffle data (default: time-ordered split)
- `--eval-dir`: Evaluation results save directory (default: "models/evaluation")

## Output Signal Meanings

After model training and testing, trading signals will be generated:

1. **Signal Types**:
   - `bullish`: Bullish signal, expecting market to rise
   - `bearish`: Bearish signal, expecting market to fall
   - `neutral`: Neutral signal, market direction unclear

2. **Confidence Level**:
   - Range from 0 to 1, higher values indicate greater model confidence in prediction
   - Example: `bearish, confidence: 0.73` means 73% confidence that the market will fall

## Detailed Decision Reasoning Analysis

Taking deep learning model output as an example:
```
Decision reasoning: LSTM model predicts next 10 days: short-term return (-0.31%), medium-term return (-17.68%), long-term return (-18.54%). Expected positive return days ratio: 0.00%; Random forest model predicts fall probability: 55.86%, rise probability: 44.14%; LSTM technical analysis result: bearish, confidence: 90.00%; Random forest technical analysis result: bearish, confidence: 55.86%; Comprehensive analysis generates bearish signal, confidence: 72.93%. Please note that even under bearish signals, the market may still rebound, suggesting timely profit-taking.
```

## Model Voting Mechanism

When using `--model all`, the system combines signals from all three models:

1. Count signals from each model (bullish, bearish, neutral)
2. Calculate weighted score (range from -1 to 1)
3. Generate final comprehensive signal

## Actual Running Effect Example

```
==================================================
Comprehensive Results:
==================================================
dl     model: bearish  (confidence: 0.73)
rl     model: bearish  (confidence: 1.00)
factor model: bearish  (confidence: 0.70)

Model Voting Results:

Bullish: 0, Bearish: 3, Neutral: 0

Weighted Score: -1.00 (range: -1 to 1)

Comprehensive Signal: bearish
```

## Evaluation Results Output

Evaluation mode (`--action evaluate`) will generate the following content:

1. Training set and test set data distribution comparison charts
2. Model performance metrics (MSE, RMSE, MAE, R², etc.)
3. Prediction results visualization charts
4. Model prediction error analysis
5. Feature importance analysis (applicable to random forest models)
6. Future price prediction display

All evaluation results are saved by default in the `models/evaluation` directory.

## Model Training Log Example

The system will generate detailed training logs, including training progress, model performance, and final results:

```
2025-05-07 19:33:18 - deep_learning - INFO - Starting LSTM model training...
2025-05-07 19:33:18 - deep_learning - INFO - LSTM training using feature columns: ['close', 'ma5', 'ma10', 'ma20', 'rsi', 'macd']
2025-05-07 19:33:18 - deep_learning - INFO - Initializing LSTM model, input dimension: 6, hidden dimension: 128, layers: 3
2025-05-07 19:33:20 - deep_learning - INFO - Epoch 1/100, Training Loss: 0.1068, Validation Loss: 0.0213
...
2025-05-07 19:33:32 - deep_learning - INFO - LSTM model training completed
2025-05-07 19:33:32 - deep_learning - INFO - Label distribution - Rise: 40.17%, Fall: 59.83%
```

```
2025-05-07 19:33:53 - factor_mining - INFO - Starting factor evolution, population size: 100, iterations: 50
gen     nevals  avg             min     max             std
0       100     -0.376797       -0.5    0.210664        0.205371
...
50      56      0.15184         -0.5    0.236512        0.222616
2025-05-07 19:34:55 - factor_mining - INFO - Factor GP_Factor_1 generation completed, fitness: 0.2365
```

## Notes

1. **Data Requirements**: Reinforcement learning models require sufficient historical data, recommend using at least one year of data
2. **Model Storage**: Trained models will be saved in the `models/` directory
3. **Windows Path**: Windows systems need to escape double quotes when using custom parameters, for example:
   ```
   python -m model.train.train --ticker 600054 --model dl --params "{\"hidden_dim\": 128}"
   ```

## Known Issues

In some cases, reinforcement learning model training may show the following warning:
```
Warning: Insufficient data. window_size=20, max_steps=252, data length=100
Adjusted max_steps to 75
```

This is because the data length is insufficient to meet the default window_size and max_steps parameters, and the system will automatically adjust parameters to continue training.
