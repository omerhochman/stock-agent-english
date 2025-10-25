#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Add project root directory to system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

import matplotlib

from model.deap_factors import FactorAgent, FactorMiningModule
from model.dl import MLAgent, preprocess_stock_data
from model.rl import RLTrader, RLTradingAgent, StockTradingEnv
from model.split_evaluate import split_and_evaluate
from src.tools.api import get_price_history

# Configure Chinese fonts based on operating system
if sys.platform.startswith("win"):
    # Windows system
    matplotlib.rc("font", family="Microsoft YaHei")
elif sys.platform.startswith("linux"):
    # Linux system
    matplotlib.rc("font", family="WenQuanYi Micro Hei")
else:
    # macOS system
    matplotlib.rc("font", family="PingFang SC")

# For normal display of negative signs
matplotlib.rcParams["axes.unicode_minus"] = False


def process_data(ticker, start_date, end_date, verbose=True):
    """Get and process stock data"""
    if verbose:
        print(f"Getting data for {ticker} from {start_date} to {end_date}...")

    try:
        prices_data = get_price_history(ticker, start_date, end_date)

        if isinstance(prices_data, list):
            if verbose:
                print(
                    f"Got {len(prices_data)} records (list format), converting to DataFrame..."
                )
            prices_df = pd.DataFrame(prices_data)

            # Ensure date column format is correct
            if "date" in prices_df.columns:
                prices_df["date"] = pd.to_datetime(prices_df["date"])
        else:
            if verbose:
                if hasattr(prices_data, "shape"):
                    print(f"Got {prices_data.shape[0]} records...")
                else:
                    print(f"Got data type: {type(prices_data)}...")
            prices_df = prices_data

        if verbose:
            if isinstance(prices_df, pd.DataFrame):
                print(f"Data processing completed. Shape: {prices_df.shape}")
                if not prices_df.empty:
                    print(f"Column names: {list(prices_df.columns)}")
            else:
                print(f"Data processing completed. Data type: {type(prices_df)}")

        return prices_df

    except Exception as e:
        print(f"Error getting or processing data: {e}")
        return None


def add_technical_indicators(df):
    """Add missing technical indicators"""
    # Ensure DataFrame has necessary columns
    if "close" not in df.columns and "Close" in df.columns:
        df["close"] = df["Close"]
    if "open" not in df.columns and "Open" in df.columns:
        df["open"] = df["Open"]
    if "high" not in df.columns and "High" in df.columns:
        df["high"] = df["High"]
    if "low" not in df.columns and "Low" in df.columns:
        df["low"] = df["Low"]
    if "volume" not in df.columns and "Volume" in df.columns:
        df["volume"] = df["Volume"]

    # Add necessary technical indicators (if they don't exist)
    if "ma10" not in df.columns:
        df["ma10"] = df["close"].rolling(window=10).mean()

    if "rsi" not in df.columns:
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))

    if (
        "macd" not in df.columns
        or "macd_signal" not in df.columns
        or "macd_hist" not in df.columns
    ):
        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = exp1 - exp2
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

    if "volatility_5d" not in df.columns:
        df["volatility_5d"] = df["close"].pct_change().rolling(window=5).std()

    if "volatility_10d" not in df.columns:
        df["volatility_10d"] = df["close"].pct_change().rolling(window=10).std()

    if "volatility_20d" not in df.columns:
        df["volatility_20d"] = df["close"].pct_change().rolling(window=20).std()

    # Fill NaN values
    df = df.ffill().bfill().fillna(0)

    return df


def train_dl_model(prices_df, params=None, save_dir="models", verbose=True):
    """Train deep learning model"""
    if verbose:
        print("Preparing to train deep learning model...")

    try:
        # Preprocess data
        processed_data = preprocess_stock_data(prices_df)

        if verbose:
            print(f"Preprocessing completed, data shape: {processed_data.shape}")

        # Set default parameters (not directly passed to train_models, but set separately)
        default_params = {
            "seq_length": 15,  # Increase sequence length to capture more historical information
            "forecast_days": 10,  # More meaningful forecast period
            "hidden_dim": 128,  # Increase hidden layer dimension, improve model capacity
            "num_layers": 3,  # Increase number of layers
            "epochs": 100,  # Increase training epochs
            "batch_size": 32,  # Appropriately adjust batch size
            "learning_rate": 0.0005,  # Reduce learning rate to improve stability
        }

        # Merge user-provided parameters
        if params:
            for key, value in params.items():
                if key in default_params:
                    default_params[key] = value

        if verbose:
            print("Training model with the following parameters:")
            for key, value in default_params.items():
                print(f"  {key}: {value}")

        # Initialize and train model
        ml_agent = MLAgent(model_dir=save_dir)

        # Check if data volume is sufficient
        if len(processed_data) < 50:
            print(
                f"Warning: Insufficient data ({len(processed_data)} rows), model training may be inadequate"
            )

        # Prepare feature columns
        # Ensure all necessary features are in the processed data
        feature_cols = ["close", "ma5", "ma10", "ma20", "rsi", "macd"]
        missing_features = [
            col for col in feature_cols if col not in processed_data.columns
        ]
        if missing_features:
            print(
                f"Warning: The following features are missing from the data: {missing_features}"
            )
            print("Will use available feature columns")
            feature_cols = [
                col for col in feature_cols if col in processed_data.columns
            ]
            if not feature_cols:
                feature_cols = ["close"]
                print("Only using close column as feature")

        if verbose:
            print(f"Using the following feature columns: {feature_cols}")

        # Custom parameters
        custom_params = {
            "seq_length": default_params["seq_length"],
            "forecast_days": default_params["forecast_days"],
            "hidden_dim": default_params["hidden_dim"],
            "num_layers": default_params["num_layers"],
            "epochs": default_params["epochs"],
            "batch_size": default_params["batch_size"],
            "learning_rate": default_params["learning_rate"],
            "feature_cols": feature_cols,  # Use pre-validated feature columns
        }

        # Ensure custom parameters are passed to MLAgent
        ml_agent.train_models(
            processed_data,
            seq_length=custom_params["seq_length"],
            feature_cols=custom_params["feature_cols"],
            hidden_dim=custom_params["hidden_dim"],
            num_layers=custom_params["num_layers"],
            forecast_days=custom_params["forecast_days"],
            epochs=custom_params["epochs"],
            batch_size=custom_params["batch_size"],
            learning_rate=custom_params["learning_rate"],
        )

        # Generate trading signals
        signals = ml_agent.generate_signals(processed_data)

        if verbose:
            print(
                f"Training completed. Generated trading signal: {signals.get('signal', 'unknown')}, confidence: {signals.get('confidence', 0)}"
            )
            if "reasoning" in signals:
                print(f"Signal analysis: {signals['reasoning']}")

        return ml_agent, signals

    except Exception as e:
        print(f"Error training deep learning model: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def train_rl_model(prices_df, params=None, save_dir="models", verbose=True):
    """Train reinforcement learning model"""
    if verbose:
        print("Preparing to train reinforcement learning model...")

    try:
        # Check if data volume is sufficient
        if len(prices_df) < 100:
            print(
                f"Warning: Insufficient data ({len(prices_df)} rows), may not be enough to train RL model."
            )
            print("Recommend using at least 100 trading days of data.")

        # Add missing technical indicators
        enhanced_df = add_technical_indicators(prices_df.copy())

        # Confirm all necessary technical indicators have been added
        required_indicators = [
            "ma10",
            "rsi",
            "macd",
            "macd_signal",
            "macd_hist",
            "volatility_5d",
            "volatility_10d",
            "volatility_20d",
        ]
        missing_indicators = [
            ind for ind in required_indicators if ind not in enhanced_df.columns
        ]
        if missing_indicators:
            print(f"Warning: Still missing technical indicators: {missing_indicators}")
            print("Attempting to continue training...")

        # Set default parameters
        window_size = 10
        available_data_points = len(enhanced_df) - window_size
        default_params = {
            "n_episodes": 100,
            "batch_size": 32,
            "reward_scaling": 1.0,
            "initial_balance": 100000,
            "transaction_fee_percent": 0.001,
            "window_size": window_size,
            "max_steps": max(
                20, available_data_points // 2
            ),  # Use half of available data points, but at least 20 steps
        }

        # Merge user-provided parameters
        if params:
            for key, value in params.items():
                if key in default_params:
                    default_params[key] = value

        if verbose:
            print("Training model with the following parameters:")
            for key, value in default_params.items():
                print(f"  {key}: {value}")

        # Temporarily save and load data to ensure correct format
        temp_csv = os.path.join(save_dir, "temp_training_data.csv")
        enhanced_df.to_csv(temp_csv, index=False)
        df_for_training = pd.read_csv(temp_csv)

        # Convert column types and fill NaN
        for col in df_for_training.columns:
            if col == "date":
                df_for_training[col] = pd.to_datetime(df_for_training[col])
            elif df_for_training[col].dtype == "object":
                df_for_training[col] = pd.to_numeric(
                    df_for_training[col], errors="coerce"
                )

        # Check and fill NaN again
        df_for_training = df_for_training.fillna(0)

        # Check for infinite values
        values_array = df_for_training.select_dtypes(include=[np.number]).values.astype(
            float
        )
        if np.isinf(values_array).any():
            print("Warning: Infinite values found in data, replacing with 0")
            df_for_training = df_for_training.replace([np.inf, -np.inf], 0)

        # Initialize RL trading system
        rl_trader = RLTrader(model_dir=save_dir)

        # Train model
        if verbose:
            print("Starting RL model training, this may take some time...")

        training_history = rl_trader.train(
            df=df_for_training,
            initial_balance=default_params["initial_balance"],
            transaction_fee_percent=default_params["transaction_fee_percent"],
            n_episodes=default_params["n_episodes"],
            batch_size=default_params["batch_size"],
            reward_scaling=default_params["reward_scaling"],
            max_steps=default_params["max_steps"],
        )

        # Create RLTradingAgent and load the newly trained model
        rl_agent = RLTradingAgent(model_dir=save_dir)
        success = rl_agent.load_model("best_model")

        # Only generate signals if loading is successful
        if success:
            signals = rl_agent.generate_signals(df_for_training)
            if verbose:
                print(
                    f"Training completed. Generated trading signal: {signals.get('signal', 'unknown')}, confidence: {signals.get('confidence', 0)}"
                )
        else:
            print(
                "Warning: Unable to load model after training, training may have failed"
            )
            signals = {"signal": "neutral", "confidence": 0.5}

        # Clean up temporary files
        try:
            os.remove(temp_csv)
        except:
            pass

        # Only return agent if model loading is successful
        return (rl_agent, signals) if success else (None, signals)

    except Exception as e:
        print(f"Error training reinforcement learning model: {e}")
        import traceback

        traceback.print_exc()
        return None, {"signal": "neutral", "confidence": 0.5}


def train_factor_model(prices_df, params=None, save_dir="factors", verbose=True):
    """Train genetic programming factor model"""
    if verbose:
        print("Preparing to train genetic programming factor model...")

    try:
        # Set default parameters
        default_params = {
            "n_factors": 3,  # Number of factors
            "population_size": 50,  # Population size
            "n_generations": 20,  # Number of generations
            "future_return_periods": 5,
            "min_fitness": 0.03,  # Fitness threshold
        }

        # Merge user-provided parameters
        if params:
            for key, value in params.items():
                if key in default_params:
                    default_params[key] = value

        if verbose:
            print("Generating factors with the following parameters:")
            for key, value in default_params.items():
                print(f"  {key}: {value}")

        # Initialize factor mining module
        factor_agent = FactorAgent(model_dir=save_dir)

        # Ensure data format is correct
        if "close" not in prices_df.columns and "Close" in prices_df.columns:
            prices_df["close"] = prices_df["Close"]
        if "open" not in prices_df.columns and "Open" in prices_df.columns:
            prices_df["open"] = prices_df["Open"]
        if "high" not in prices_df.columns and "High" in prices_df.columns:
            prices_df["high"] = prices_df["High"]
        if "low" not in prices_df.columns and "Low" in prices_df.columns:
            prices_df["low"] = prices_df["Low"]
        if "volume" not in prices_df.columns and "Volume" in prices_df.columns:
            prices_df["volume"] = prices_df["Volume"]

        # Pre-add some basic technical indicators to enrich feature space
        enhanced_df = prices_df.copy()

        # Add some basic technical indicators
        if verbose:
            print("Adding basic technical indicators to enrich feature space...")

        # Add some basic moving averages as starting points
        enhanced_df["ma5"] = enhanced_df["close"].rolling(window=5).mean()
        enhanced_df["ma10"] = enhanced_df["close"].rolling(window=10).mean()
        enhanced_df["ma20"] = enhanced_df["close"].rolling(window=20).mean()

        # Add some basic price momentum indicators
        enhanced_df["returns_1d"] = enhanced_df["close"].pct_change(1)
        enhanced_df["returns_5d"] = enhanced_df["close"].pct_change(5)

        # Add volatility indicators
        enhanced_df["volatility_10d"] = (
            enhanced_df["returns_1d"].rolling(window=10).std()
        )

        # Fill NaN values
        enhanced_df = enhanced_df.fillna(0)

        # Generate factors
        if verbose:
            print("Starting factor generation, this may take a long time...")

        # Generate factors
        factors = factor_agent.generate_factors(
            price_data=enhanced_df, n_factors=default_params["n_factors"]
        )

        # Generate trading signals
        signals = factor_agent.generate_signals(enhanced_df)

        if verbose:
            print(
                f"Factor generation completed. Generated trading signal: {signals.get('signal', 'unknown')}, confidence: {signals.get('confidence', 0)}"
            )
            print(f"Generated {len(factors)} factors")

        return factor_agent, signals

    except Exception as e:
        print(f"Error training genetic programming factor model: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def load_model(model_type, model_dir, verbose=True):
    """Load trained model"""
    if verbose:
        print(f"Loading {model_type} model...")

    try:
        if model_type == "dl":
            agent = MLAgent(model_dir=model_dir)
            agent.load_models()
            if verbose:
                print("Deep learning model loaded successfully")
            return agent

        elif model_type == "rl":
            agent = RLTradingAgent(model_dir=model_dir)

            # Check if model file exists
            model_path = os.path.join(model_dir, "best_model.pth")
            if not os.path.exists(model_path):
                if verbose:
                    print(f"Model file {model_path} does not exist, cannot load")
                return None

            success = agent.load_model("best_model")  # Specify model name
            if success and verbose:
                print("Reinforcement learning model loaded successfully")
            elif verbose:
                print("Reinforcement learning model loading failed")
            return agent if success else None

        elif model_type == "factor":
            agent = FactorAgent(model_dir=model_dir)
            factors = agent.load_factors()
            if factors and verbose:
                print(
                    f"Factor model loaded successfully, loaded {len(factors)} factors"
                )
            elif verbose:
                print("Factor model loading failed or no factors found")
            return agent if factors else None

        else:
            if verbose:
                print(f"Unsupported model type: {model_type}")
            return None

    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback

        traceback.print_exc()
        return None


def generate_signals(agent, model_type, prices_df, verbose=True):
    """Generate trading signals using model"""
    if verbose:
        print(f"Generating trading signals using {model_type} model...")

    try:
        if agent is None:
            if verbose:
                print(
                    f"Warning: {model_type} model not available, using neutral signal"
                )
            return {"signal": "neutral", "confidence": 0.5}

        # Add missing technical indicators
        prices_df = add_technical_indicators(prices_df.copy())

        # Initialize signal to default value, ensure all paths have values
        signals = {"signal": "neutral", "confidence": 0.5}

        # For factor model, special handling is needed
        if model_type == "factor":
            # Signal generation is handled internally by factor model, no need to pre-calculate features
            signals = agent.generate_signals(prices_df)
        elif model_type == "rl":
            window_size = 20  # RL model default window size
            if (
                len(prices_df) < window_size + 10
            ):  # Ensure at least window_size+10 data points
                if verbose:
                    print(
                        f"Warning: Insufficient data ({len(prices_df)} rows) to use RL model. Need at least {window_size + 10} rows."
                    )
                return {
                    "signal": "neutral",
                    "confidence": 0.5,
                    "error": "Insufficient data",
                }
            signals = agent.generate_signals(prices_df)
        else:
            # Detailed error handling
            try:
                signals = agent.generate_signals(prices_df)
            except Exception as e:
                import traceback

                print(f"Error generating signals: {e}")
                traceback.print_exc()
                # Use simple dictionary return
                signals = {"signal": "neutral", "confidence": 0.5, "error": str(e)}

        if verbose:
            print(
                f"Generated trading signal: {signals.get('signal', 'unknown')}, confidence: {signals.get('confidence', 0)}"
            )
            if "reasoning" in signals:
                print(f"Decision reasoning: {signals['reasoning']}")
            # Print complete signal dictionary for debugging
            print(f"Complete signal information: {signals}")

        return signals

    except Exception as e:
        print(f"Error generating trading signals: {e}")
        import traceback

        traceback.print_exc()
        return {"signal": "neutral", "confidence": 0.5}


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Stock trading model training and testing tool"
    )

    # Basic parameters
    parser.add_argument(
        "--ticker", type=str, required=True, help="Stock code, e.g.: 600519"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=(datetime.now() - timedelta(days=365 * 2)).strftime("%Y-%m-%d"),
        help="Start date, format: YYYY-MM-DD, default is two years ago",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=(datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
        help="End date, format: YYYY-MM-DD, default is yesterday",
    )

    # Model selection and operations
    parser.add_argument(
        "--model",
        type=str,
        choices=["dl", "rl", "factor", "all"],
        default="all",
        help="Model type: dl (deep learning), rl (reinforcement learning), factor (genetic programming factor), all (all)",
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["train", "test", "both", "evaluate"],
        default="both",
        help="Operation type: train (training only), test (testing only), both (train and test), evaluate (split data and evaluate)",
    )

    # Model parameters
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help='Model parameters, JSON format string, e.g.: \'{"hidden_dim": 128, "epochs": 100}\'',
    )

    # Data split parameters
    parser.add_argument(
        "--train-ratio", type=float, default=0.7, help="Training set ratio, default 0.7"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2, help="Validation set ratio, default 0.2"
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.1, help="Test set ratio, default 0.1"
    )
    parser.add_argument(
        "--shuffle", action="store_true", help="Whether to shuffle data"
    )

    # Other options
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help='Model save directory, default "models"',
    )
    parser.add_argument(
        "--factor-dir",
        type=str,
        default="factors",
        help='Factor model save directory, default "factors"',
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default="models/evaluation",
        help='Evaluation results save directory, default "models/evaluation"',
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True, help="Show detailed output"
    )

    return parser.parse_args()


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()

    # Parse JSON parameters
    params = None
    if args.params:
        try:
            params = json.loads(args.params)
            if args.verbose:
                print(f"Using custom parameters: {params}")
        except json.JSONDecodeError:
            print(f"Error: Unable to parse parameter JSON: {args.params}")
            print(
                'Please use correct JSON format, e.g.: \'{"hidden_dim": 128, "epochs": 100}\''
            )
            return

    # Get and process stock data
    prices_df = process_data(args.ticker, args.start_date, args.end_date, args.verbose)
    if prices_df is None or (isinstance(prices_df, pd.DataFrame) and prices_df.empty):
        print("Error: Unable to get stock data or data is empty")
        return

    # Data split evaluation mode
    if args.action == "evaluate":
        if args.verbose:
            print(f"\n{'='*50}")
            print(
                f"Starting data splitting and model evaluation - Stock code: {args.ticker}"
            )
            print(
                f"Data split ratio - Training set: {args.train_ratio*100:.1f}%, Validation set: {args.val_ratio*100:.1f}%, Test set: {args.test_ratio*100:.1f}%"
            )
            print(f"{'='*50}")

        # Execute model training and evaluation
        models_to_evaluate = (
            ["dl", "rl", "factor"] if args.model == "all" else [args.model]
        )

        evaluation_results = {}
        for model_type in models_to_evaluate:
            if args.verbose:
                print(f"\n{'='*50}")
                print(f"Starting evaluation of {model_type} model")
                print(f"{'='*50}")

            save_dir = args.factor_dir if model_type == "factor" else args.model_dir

            # Call split_evaluate module for training and evaluation
            agent, model_results = split_and_evaluate(
                ticker=args.ticker,
                price_data=prices_df,
                model_type=model_type,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                shuffle=args.shuffle,
                params=params,
                save_dir=save_dir,
                eval_dir=args.eval_dir,
                verbose=args.verbose,
            )

            evaluation_results[model_type] = model_results

            if args.verbose:
                print(f"\n{'='*50}")
                print(f"{model_type} model evaluation completed")
                print(f"{'='*50}")

        # Print comprehensive evaluation results
        if args.verbose and len(evaluation_results) > 0:
            print("\n")
            print("=" * 50)
            print("Evaluation Results Summary:")
            print("=" * 50)

            for model_type, results in evaluation_results.items():
                print(f"\n{model_type} model:")
                if "results" in results:
                    model_results = results["results"]
                    for sub_model, metrics in model_results.items():
                        print(f"  {sub_model}:")
                        if "metrics" in metrics:
                            for metric_name, metric_value in metrics["metrics"].items():
                                if isinstance(metric_value, (int, float)):
                                    print(f"    {metric_name}: {metric_value:.4f}")
                                else:
                                    print(f"    {metric_name}: {metric_value}")

                print(f"  Training set size: {results.get('train_size', 'N/A')}")
                print(f"  Test set size: {results.get('test_size', 'N/A')}")

            print(f"\nEvaluation results details saved to {args.eval_dir} directory")

        return

    # Execute model training and testing (original mode)
    models_to_train = ["dl", "rl", "factor"] if args.model == "all" else [args.model]
    results = {}

    for model_type in models_to_train:
        if args.verbose:
            print(f"\n{'='*50}")
            print(f"Starting processing of {model_type} model")
            print(f"{'='*50}")

        model_dir = args.factor_dir if model_type == "factor" else args.model_dir
        agent = None
        signals = None

        # Ensure directory exists
        os.makedirs(model_dir, exist_ok=True)

        # Train model
        if args.action in ["train", "both"]:
            if args.verbose:
                print(f"\n--- Training {model_type} model ---")

            if model_type == "dl":
                agent, signals = train_dl_model(
                    prices_df, params, model_dir, args.verbose
                )
            elif model_type == "rl":
                agent, signals = train_rl_model(
                    prices_df, params, model_dir, args.verbose
                )
            elif model_type == "factor":
                agent, signals = train_factor_model(
                    prices_df, params, model_dir, args.verbose
                )

        # Test model
        if args.action in ["test", "both"]:
            if args.verbose:
                print(f"\n--- Testing {model_type} model ---")

            # If only testing but not training, try to load existing model
            if args.action == "test":
                agent = load_model(model_type, model_dir, args.verbose)

            # Generate signals - only generate signals if trained or loaded
            if agent is not None:
                signals = generate_signals(agent, model_type, prices_df, args.verbose)
            else:
                # If model training or loading failed, provide neutral signal
                if args.verbose:
                    print(
                        f"Warning: Unable to train or load {model_type} model, using neutral signal"
                    )
                signals = {"signal": "neutral", "confidence": 0.5}

        # Save results
        if signals:
            results[model_type] = signals

    # Print comprehensive results
    if args.verbose and len(results) > 0:
        print("\n")
        print("=" * 50)
        print("Comprehensive Results:")
        print("=" * 50)

        for model_type, signals in results.items():
            signal = signals.get("signal", "unknown")
            confidence = signals.get("confidence", 0)
            print(f"{model_type:6} model: {signal:8} (confidence: {confidence:.2f})")

        if len(results) > 1:
            signal_counts = {"bullish": 0, "bearish": 0, "neutral": 0}
            weighted_sum = 0
            total_weight = 0

            for model_type, signals in results.items():
                signal = signals.get("signal", "neutral")
                confidence = signals.get("confidence", 0.5)

                signal_counts[signal] += 1

                # Use confidence as weight
                if signal == "bullish":
                    weighted_sum += confidence
                elif signal == "bearish":
                    weighted_sum -= confidence

                total_weight += confidence

            # Calculate comprehensive signal
            if total_weight > 0:
                normalized_score = weighted_sum / total_weight
            else:
                normalized_score = 0

            # Determine final signal
            if normalized_score > 0.2:
                final_signal = "bullish"
            elif normalized_score < -0.2:
                final_signal = "bearish"
            else:
                final_signal = "neutral"

            print("\nModel voting results:\n")
            print(
                f"Bullish: {signal_counts['bullish']}, Bearish: {signal_counts['bearish']}, Neutral: {signal_counts['neutral']}\n"
            )
            print(f"Weighted score: {normalized_score:.2f} (range: -1 to 1)\n")
            print(f"Comprehensive signal: {final_signal}\n")


if __name__ == "__main__":
    main()
