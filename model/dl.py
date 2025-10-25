import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.utils.logging_config import setup_logger

# Setup logging
logger = setup_logger("deep_learning")

# Default forecast days (can be overridden by function parameters)
DEFAULT_FORECAST_DAYS = 10

# Set random seeds to ensure reproducible results
torch.manual_seed(42)
np.random.seed(42)


class LSTMModel(nn.Module):
    """General LSTM model for loading saved models"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout: float = 0.2,
    ):
        """
        Initialize model
        Args:
            input_dim: Number of input features
            hidden_dim: LSTM hidden layer dimension
            num_layers: Number of LSTM layers
            output_dim: Output dimension (usually number of forecast days)
            dropout: Dropout probability
        """
        super(LSTMModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """Forward propagation"""
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # LSTM forward propagation
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Use the hidden state of the last time step
        out = self.fc(lstm_out[:, -1, :])
        return out


# For backward compatibility, keep original class name alias
StockLSTM = LSTMModel


class DeepLearningModule:
    """Deep learning management class, responsible for model training, evaluation and prediction"""

    def __init__(self, model_dir: str = "models"):
        """
        Initialize deep learning module

        Args:
            model_dir: Model save directory
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            f"Deep learning module initialization completed, using device: {self.device}"
        )

        # Save data processors and models
        self.price_scaler = None
        self.feature_scaler = None
        self.lstm_model = None
        self.rf_model = None

    def _create_lstm_sequences(
        self, data: np.ndarray, seq_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create LSTM input sequences

        Args:
            data: Input feature data
            seq_length: Sequence length (time window)

        Returns:
            X: Input sequences
            y: Target values (next time point)
        """
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i : i + seq_length])
            y.append(
                data[i + seq_length, 0]
            )  # Only predict first column (usually closing price)
        return np.array(X), np.array(y).reshape(-1, 1)

    def train_lstm_model(
        self,
        price_data: pd.DataFrame,
        target_col: str = "close",
        feature_cols: Optional[List[str]] = None,
        seq_length: int = 10,
        forecast_days: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 3,
        epochs: int = 50,
        batch_size: int = 16,
        learning_rate: float = 0.001,
    ):
        """
        Train LSTM model for price prediction

        Args:
            price_data: DataFrame containing price data
            target_col: Target column name (usually closing price)
            feature_cols: List of feature column names (uses target_col if None)
            seq_length: Input sequence length
            forecast_days: Number of forecast days (default 10)
            hidden_dim: LSTM hidden layer dimension (default 128)
            num_layers: Number of LSTM layers (default 3)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate

        Returns:
            Trained model
        """
        try:
            logger.info("Starting LSTM model training...")

            # If feature_cols is None, only use target_col
            if feature_cols is None:
                feature_cols = [target_col]

            # Log used feature columns
            logger.info(f"LSTM training using feature columns: {feature_cols}")

            # Ensure all features are in DataFrame
            missing_features = [
                col for col in feature_cols if col not in price_data.columns
            ]
            if missing_features:
                raise ValueError(
                    f"The following feature columns do not exist in data: {missing_features}"
                )

            # Extract feature data
            data = price_data[feature_cols].values

            # Data normalization
            self.price_scaler = MinMaxScaler()
            data_scaled = self.price_scaler.fit_transform(data)

            # Create sequence data
            X, y = self._create_lstm_sequences(data_scaled, seq_length)

            # Split training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )

            # Convert to PyTorch tensors
            X_train = torch.FloatTensor(X_train)
            y_train = torch.FloatTensor(y_train)
            X_val = torch.FloatTensor(X_val)
            y_val = torch.FloatTensor(y_val)

            # Create data loader
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )

            # Initialize model
            input_dim = len(feature_cols)
            logger.info(
                f"Initializing LSTM model, input dimension: {input_dim}, hidden dimension: {hidden_dim}, layers: {num_layers}"
            )
            self.lstm_model = LSTMModel(
                input_dim, hidden_dim, num_layers, forecast_days
            )
            self.lstm_model.to(self.device)

            # Define optimizer and loss function
            optimizer = optim.Adam(self.lstm_model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()

            # Train model
            self.lstm_model.train()
            for epoch in range(epochs):
                total_loss = 0
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                    # Forward propagation
                    optimizer.zero_grad()
                    outputs = self.lstm_model(batch_X)

                    # Calculate loss
                    # Compare first prediction value with actual value
                    loss = criterion(outputs[:, 0].unsqueeze(1), batch_y)

                    # Backward propagation
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                # Validation
                if epoch % 5 == 0:
                    self.lstm_model.eval()
                    with torch.no_grad():
                        val_outputs = self.lstm_model(X_val.to(self.device))
                        val_loss = criterion(
                            val_outputs[:, 0].unsqueeze(1), y_val.to(self.device)
                        )

                    logger.info(
                        f"Epoch {epoch+1}/{epochs}, Training Loss: {total_loss/len(train_loader):.4f}, Validation Loss: {val_loss.item():.4f}"
                    )
                    self.lstm_model.train()

            # Save model
            torch.save(self.lstm_model.state_dict(), f"{self.model_dir}/lstm_model.pth")
            joblib.dump(self.price_scaler, f"{self.model_dir}/price_scaler.pkl")

            logger.info("LSTM model training completed")
            return self.lstm_model

        except Exception as e:
            logger.error(f"Error training LSTM model: {str(e)}")
            raise

    def predict_lstm(
        self,
        price_data: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        seq_length: int = 10,
        target_col: str = "close",
    ) -> np.ndarray:
        """
        Use LSTM model to predict future prices

        Args:
            price_data: DataFrame containing historical price data
            feature_cols: List of feature column names
            seq_length: Input sequence length
            target_col: Target column name

        Returns:
            Future price prediction results
        """
        if self.lstm_model is None:
            raise ValueError(
                "LSTM model not trained, please call train_lstm_model method first"
            )

        try:
            # If feature_cols is None, get feature count from price_scaler
            if feature_cols is None:
                # Determine feature count in price_scaler
                n_features = self.price_scaler.n_features_in_
                logger.info(f"Detected feature count from price_scaler: {n_features}")

                # Determine feature columns to use based on feature count
                if n_features == 1:
                    feature_cols = [target_col]
                elif n_features == 6:
                    feature_cols = ["close", "ma5", "ma10", "ma20", "rsi", "macd"]
                else:
                    # If unsure which features to use, try selecting enough features from DataFrame
                    all_numeric_cols = price_data.select_dtypes(
                        include=np.number
                    ).columns.tolist()
                    feature_cols = all_numeric_cols[:n_features]
                    logger.info(f"Auto-selected feature columns: {feature_cols}")

            # Ensure all features exist
            missing_features = [
                col for col in feature_cols if col not in price_data.columns
            ]
            if missing_features:
                logger.warning(
                    f"The following features are missing from data: {missing_features}"
                )
                # Try to generate missing features
                for col in missing_features:
                    if col == "ma5":
                        price_data["ma5"] = price_data["close"].rolling(window=5).mean()
                    elif col == "ma10":
                        price_data["ma10"] = (
                            price_data["close"].rolling(window=10).mean()
                        )
                    elif col == "ma20":
                        price_data["ma20"] = (
                            price_data["close"].rolling(window=20).mean()
                        )
                    elif col == "rsi":
                        delta = price_data["close"].diff()
                        gain = (delta.where(delta > 0, 0)).fillna(0)
                        loss = (-delta.where(delta < 0, 0)).fillna(0)
                        avg_gain = gain.rolling(window=14).mean()
                        avg_loss = loss.rolling(window=14).mean()
                        rs = avg_gain / avg_loss
                        price_data["rsi"] = 100 - (100 / (1 + rs))
                    elif col == "macd":
                        ema12 = price_data["close"].ewm(span=12, adjust=False).mean()
                        ema26 = price_data["close"].ewm(span=26, adjust=False).mean()
                        price_data["macd"] = ema12 - ema26

            # Fill NaN values
            price_data = price_data.ffill().bfill()

            # Extract latest sequence data
            data = price_data[feature_cols].values[-seq_length:]

            # Data normalization
            data_scaled = self.price_scaler.transform(data)

            # Convert to model input format
            X = torch.FloatTensor(data_scaled).unsqueeze(0).to(self.device)

            # Prediction
            self.lstm_model.eval()
            with torch.no_grad():
                predictions = self.lstm_model(X)

            # Convert back to original scale
            # Create zero array with same feature count as original
            zeros = np.zeros((predictions.shape[1], len(feature_cols)))
            # Put prediction values in first column (closing price column)
            zeros[:, 0] = predictions.cpu().numpy()[0]
            predictions_rescaled = self.price_scaler.inverse_transform(zeros)

            # Only return prediction results for closing price column
            return predictions_rescaled[:, 0]

        except Exception as e:
            logger.error(f"Error in LSTM prediction: {str(e)}")
            raise

    def train_stock_classifier(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        n_estimators: int = 200,
        random_state: int = 42,
    ):
        """
        Train stock classifier for stock selection

        Args:
            features: Feature DataFrame
            labels: Label Series (1 for up, 0 for down)
            n_estimators: Number of decision trees
            random_state: Random seed

        Returns:
            Trained model
        """
        try:
            logger.info("Starting random forest classifier training...")

            # Check class distribution
            class_distribution = labels.value_counts(normalize=True)
            logger.info(f"Original class distribution: {class_distribution.to_dict()}")

            # Determine if class balancing is needed
            class_weight = None
            if len(class_distribution) > 1:
                ratio_0 = class_distribution.get(0, 0)
                ratio_1 = class_distribution.get(1, 0)

                if ratio_0 > 0.65 or ratio_1 > 0.65:
                    logger.info(
                        "Detected severe class imbalance, will apply balancing weights"
                    )
                    # Custom class weights to ensure model doesn't always predict majority class
                    class_weight = {
                        0: 1.0 / (ratio_0 if ratio_0 > 0 else 0.5),
                        1: 1.0 / (ratio_1 if ratio_1 > 0 else 0.5),
                    }
                    logger.info(f"Applied class weights: {class_weight}")

            # Data standardization
            self.feature_scaler = StandardScaler()
            features_scaled = self.feature_scaler.fit_transform(features)

            # Split training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                features_scaled,
                labels,
                test_size=0.2,
                random_state=random_state,
                stratify=labels,
            )

            # Train random forest classifier
            self.rf_model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1,
                max_depth=6,  # Limit tree depth to reduce overfitting
                min_samples_leaf=5,  # Ensure leaf nodes have enough samples
                class_weight=class_weight,
                max_features="sqrt",  # Use feature subset to increase diversity
            )
            self.rf_model.fit(X_train, y_train)

            # Validation
            train_accuracy = self.rf_model.score(X_train, y_train)
            val_accuracy = self.rf_model.score(X_val, y_val)

            # Class prediction ratio on validation set
            y_val_pred = self.rf_model.predict(X_val)
            val_pred_distribution = pd.Series(y_val_pred).value_counts(normalize=True)
            logger.info(
                f"Validation set prediction class distribution: {val_pred_distribution.to_dict()}"
            )

            logger.info(
                f"Classifier training completed, training accuracy: {train_accuracy:.4f}, validation accuracy: {val_accuracy:.4f}"
            )

            # Feature importance
            feature_importance = {
                features.columns[i]: importance
                for i, importance in enumerate(self.rf_model.feature_importances_)
            }
            logger.info(
                f"Feature importance: {sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]}"
            )

            # Save model
            joblib.dump(self.rf_model, f"{self.model_dir}/rf_classifier.pkl")
            joblib.dump(self.feature_scaler, f"{self.model_dir}/feature_scaler.pkl")

            return self.rf_model

        except Exception as e:
            logger.error(f"Error training stock classifier: {str(e)}")
            raise

    def predict_stock_returns(self, features: pd.DataFrame) -> Dict[str, float]:
        """
        Predict stock classification results and probabilities

        Args:
            features: Feature DataFrame

        Returns:
            Dictionary containing prediction classification and probabilities
        """
        if self.rf_model is None:
            raise ValueError(
                "Random forest model not trained, please call train_stock_classifier method first"
            )

        try:
            # Data standardization
            features_scaled = self.feature_scaler.transform(features)

            # Predict class and probabilities
            predictions = self.rf_model.predict(features_scaled)
            probabilities = self.rf_model.predict_proba(features_scaled)

            # Get original probabilities
            raw_probability_up = probabilities[0][1]  # Up probability
            raw_probability_down = probabilities[0][0]  # Down probability

            # Adjust extreme probabilities to prevent overconfident predictions
            calibrated_prob_up = min(max(raw_probability_up, 0.05), 0.95)
            calibrated_prob_down = min(max(raw_probability_down, 0.05), 0.95)

            # Renormalize probabilities
            total = calibrated_prob_up + calibrated_prob_down
            calibrated_prob_up = calibrated_prob_up / total
            calibrated_prob_down = calibrated_prob_down / total

            # Log original and calibrated probabilities
            logger.info(
                f"Original prediction probabilities - Up: {raw_probability_up:.4f}, Down: {raw_probability_down:.4f}"
            )
            logger.info(
                f"Calibrated probabilities - Up: {calibrated_prob_up:.4f}, Down: {calibrated_prob_down:.4f}"
            )

            # Return prediction results
            return {
                "prediction": predictions[0],  # 1 for up, 0 for down
                "probability": (
                    calibrated_prob_up if predictions[0] == 1 else calibrated_prob_down
                ),
                "up_probability": calibrated_prob_up,
                "down_probability": calibrated_prob_down,
                "expected_return": (calibrated_prob_up - 0.5)
                * 2,  # Convert to -1 to 1 range
            }

        except Exception as e:
            logger.error(f"Error predicting stock classification: {str(e)}")
            raise

    def load_models(
        self,
        lstm_path: Optional[str] = None,
        rf_path: Optional[str] = None,
        price_scaler_path: Optional[str] = None,
        feature_scaler_path: Optional[str] = None,
    ):
        """
        Load trained models

        Args:
            lstm_path: LSTM model path
            rf_path: Random forest model path
            price_scaler_path: Price scaler path
            feature_scaler_path: Feature scaler path
        """
        try:
            # Set default paths
            if lstm_path is None:
                lstm_path = f"{self.model_dir}/lstm_model.pth"
            if rf_path is None:
                rf_path = f"{self.model_dir}/rf_classifier.pkl"
            if price_scaler_path is None:
                price_scaler_path = f"{self.model_dir}/price_scaler.pkl"
            if feature_scaler_path is None:
                feature_scaler_path = f"{self.model_dir}/feature_scaler.pkl"

            # Load LSTM model
            if os.path.exists(lstm_path) and os.path.exists(price_scaler_path):
                # First load scaler to get feature dimensions
                self.price_scaler = joblib.load(price_scaler_path)
                input_dim = getattr(self.price_scaler, "n_features_in_", 1)

                logger.info(
                    f"Feature dimension detected from price_scaler: {input_dim}"
                )

                # Need to define model structure first, using correct input dimension
                self.lstm_model = LSTMModel(
                    input_dim=input_dim, hidden_dim=128, num_layers=3, output_dim=10
                )

                # Add code to try loading the model
                try:
                    state_dict = torch.load(lstm_path, map_location=self.device)
                    logger.info(
                        f"Starting to try loading model parameters, detected the following parameters: {list(state_dict.keys())}"
                    )

                    # Manually create a new model with correct structure
                    # First determine key parameters of the model
                    # Check shape of first layer parameters to determine actual hidden layer size
                    if "lstm.weight_ih_l0" in state_dict:
                        # LSTM parameter shape is [4*hidden_size, input_dim]
                        lstm_param_shape = state_dict["lstm.weight_ih_l0"].shape
                        actual_hidden_size = lstm_param_shape[0] // 4
                        actual_input_dim = lstm_param_shape[1]
                        logger.info(
                            f"Inferred from model parameters: hidden_size={actual_hidden_size}, input_dim={actual_input_dim}"
                        )

                        # Check output layer size
                        if "fc.weight" in state_dict:
                            fc_param_shape = state_dict["fc.weight"].shape
                            actual_output_dim = fc_param_shape[0]
                            logger.info(
                                f"Inferred from model parameters: output_dim={actual_output_dim}"
                            )
                        else:
                            actual_output_dim = 10  # Default value

                        # Check number of layers
                        actual_num_layers = 1
                        while f"lstm.weight_ih_l{actual_num_layers}" in state_dict:
                            actual_num_layers += 1
                        logger.info(
                            f"Inferred from model parameters: LSTM layers={actual_num_layers}"
                        )

                        # Recreate model with correct parameters
                        logger.info(
                            f"Recreating model with inferred parameters: input_dim={actual_input_dim}, hidden_dim={actual_hidden_size}, num_layers={actual_num_layers}, output_dim={actual_output_dim}"
                        )
                        self.lstm_model = LSTMModel(
                            input_dim=actual_input_dim,
                            hidden_dim=actual_hidden_size,
                            num_layers=actual_num_layers,
                            output_dim=actual_output_dim,
                        )

                        # Try loading the state dict that should now match
                        self.lstm_model.load_state_dict(state_dict)
                        logger.info(
                            "✓ LSTM model loaded successfully, using model structure inferred from parameters"
                        )
                    else:
                        # If key parameters cannot be found, retry with default values
                        logger.warning(
                            "Cannot infer model structure from parameters, using default parameters"
                        )
                        self.lstm_model = LSTMModel(
                            input_dim=input_dim,
                            hidden_dim=128,
                            num_layers=3,
                            output_dim=10,
                        )
                        self.lstm_model.load_state_dict(state_dict)
                        logger.info("Model loaded successfully with default parameters")
                except Exception as e:
                    logger.error(f"Final model loading failure: {str(e)}")
                    logger.warning(
                        "Will use default model initialization, but will not load pretrained weights"
                    )
                    self.lstm_model = LSTMModel(
                        input_dim=input_dim, hidden_dim=128, num_layers=3, output_dim=10
                    )

                self.lstm_model.to(self.device)
                self.lstm_model.eval()

                logger.info("LSTM model loading process completed")
            else:
                logger.warning("LSTM model or scaler file does not exist")

            # Load random forest model
            if os.path.exists(rf_path) and os.path.exists(feature_scaler_path):
                self.rf_model = joblib.load(rf_path)
                self.feature_scaler = joblib.load(feature_scaler_path)
                logger.info("Random forest model loaded successfully")
            else:
                logger.warning("Random forest model or scaler file does not exist")

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    def prepare_features(
        self,
        price_data: pd.DataFrame,
        technical_indicators: Dict[str, pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Prepare model features

        Args:
            price_data: Price data DataFrame
            technical_indicators: Technical indicators dictionary

        Returns:
            Feature DataFrame
        """
        # Create feature DataFrame
        features = pd.DataFrame()

        # Add price features
        if "close" in price_data.columns:
            # Price change rate (1 day, 3 days, 5 days)
            features["price_change_1d"] = price_data["close"].pct_change(1)
            features["price_change_3d"] = price_data["close"].pct_change(3)
            features["price_change_5d"] = price_data["close"].pct_change(5)

            # Volatility (10 days, 20 days)
            features["volatility_10d"] = (
                price_data["close"].pct_change().rolling(10).std()
            )
            features["volatility_20d"] = (
                price_data["close"].pct_change().rolling(20).std()
            )

        # Add volume-price features
        if "volume" in price_data.columns:
            # Volume change rate
            features["volume_change_1d"] = price_data["volume"].pct_change(1)
            features["volume_change_5d"] = price_data["volume"].pct_change(5)

            # Price-volume relationship
            if "close" in price_data.columns:
                features["price_volume_corr"] = (
                    price_data["close"].rolling(10).corr(price_data["volume"])
                )

        # Add technical indicators
        if technical_indicators:
            for name, indicator in technical_indicators.items():
                features[name] = indicator

        # Handle missing values
        features = features.dropna()

        return features

    def generate_training_labels(
        self,
        price_data: pd.DataFrame,
        forward_days: int = 5,
        return_threshold: float = 0.01,
    ) -> pd.Series:
        """
        Generate training labels

        Args:
            price_data: Price data DataFrame
            forward_days: Future days
            return_threshold: Return threshold

        Returns:
            Label Series (1 for up exceeding threshold, 0 for down below negative threshold)
        """
        # Calculate future returns
        future_returns = (
            price_data["close"].shift(-forward_days) / price_data["close"] - 1
        )

        # Generate labels (1 for up exceeding threshold, 0 for down below negative threshold)
        # Use symmetric threshold to make up and down opportunities more balanced
        labels = (future_returns > return_threshold).astype(int)

        # Count label distribution and output logs
        label_counts = labels.value_counts()
        up_ratio = label_counts.get(1, 0) / len(labels) if len(labels) > 0 else 0
        down_ratio = label_counts.get(0, 0) / len(labels) if len(labels) > 0 else 0
        logger.info(f"Label distribution - Up: {up_ratio:.2%}, Down: {down_ratio:.2%}")

        return labels


# Model wrapper for integration into Agent system
class MLAgent:
    """Machine Learning Agent, responsible for providing prediction signals to Portfolio Manager"""

    def __init__(self, model_dir: str = "models"):
        """Initialize ML Agent"""
        self.dl_module = DeepLearningModule(model_dir)
        self.is_trained = False
        self.logger = setup_logger("ml_agent")

    def train_models(
        self,
        price_data: pd.DataFrame,
        technical_indicators: Dict[str, pd.Series] = None,
        seq_length: int = 10,
        feature_cols: List[str] = None,
        hidden_dim: int = 128,
        num_layers: int = 3,
        forecast_days: int = 10,
        epochs: int = 50,
        batch_size: int = 16,
        learning_rate: float = 0.001,
    ):
        """
        Train all models

        Args:
            price_data: Price data
            technical_indicators: Technical indicators
            seq_length: Sequence length
            feature_cols: Feature column name list
            hidden_dim: LSTM hidden layer dimension (default 128)
            num_layers: LSTM layers (default 3)
            forecast_days: Forecast days (default 10)
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        try:
            # If feature_cols is None, use default values
            if feature_cols is None:
                feature_cols = ["close", "ma5", "ma10", "ma20", "rsi", "macd"]
                self.logger.info(
                    f"No feature columns specified, using default features: {feature_cols}"
                )

            # Ensure feature columns exist in data, otherwise add them
            missing_features = [
                col for col in feature_cols if col not in price_data.columns
            ]
            if missing_features:
                self.logger.warning(
                    f"Data missing the following features, will add automatically: {missing_features}"
                )
                for col in missing_features:
                    if col == "ma5":
                        price_data["ma5"] = price_data["close"].rolling(window=5).mean()
                    elif col == "ma10":
                        price_data["ma10"] = (
                            price_data["close"].rolling(window=10).mean()
                        )
                    elif col == "ma20":
                        price_data["ma20"] = (
                            price_data["close"].rolling(window=20).mean()
                        )
                    elif col == "rsi":
                        delta = price_data["close"].diff()
                        gain = (delta.where(delta > 0, 0)).fillna(0)
                        loss = (-delta.where(delta < 0, 0)).fillna(0)
                        avg_gain = gain.rolling(window=14).mean()
                        avg_loss = loss.rolling(window=14).mean()
                        rs = avg_gain / avg_loss
                        price_data["rsi"] = 100 - (100 / (1 + rs))
                    elif col == "macd":
                        ema12 = price_data["close"].ewm(span=12, adjust=False).mean()
                        ema26 = price_data["close"].ewm(span=26, adjust=False).mean()
                        price_data["macd"] = ema12 - ema26

            # Fill NaN values
            price_data = price_data.ffill().bfill()

            self.logger.info(
                f"Starting model training with parameters: seq_length={seq_length}, hidden_dim={hidden_dim}, "
                f"num_layers={num_layers}, epochs={epochs}, forecast_days={forecast_days}"
            )
            self.logger.info(f"Using feature columns: {feature_cols}")

            # Prepare LSTM training data using passed parameters
            self.dl_module.train_lstm_model(
                price_data=price_data,
                target_col="close",
                feature_cols=feature_cols,
                seq_length=seq_length,
                forecast_days=forecast_days,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )

            # Prepare classifier training data
            features = self.dl_module.prepare_features(price_data, technical_indicators)
            labels = self.dl_module.generate_training_labels(
                price_data, forward_days=forecast_days
            )

            # Ensure features and labels have the same index
            common_idx = features.index.intersection(labels.index)
            if len(common_idx) > 0:
                self.logger.info(
                    f"Training random forest classifier with {len(common_idx)} samples"
                )
                self.dl_module.train_stock_classifier(
                    features=features.loc[common_idx],
                    labels=labels.loc[common_idx],
                    n_estimators=200,
                )
                self.is_trained = True
            else:
                self.logger.warning(
                    "Features and labels have no common index, cannot train classifier"
                )

        except Exception as e:
            self.logger.error(f"Error training models: {str(e)}")
            import traceback

            traceback.print_exc()

    def load_models(self):
        """Load trained models"""
        try:
            self.dl_module.load_models()
            self.is_trained = True
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")

    def generate_signals(
        self,
        price_data: pd.DataFrame,
        technical_indicators: Dict[str, pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Generate trading signals

        Args:
            price_data: Price data
            technical_indicators: Technical indicators

        Returns:
            Dictionary containing trading signals
        """
        signals = {}

        try:
            # Load models (if not already loaded)
            if not self.is_trained:
                self.load_models()

            # Use LSTM to predict future prices
            lstm_predictions = None
            if self.dl_module.lstm_model is not None:
                # Determine forecast days
                forecast_days = self.dl_module.lstm_model.fc.out_features
                self.logger.info(
                    f"Using LSTM model to predict future {forecast_days} days prices"
                )

                # Get feature count from price_scaler to determine correct feature_cols
                n_features = self.dl_module.price_scaler.n_features_in_
                if n_features == 1:
                    lstm_feature_cols = ["close"]
                elif n_features == 6:
                    lstm_feature_cols = ["close", "ma5", "ma10", "ma20", "rsi", "macd"]
                else:
                    # Try to select appropriate features based on data
                    all_numeric_cols = price_data.select_dtypes(
                        include=np.number
                    ).columns.tolist()
                    lstm_feature_cols = all_numeric_cols[:n_features]

                self.logger.info(
                    f"LSTM prediction will use the following features: {lstm_feature_cols}"
                )

                lstm_predictions = self.dl_module.predict_lstm(
                    price_data=price_data,
                    feature_cols=lstm_feature_cols,
                    seq_length=10,
                    target_col="close",
                )

                # Calculate expected returns
                current_price = price_data["close"].iloc[-1]
                future_prices = lstm_predictions
                expected_returns = [
                    price / current_price - 1 for price in future_prices
                ]

                signals["lstm_predictions"] = {
                    "future_prices": future_prices.tolist(),
                    "expected_returns": expected_returns,
                    "forecast_days": forecast_days,
                }

                # Calculate statistics for predicted growth rates
                avg_return = np.mean(expected_returns)
                median_return = np.median(expected_returns)
                positive_count = sum(1 for r in expected_returns if r > 0)
                negative_count = sum(1 for r in expected_returns if r < 0)

                self.logger.info(
                    f"LSTM prediction - Average return: {avg_return:.4f}, Median return: {median_return:.4f}"
                )
                self.logger.info(
                    f"LSTM prediction - Positive return days: {positive_count}, Negative return days: {negative_count}"
                )

                # Generate signals based on LSTM predictions, adjust thresholds and confidence calculation
                positive_ratio = (
                    positive_count / len(expected_returns)
                    if len(expected_returns) > 0
                    else 0.5
                )

                # Use more balanced prediction logic
                if (
                    avg_return > 0.02 and positive_ratio > 0.6
                ):  # Positive signal threshold
                    signals["lstm_signal"] = "bullish"
                    signals["lstm_confidence"] = min(
                        0.5 + positive_ratio * 0.4, 0.9
                    )  # More moderate confidence
                elif (
                    avg_return < -0.02 and positive_ratio < 0.4
                ):  # Negative signal threshold
                    signals["lstm_signal"] = "bearish"
                    signals["lstm_confidence"] = min(
                        0.5 + (1 - positive_ratio) * 0.4, 0.9
                    )
                else:
                    signals["lstm_signal"] = "neutral"
                    signals["lstm_confidence"] = 0.5

            # Use classifier to predict up/down probabilities
            rf_prediction = None
            if self.dl_module.rf_model is not None:
                # Prepare features
                features = self.dl_module.prepare_features(
                    price_data, technical_indicators
                )
                if not features.empty:
                    # Get latest feature row
                    latest_features = features.iloc[[-1]]
                    rf_prediction = self.dl_module.predict_stock_returns(
                        latest_features
                    )

                    signals["rf_prediction"] = rf_prediction

                    # Get up and down probabilities
                    up_probability = rf_prediction.get("up_probability", 0.5)
                    down_probability = rf_prediction.get("down_probability", 0.5)

                    # Generate signals based on random forest predictions, apply more conservative thresholds
                    if rf_prediction["prediction"] == 1:
                        signals["rf_signal"] = "bullish"
                        signals["rf_confidence"] = rf_prediction.get(
                            "probability", up_probability
                        )
                    else:
                        signals["rf_signal"] = "bearish"
                        signals["rf_confidence"] = rf_prediction.get(
                            "probability", down_probability
                        )

                    # Log prediction results
                    self.logger.info(
                        f"RF prediction - Signal: {signals['rf_signal']}, Confidence: {signals['rf_confidence']:.4f}"
                    )

            # Combine both models to generate final signal
            if "lstm_signal" in signals and "rf_signal" in signals:
                # Simple weighted average, but adjust weights to reduce bias
                lstm_weight = 0.5  # LSTM and random forest weights are equal
                rf_weight = 0.5

                lstm_score = {"bullish": 1, "neutral": 0, "bearish": -1}[
                    signals["lstm_signal"]
                ] * signals["lstm_confidence"]
                rf_score = {"bullish": 1, "neutral": 0, "bearish": -1}[
                    signals["rf_signal"]
                ] * signals["rf_confidence"]

                combined_score = lstm_weight * lstm_score + rf_weight * rf_score

                # Use more conservative thresholds
                if combined_score > 0.3:
                    signals["signal"] = "bullish"
                    signals["confidence"] = min(abs(combined_score), 0.85)
                elif combined_score < -0.3:
                    signals["signal"] = "bearish"
                    signals["confidence"] = min(abs(combined_score), 0.85)
                else:
                    signals["signal"] = "neutral"
                    signals["confidence"] = 0.5

                self.logger.info(
                    f"Final signal - {signals['signal']}, Confidence: {signals['confidence']:.4f}, Combined score: {combined_score:.4f}"
                )
            elif "lstm_signal" in signals:
                signals["signal"] = signals["lstm_signal"]
                signals["confidence"] = signals["lstm_confidence"]
            elif "rf_signal" in signals:
                signals["signal"] = signals["rf_signal"]
                signals["confidence"] = signals["rf_confidence"]
            else:
                signals["signal"] = "neutral"
                signals["confidence"] = 0.5

            # Add processing logic
            signals["reasoning"] = self._generate_reasoning(signals)

        except Exception as e:
            self.logger.error(f"Error generating trading signals: {str(e)}")
            signals["signal"] = "neutral"
            signals["confidence"] = 0.5
            signals["error"] = str(e)

        return signals

    def _generate_reasoning(self, signals: Dict[str, Any]) -> str:
        """
        Generate decision reasoning based on signals

        Args:
            signals: Signal dictionary

        Returns:
            Decision reasoning
        """
        reasoning_parts = []

        # LSTM model reasoning
        if "lstm_predictions" in signals:
            future_prices = signals["lstm_predictions"]["future_prices"]
            expected_returns = signals["lstm_predictions"]["expected_returns"]
            forecast_days = signals["lstm_predictions"].get(
                "forecast_days", DEFAULT_FORECAST_DAYS
            )

            avg_return = np.mean(expected_returns)
            positive_count = sum(1 for r in expected_returns if r > 0)
            negative_count = sum(1 for r in expected_returns if r < 0)

            positive_ratio = (
                positive_count / len(expected_returns)
                if len(expected_returns) > 0
                else 0
            )

            short_term_return = expected_returns[0] if expected_returns else 0
            medium_term_avg = (
                np.mean(expected_returns[1 : min(5, len(expected_returns))])
                if len(expected_returns) > 1
                else avg_return
            )
            long_term_avg = (
                np.mean(expected_returns[min(5, len(expected_returns)) :])
                if len(expected_returns) > 5
                else avg_return
            )

            reasoning_parts.append(
                f"LSTM model predicts next {forecast_days} days: Short-term return ({short_term_return:.2%}), Medium-term return ({medium_term_avg:.2%}), Long-term return ({long_term_avg:.2%}). "
                f"Expected positive return days ratio: {positive_ratio:.2%}"
            )

        # Random forest model reasoning
        if "rf_prediction" in signals:
            pred = signals["rf_prediction"]["prediction"]
            up_prob = signals["rf_prediction"].get("up_probability", 0.5)
            down_prob = signals["rf_prediction"].get("down_probability", 0.5)

            if pred == 1:
                reasoning_parts.append(
                    f"Random forest model predicts up probability: {up_prob:.2%}, down probability: {down_prob:.2%}"
                )
            else:
                reasoning_parts.append(
                    f"Random forest model predicts down probability: {down_prob:.2%}, up probability: {up_prob:.2%}"
                )

        # Technical indicator evaluation
        if "lstm_signal" in signals:
            reasoning_parts.append(
                f"LSTM technical analysis result: {signals['lstm_signal']}, confidence: {signals['lstm_confidence']:.2%}"
            )

        if "rf_signal" in signals:
            reasoning_parts.append(
                f"Random forest technical analysis result: {signals['rf_signal']}, confidence: {signals['rf_confidence']:.2%}"
            )

        # Combined strategy reasoning
        if "signal" in signals:
            if signals["signal"] == "bullish":
                reasoning_parts.append(
                    f"Comprehensive analysis generates bullish signal, confidence: {signals['confidence']:.2%}. "
                    f"Please note that even with bullish signals, market risks still exist, recommend setting stop loss."
                )
            elif signals["signal"] == "bearish":
                reasoning_parts.append(
                    f"Comprehensive analysis generates bearish signal, confidence: {signals['confidence']:.2%}. "
                    f"Please note that even with bearish signals, market may still rebound, recommend timely profit taking."
                )
            else:
                reasoning_parts.append(
                    f"Comprehensive analysis generates neutral signal, confidence: {signals['confidence']:.2%}. "
                    f"Market direction is unclear, recommend cautious operation or holding cash and waiting."
                )

        return "; ".join(reasoning_parts)


def preprocess_stock_data(
    price_df: pd.DataFrame, technical_indicators: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Preprocess stock data for deep learning models

    Args:
        price_df: Price data DataFrame
        technical_indicators: Technical indicators dictionary

    Returns:
        Processed DataFrame
    """
    # Ensure data is DataFrame
    if not isinstance(price_df, pd.DataFrame):
        raise ValueError("Price data must be pandas DataFrame")

    # Ensure necessary columns exist
    required_cols = ["close"]
    if not all(col in price_df.columns for col in required_cols):
        raise ValueError(
            f"Price data must contain the following columns: {required_cols}"
        )

    # Copy data to avoid modifying original data
    df = price_df.copy()

    df.set_index("date", inplace=True)

    # Calculate returns
    df["returns"] = df["close"].pct_change()

    # Calculate technical indicators
    # Moving averages
    df["ma5"] = df["close"].rolling(window=5).mean()
    df["ma10"] = df["close"].rolling(window=10).mean()
    df["ma20"] = df["close"].rolling(window=20).mean()

    # MACD
    df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema12"] - df["ema26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # Add external technical indicators
    if technical_indicators:
        for name, indicator in technical_indicators.items():
            if isinstance(indicator, pd.Series):
                df[name] = indicator

    # Add lag features
    for lag in [1, 2, 3, 5, 10]:
        df[f"close_lag_{lag}"] = df["close"].shift(lag)
        df[f"returns_lag_{lag}"] = df["returns"].shift(lag)

    # Add rolling statistics
    for window in [5, 10, 20]:
        df[f"volatility_{window}d"] = df["returns"].rolling(window=window).std()
        df[f"max_{window}d"] = df["close"].rolling(window=window).max()
        df[f"min_{window}d"] = df["close"].rolling(window=window).min()

    # Remove missing values
    df = df.dropna()

    return df
