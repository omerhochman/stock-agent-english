import os
import random
from typing import Any, Dict, Tuple

import gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gym import spaces
from torch.distributions import Categorical

from src.utils.logging_config import setup_logger

# Setup logging
logger = setup_logger("reinforcement_learning")

# Set random seeds to ensure reproducible results
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Explicitly set CUDA device usage
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
# Ensure GPU is used when CUDA is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA device count: {torch.cuda.device_count()}")


class StockTradingEnv(gym.Env):
    """Stock trading environment, implements OpenAI Gym interface"""

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance=100000,
        transaction_fee_percent=0.001,
        tech_indicators=None,
        reward_scaling=1.0,
        max_steps=252,
        window_size=20,
    ):
        """
        Initialize trading environment

        Args:
            df: Price data DataFrame
            initial_balance: Initial capital
            transaction_fee_percent: Transaction fee percentage
            tech_indicators: Technical indicators list
            reward_scaling: Reward scaling factor
            max_steps: Maximum steps per episode
            window_size: Observation window size
        """
        super(StockTradingEnv, self).__init__()

        self.df = df.copy()
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        self.reward_scaling = reward_scaling
        self.max_steps = max_steps
        self.window_size = window_size

        # Technical indicators list, use default indicators if None
        if tech_indicators is None:
            self.tech_indicator_columns = [
                "ma5",
                "ma10",
                "ma20",
                "rsi",
                "macd",
                "macd_signal",
                "macd_hist",
                "volatility_5d",
                "volatility_10d",
                "volatility_20d",
            ]
        else:
            self.tech_indicator_columns = tech_indicators

        # Ensure all technical indicators are in DataFrame
        missing_columns = [
            col for col in self.tech_indicator_columns if col not in self.df.columns
        ]
        if missing_columns:
            logger.warning(
                f"The following technical indicators do not exist in DataFrame: {missing_columns}"
            )
            # Remove non-existent indicators from list
            self.tech_indicator_columns = [
                col for col in self.tech_indicator_columns if col in self.df.columns
            ]

        # State space: price, technical indicators, position, balance, etc.
        # Feature count = window size * (price features + technical indicators) + position and balance info
        num_price_features = 5  # OHLCV: open, high, low, close, volume
        num_tech_indicators = len(self.tech_indicator_columns)
        observation_shape = (
            self.window_size * (num_price_features + num_tech_indicators) + 2,
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=observation_shape, dtype=np.float32
        )

        # Action space: buy, sell, hold (0: hold, 1: buy, 2: sell)
        self.action_space = spaces.Discrete(3)

        # Current step and position
        self.current_step = None
        self.current_position = 0
        self.total_reward = 0
        self.history = []

        # Initialize environment
        self.reset()

    def _next_observation(self) -> np.ndarray:
        """
        Get current window state features

        Returns:
            State feature array
        """
        # Ensure within data range
        end_idx = min(self.current_step + 1, len(self.df))
        start_idx = max(0, end_idx - self.window_size)

        # Fill insufficient window size parts with first data
        padding_size = self.window_size - (end_idx - start_idx)

        # Extract price data
        price_features = ["open", "high", "low", "close", "volume"]
        prices = self.df[price_features].iloc[start_idx:end_idx].values

        # Extract technical indicators
        tech_indicators = (
            self.df[self.tech_indicator_columns].iloc[start_idx:end_idx].values
        )

        # Fill insufficient parts
        if padding_size > 0:
            price_padding = np.repeat(prices[:1], padding_size, axis=0)
            tech_padding = np.repeat(tech_indicators[:1], padding_size, axis=0)
            prices = np.vstack([price_padding, prices])
            tech_indicators = np.vstack([tech_padding, tech_indicators])

        # Normalize data
        prices_mean = np.mean(prices, axis=0)
        prices_std = np.std(prices, axis=0)
        tech_mean = np.mean(tech_indicators, axis=0)
        tech_std = np.std(tech_indicators, axis=0)

        # Avoid division by zero
        prices_std = np.where(prices_std == 0, 1, prices_std)
        tech_std = np.where(tech_std == 0, 1, tech_std)

        normalized_prices = (prices - prices_mean) / prices_std
        normalized_tech = (tech_indicators - tech_mean) / tech_std

        # Concatenate price and technical indicators
        features = np.column_stack([normalized_prices, normalized_tech])

        # Flatten to 1D array
        flattened_features = features.flatten()

        # Add position and balance information
        current_price = self.df["close"].iloc[self.current_step]
        position_value = self.current_position * current_price / self.initial_balance
        cash_ratio = self.balance / self.initial_balance

        # Build complete state features
        observation = np.append(flattened_features, [position_value, cash_ratio])

        return observation.astype(np.float32)

    def _take_action(self, action: int) -> float:
        """
        Execute trading action and calculate reward

        Args:
            action: Trading action (0: hold, 1: buy, 2: sell)

        Returns:
            Reward value
        """
        current_price = self.df["close"].iloc[self.current_step]
        prev_portfolio_value = self.balance + self.current_position * current_price

        # Execute trade
        if action == 1:  # Buy
            # Calculate maximum number of shares that can be bought
            max_possible_shares = self.balance // (
                current_price * (1 + self.transaction_fee_percent)
            )
            # Use 25% of capital each time to buy
            shares_to_buy = max(max_possible_shares // 4, 1)

            if shares_to_buy > 0 and self.balance >= shares_to_buy * current_price * (
                1 + self.transaction_fee_percent
            ):
                cost = (
                    shares_to_buy * current_price * (1 + self.transaction_fee_percent)
                )
                self.balance -= cost
                self.current_position += shares_to_buy
                trade_info = {
                    "step": self.current_step,
                    "action": "buy",
                    "shares": shares_to_buy,
                    "price": current_price,
                    "cost": cost,
                }
                self.history.append(trade_info)

        elif action == 2:  # Sell
            if self.current_position > 0:
                # Sell 25% of current position each time
                shares_to_sell = max(self.current_position // 4, 1)

                sell_amount = (
                    shares_to_sell * current_price * (1 - self.transaction_fee_percent)
                )
                self.balance += sell_amount
                self.current_position -= shares_to_sell
                trade_info = {
                    "step": self.current_step,
                    "action": "sell",
                    "shares": shares_to_sell,
                    "price": current_price,
                    "amount": sell_amount,
                }
                self.history.append(trade_info)

        # Calculate new portfolio value
        new_portfolio_value = self.balance + self.current_position * current_price

        # Calculate return rate
        portfolio_return = (new_portfolio_value / prev_portfolio_value) - 1

        # Apply reward scaling
        reward = portfolio_return * self.reward_scaling

        # Encourage trading, reduce holding penalty
        if action == 0:  # Hold
            # If position ratio is low, give small penalty
            position_ratio = self.current_position * current_price / new_portfolio_value
            if position_ratio < 0.1:  # Position less than 10%
                reward -= 0.001  # Small penalty, encourage position building
        else:  # Trade (buy or sell)
            # Slight trading reward, encourage exploration
            reward += 0.0005

        self.total_reward += reward

        return reward

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step of environment interaction

        Args:
            action: Trading action

        Returns:
            observation: New state
            reward: Reward value
            done: Whether finished
            info: Additional information
        """
        # Execute action and get reward
        reward = self._take_action(action)

        # Move forward one step
        self.current_step += 1

        # Check if finished
        done = (self.current_step >= len(self.df) - 1) or (
            self.current_step >= self.max_steps
        )

        # Get new state
        observation = self._next_observation()

        # Prepare additional information
        current_price = (
            self.df["close"].iloc[self.current_step]
            if self.current_step < len(self.df)
            else self.df["close"].iloc[-1]
        )
        info = {
            "current_step": self.current_step,
            "current_price": current_price,
            "current_position": self.current_position,
            "balance": self.balance,
            "portfolio_value": self.balance + self.current_position * current_price,
            "total_reward": self.total_reward,
        }

        return observation, reward, done, info

    def reset(self):
        """
        Reset environment

        Returns:
            Initial state
        """
        max_start_idx = len(self.df) - self.max_steps
        min_start_idx = self.window_size

        if min_start_idx >= max_start_idx:
            logger.info(
                f"Warning: Insufficient data. window_size={self.window_size}, max_steps={self.max_steps}, data length={len(self.df)}"
            )
            # Adjust parameters to ensure starting point can be selected
            adjusted_max_steps = len(self.df) - min_start_idx - 5  # Add some buffer
            self.max_steps = max(10, adjusted_max_steps)  # At least 10 steps
            max_start_idx = len(self.df) - self.max_steps
            logger.info(f"Adjusted max_steps to {self.max_steps}")

        # Randomly select starting point
        self.current_step = np.random.randint(min_start_idx, max_start_idx)

        # Reset capital and position
        self.balance = self.initial_balance
        self.current_position = 0
        self.total_reward = 0
        self.history = []

        return self._next_observation()

    def render(self, mode="human"):
        """
        Render environment

        Args:
            mode: Render mode
        """
        if mode == "human":
            current_price = self.df["close"].iloc[self.current_step]
            portfolio_value = self.balance + self.current_position * current_price
            logger.info(f"Step: {self.current_step}")
            logger.info(f"Price: {current_price:.2f}")
            logger.info(f"Balance: {self.balance:.2f}")
            logger.info(f"Shares: {self.current_position}")
            logger.info(f"Portfolio Value: {portfolio_value:.2f}")
            logger.info(f"Total Reward: {self.total_reward:.4f}")
            logger.info("-------------------")


class ActorCritic(nn.Module):
    """PPO Actor-Critic network"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        Initialize Actor-Critic network

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            hidden_dim: Hidden layer dimension
        """
        super(ActorCritic, self).__init__()

        # Shared feature extraction layer
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor network (policy network)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )

        # Critic network (value network)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        """
        Forward propagation

        Args:
            state: State

        Returns:
            action_probs: Action probabilities
            state_value: State value
        """
        features = self.feature_extractor(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value

    def act(self, state):
        """
        Select action based on policy

        Args:
            state: State

        Returns:
            action: Selected action
            action_prob: Probability of selected action
            state_value: State value
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)  # Use global DEVICE
        action_probs, state_value = self.forward(state)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        action_prob = action_probs[0, action.item()]
        return action.item(), action_prob.item(), state_value.item()


class PPOAgent:
    """PPO intelligent trading agent"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 0.0003,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        k_epochs: int = 4,
        device: str = None,
    ):
        """
        Initialize PPO agent

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            hidden_dim: Hidden layer dimension
            lr: Learning rate
            gamma: Discount factor
            eps_clip: PPO clipping parameter
            k_epochs: Number of policy update rounds
            device: Running device ('cuda' or 'cpu')
        """
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        # Explicitly use CUDA device
        if device is None:
            self.device = DEVICE  # Use global device configuration
        else:
            self.device = torch.device(device)

        logger.info(f"PPO agent initialization completed, using device: {self.device}")

        # Policy network
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # Old policy
        self.old_policy = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.old_policy.load_state_dict(self.policy.state_dict())

        # Loss function
        self.MseLoss = nn.MSELoss()

        # Experience buffer
        self.buffer = []

    def select_action(self, state):
        """
        Select action based on current policy

        Args:
            state: State

        Returns:
            action: Selected action
            action_prob: Action probability
            state_value: State value
        """
        with torch.no_grad():
            return self.old_policy.act(state)

    def update_old_policy(self):
        """Update old policy"""
        self.old_policy.load_state_dict(self.policy.state_dict())

    def store_transition(self, transition):
        """
        Store state transition

        Args:
            transition: (state, action, action_prob, reward, next_state, done)
        """
        self.buffer.append(transition)

    def update(self):
        """
        Update policy

        Returns:
            actor_loss: Actor network loss
            critic_loss: Critic network loss
        """
        # Extract data from buffer
        old_states = torch.FloatTensor(np.array([t[0] for t in self.buffer])).to(
            self.device
        )
        old_actions = torch.LongTensor(np.array([t[1] for t in self.buffer])).to(
            self.device
        )
        old_action_probs = torch.FloatTensor(np.array([t[2] for t in self.buffer])).to(
            self.device
        )
        rewards = [t[3] for t in self.buffer]
        next_states = [t[4] for t in self.buffer]
        dones = [t[5] for t in self.buffer]

        # Calculate cumulative returns
        returns = []
        discounted_reward = 0
        for reward, next_state, done in zip(
            reversed(rewards), reversed(next_states), reversed(dones)
        ):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)

        # Normalize returns
        returns = torch.FloatTensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # Multiple optimization
        actor_loss_epoch = 0
        critic_loss_epoch = 0

        for _ in range(self.k_epochs):
            # Get current policy action probabilities and state values
            action_probs, state_values = self.policy(old_states)
            state_values = state_values.squeeze()

            # Calculate probability ratio between old and new policies
            dist = Categorical(action_probs)
            new_action_probs = dist.log_prob(old_actions).exp()

            # Calculate ratio
            ratios = new_action_probs / old_action_probs

            # Calculate advantages
            advantages = returns - state_values.detach()

            # PPO loss
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )
            actor_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            critic_loss = self.MseLoss(state_values, returns)

            # Total loss
            loss = actor_loss + 0.5 * critic_loss

            # Optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            actor_loss_epoch += actor_loss.item()
            critic_loss_epoch += critic_loss.item()

        # Clear buffer
        self.buffer = []

        return actor_loss_epoch / self.k_epochs, critic_loss_epoch / self.k_epochs

    def save_model(self, path):
        """
        Save model

        Args:
            path: Save path
        """
        torch.save(self.policy.state_dict(), path)

    def load_model(self, path):
        """
        Load model

        Args:
            path: Model path
        """
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.policy.load_state_dict(state_dict)
        self.old_policy.load_state_dict(self.policy.state_dict())


class RLTrader:
    """Reinforcement learning trading system, integrating PPO agent and trading environment"""

    def __init__(
        self, model_dir: str = "models", hidden_dim: int = 128, device: str = None
    ):
        """
        Initialize RL trading system

        Args:
            model_dir: Model save directory
            hidden_dim: Network hidden layer dimension
            device: Running device
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        self.hidden_dim = hidden_dim
        # Explicitly use CUDA device
        if device is None:
            self.device = DEVICE  # Use global device configuration
        else:
            self.device = device

        self.env = None
        self.agent = None
        self.trained = False

        self.logger = setup_logger("rl_trader")
        self.logger.info(
            f"RLTrader initialization completed, using device: {self.device}"
        )

    def train(
        self,
        df,
        initial_balance=100000,
        transaction_fee_percent=0.001,
        n_episodes=1000,
        batch_size=64,
        reward_scaling=100.0,
        max_steps=252,
        window_size=20,
    ):
        """
        Train reinforcement learning trading model

        Args:
            df: Price data DataFrame
            initial_balance: Initial capital
            transaction_fee_percent: Transaction fee percentage
            n_episodes: Number of training episodes
            batch_size: Batch size
            reward_scaling: Reward scaling factor
            max_steps: Maximum steps per episode
        """
        # Create trading environment
        self.env = StockTradingEnv(
            df=df,
            initial_balance=initial_balance,
            transaction_fee_percent=transaction_fee_percent,
            reward_scaling=reward_scaling,
            max_steps=max_steps,
            window_size=window_size,
        )

        # Create PPO agent
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n

        self.agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=self.hidden_dim,
            device=self.device,  # Explicitly pass device parameter
        )

        # Training records
        episode_rewards = []
        actor_losses = []
        critic_losses = []
        portfolio_values = []

        self.logger.info(f"Starting training, {n_episodes} episodes...")
        best_reward = -float("inf")

        for episode in range(n_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                # Select action
                action, action_prob, _ = self.agent.select_action(state)

                # Execute action
                next_state, reward, done, info = self.env.step(action)

                # Store transition
                self.agent.store_transition(
                    (state, action, action_prob, reward, next_state, done)
                )

                # If buffer is full, update policy
                if len(self.agent.buffer) >= batch_size:
                    actor_loss, critic_loss = self.agent.update()
                    self.agent.update_old_policy()
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)

                state = next_state
                episode_reward += reward

            # Record results
            episode_rewards.append(episode_reward)
            portfolio_values.append(info["portfolio_value"])

            # Print training progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_value = np.mean(portfolio_values[-10:])
                self.logger.info(
                    f"Episode {episode+1}/{n_episodes}, Avg Reward: {avg_reward:.4f}, Avg Portfolio: {avg_value:.2f}"
                )

            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                self.save_model("best_model")

        # Final model save
        self.save_model("final_model")
        self.trained = True

        # Return training records
        training_history = {
            "episode_rewards": episode_rewards,
            "actor_losses": actor_losses,
            "critic_losses": critic_losses,
            "portfolio_values": portfolio_values,
        }

        return training_history

    def test(
        self,
        df: pd.DataFrame,
        initial_balance: float = 100000,
        transaction_fee_percent: float = 0.001,
        model_name: str = "best_model",
    ):
        """
        Test reinforcement learning trading model

        Args:
            df: Price data DataFrame
            initial_balance: Initial capital
            transaction_fee_percent: Transaction fee percentage
            model_name: Model name

        Returns:
            Test results dictionary
        """
        # Create test environment
        test_env = StockTradingEnv(
            df=df,
            initial_balance=initial_balance,
            transaction_fee_percent=transaction_fee_percent,
            reward_scaling=1.0,
            max_steps=len(df),
        )

        # Ensure agent is loaded
        if self.agent is None:
            state_dim = test_env.observation_space.shape[0]
            action_dim = test_env.action_space.n

            self.agent = PPOAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=self.hidden_dim,
                device=self.device,  # Explicitly pass device parameter
            )
            self.load_model(model_name)

        # Test
        state = test_env.reset()
        done = False
        total_reward = 0

        # Record test process
        states = []
        actions = []
        rewards = []
        portfolio_values = []
        balances = []
        positions = []
        prices = []

        while not done:
            # Select action
            action, _, _ = self.agent.select_action(state)

            # Execute action
            next_state, reward, done, info = test_env.step(action)

            # Record
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            portfolio_values.append(info["portfolio_value"])
            balances.append(info["balance"])
            positions.append(info["current_position"])
            prices.append(info["current_price"])

            state = next_state
            total_reward += reward

        # Calculate performance metrics
        returns = np.array(portfolio_values) / initial_balance - 1
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]

        # Calculate Sharpe ratio
        avg_daily_return = np.mean(daily_returns)
        std_daily_return = np.std(daily_returns)
        sharpe_ratio = (
            (avg_daily_return / std_daily_return) * np.sqrt(252)
            if std_daily_return > 0
            else 0
        )

        # Maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown)

        # Total return
        total_return = portfolio_values[-1] / initial_balance - 1

        # Test results
        test_results = {
            "total_reward": total_reward,
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "portfolio_values": portfolio_values,
            "actions": actions,
            "positions": positions,
            "balances": balances,
            "prices": prices,
            "trades": test_env.history,
        }

        self.logger.info(
            f"Test completed, total return: {total_return:.2%}, Sharpe ratio: {sharpe_ratio:.2f}, max drawdown: {max_drawdown:.2%}"
        )

        return test_results

    def save_model(self, model_name: str = "ppo_model"):
        """
        Save model

        Args:
            model_name: Model name
        """
        if self.agent is None:
            self.logger.warning("No model to save")
            return

        model_path = os.path.join(self.model_dir, f"{model_name}.pth")
        self.agent.save_model(model_path)
        self.logger.info(f"Model saved to {model_path}")

    def load_model(self, model_name: str = "ppo_model"):
        """
        Load model

        Args:
            model_name: Model name
        """
        model_path = os.path.join(self.model_dir, f"{model_name}.pth")

        if not os.path.exists(model_path):
            self.logger.warning(f"Model file {model_path} does not exist")
            return False

        if self.agent is None:
            self.logger.warning("Please initialize agent first")
            return False

        self.agent.load_model(model_path)
        self.trained = True
        self.logger.info(f"Model loaded successfully from {model_path}")
        return True

    def generate_trading_signal(self, state):
        """
        Generate trading signal

        Args:
            state: Current state

        Returns:
            Trading signal dictionary
        """
        if not self.trained or self.agent is None:
            return {"signal": "neutral", "confidence": 0.5}

        # Get action and probability
        with torch.no_grad():
            try:
                state_tensor = (
                    torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                )
                action_probs, _ = self.agent.policy(state_tensor)
                action_probs = action_probs.squeeze().cpu().numpy()

                # Print debug information
                logger.info(f"Action probabilities: {action_probs}")

                # Select action
                action = np.argmax(action_probs)

                # Convert to trading signal
                signal_map = {0: "neutral", 1: "bullish", 2: "bearish"}

                # Ensure action is within valid range
                if action not in signal_map:
                    logger.info(f"Warning: Invalid action index {action}")
                    action = 0  # Default to neutral

                # Calculate confidence
                confidence = float(action_probs[action])

                # Build action probability dictionary
                action_prob_dict = {}
                for i, prob in enumerate(action_probs):
                    if i in signal_map:
                        action_prob_dict[signal_map[i]] = float(prob)

                # Use both action_probs and action_probabilities keys for compatibility
                return {
                    "signal": signal_map[action],
                    "confidence": confidence,
                    "action_probs": action_prob_dict,
                    "action_probabilities": action_prob_dict,
                }
            except Exception as e:
                logger.info(f"Error generating trading signal: {e}")
                import traceback

                traceback.logger.info_exc()
                return {"signal": "neutral", "confidence": 0.5}


class RLTradingAgent:
    """Reinforcement learning trading Agent, integrated into existing system"""

    def __init__(self, model_dir: str = "models"):
        """
        Initialize RL trading Agent

        Args:
            model_dir: Model save directory
        """
        self.rl_trader = RLTrader(
            model_dir=model_dir, device="cuda"
        )  # Explicitly specify using cuda
        self.window_size = 20  # Observation window size
        self.logger = setup_logger("rl_trading_agent")
        self.is_trained = False

    def train(
        self, price_data: pd.DataFrame, tech_indicators: Dict[str, pd.Series] = None
    ):
        """
        Train RL trading model

        Args:
            price_data: Price data
            tech_indicators: Technical indicators
        """
        try:
            # Prepare training data
            df = self._prepare_data(price_data, tech_indicators)

            # Train model
            training_history = self.rl_trader.train(
                df=df,
                n_episodes=500,  # Reduce episodes to speed up training
                batch_size=32,
                reward_scaling=1.0,
            )

            self.is_trained = True
            return training_history

        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            return None

    def load_model(self, model_name: str = "best_model"):
        """
        Load trained model

        Args:
            model_name: Model name
        """
        # Create empty environment to initialize agent
        dummy_df = pd.DataFrame(
            {
                "open": [100] * 100,
                "high": [110] * 100,
                "low": [90] * 100,
                "close": [105] * 100,
                "volume": [1000] * 100,
            }
        )

        # Add technical indicators
        for indicator in [
            "ma5",
            "ma10",
            "ma20",
            "rsi",
            "macd",
            "macd_signal",
            "macd_hist",
            "volatility_5d",
            "volatility_10d",
            "volatility_20d",
        ]:
            dummy_df[indicator] = 0

        # Create environment
        self.rl_trader.env = StockTradingEnv(df=dummy_df, window_size=self.window_size)

        # Initialize agent
        state_dim = self.rl_trader.env.observation_space.shape[0]
        action_dim = self.rl_trader.env.action_space.n

        self.rl_trader.agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=self.rl_trader.hidden_dim,
            device="cuda",  # Explicitly specify using cuda
        )

        # Load model
        success = self.rl_trader.load_model(model_name)
        self.is_trained = success
        return success

    def generate_signals(
        self, price_data: pd.DataFrame, tech_indicators: Dict[str, pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Generate trading signals

        Args:
            price_data: Price data
            tech_indicators: Technical indicators

        Returns:
            Dictionary containing trading signals
        """
        signals = {}

        try:
            # Load model (if not yet loaded)
            if not self.is_trained:
                loaded = self.load_model()
                if not loaded:
                    return {
                        "signal": "neutral",
                        "confidence": 0.5,
                        "error": "Model loading failed",
                    }

            # Prepare data
            df = self._prepare_data(price_data, tech_indicators)

            # Check if data volume is sufficient
            if len(df) < self.window_size + 5:  # Ensure sufficient data
                return {
                    "signal": "neutral",
                    "confidence": 0.5,
                    "error": f"Insufficient data, need at least {self.window_size + 5} data points, but only have {len(df)}",
                }

            # Create environment
            env = StockTradingEnv(
                df=df,
                window_size=self.window_size,
                max_steps=10,  # Small number of steps
            )

            # Get state
            state = env.reset()

            logger.info(f"\nState shape for signal generation: {state.shape}\n")

            # Generate signal
            signal_info = self.rl_trader.generate_trading_signal(state)

            # Add policy analysis
            signals["rl_signal"] = signal_info["signal"]
            signals["rl_confidence"] = signal_info["confidence"]

            # Try both key names
            if "action_probabilities" in signal_info:
                signals["action_probabilities"] = signal_info["action_probabilities"]
            if "action_probs" in signal_info:
                signals["action_probs"] = signal_info["action_probs"]

            # Add reasoning analysis
            signals["reasoning"] = self._generate_reasoning(signal_info)

            # Final signal
            signals["signal"] = signal_info["signal"]
            signals["confidence"] = signal_info["confidence"]

        except Exception as e:
            self.logger.error(f"Error generating trading signals: {e}")
            import traceback

            traceback.logger.info_exc()
            signals["signal"] = "neutral"
            signals["confidence"] = 0.5
            signals["error"] = str(e)

        return signals

    def _prepare_data(
        self, price_data: pd.DataFrame, tech_indicators: Dict[str, pd.Series] = None
    ) -> pd.DataFrame:
        """
        Prepare model input data

        Args:
            price_data: Price data
            tech_indicators: Technical indicators

        Returns:
            Processed DataFrame
        """
        # Copy data
        df = price_data.copy()

        # Ensure necessary columns exist
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Price data missing required column: {col}")

        # Calculate common technical indicators
        # Moving averages
        df["ma5"] = df["close"].rolling(window=5).mean()
        df["ma10"] = df["close"].rolling(window=10).mean()
        df["ma20"] = df["close"].rolling(window=20).mean()

        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = df["ema12"] - df["ema26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Volatility
        df["volatility_5d"] = df["close"].pct_change().rolling(window=5).std()
        df["volatility_10d"] = df["close"].pct_change().rolling(window=10).std()
        df["volatility_20d"] = df["close"].pct_change().rolling(window=20).std()

        # Add external technical indicators
        if tech_indicators:
            for name, indicator in tech_indicators.items():
                if name not in df.columns:
                    df[name] = indicator

        # Fill missing values
        df = df.ffill().bfill()

        return df

    def _generate_reasoning(self, signal_info: Dict[str, Any]) -> str:
        """
        Generate decision reasoning based on signal

        Args:
            signal_info: Signal information dictionary

        Returns:
            Decision reasoning
        """
        signal = signal_info["signal"]
        confidence = signal_info["confidence"]

        # Try both possible key names
        action_probs = {}
        if "action_probabilities" in signal_info:
            action_probs = signal_info["action_probabilities"]
        elif "action_probs" in signal_info:
            action_probs = signal_info["action_probs"]

        reasoning = []

        # Analyze signal
        if signal == "bullish":
            reasoning.append(
                f"Reinforcement learning model predicts bullish signal, confidence: {confidence:.2%}"
            )
            if "bullish" in action_probs:
                reasoning.append(
                    f"Buy probability: {action_probs.get('bullish', 0):.2%}"
                )
        elif signal == "bearish":
            reasoning.append(
                f"Reinforcement learning model predicts bearish signal, confidence: {confidence:.2%}"
            )
            if "bearish" in action_probs:
                reasoning.append(
                    f"Sell probability: {action_probs.get('bearish', 0):.2%}"
                )
        else:
            reasoning.append(
                f"Reinforcement learning model predicts neutral signal, confidence: {confidence:.2%}"
            )
            if "neutral" in action_probs:
                reasoning.append(
                    f"Hold probability: {action_probs.get('neutral', 0):.2%}"
                )

        # Analyze action probability distribution
        if action_probs:
            probs_str = ", ".join([f"{k}: {v:.2%}" for k, v in action_probs.items()])
            reasoning.append(f"Action probability distribution: {probs_str}")
        else:
            reasoning.append("Action probability distribution: No valid data")

        return "; ".join(reasoning)
