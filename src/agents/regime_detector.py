import numpy as np
import pandas as pd
from typing import Dict
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AdvancedRegimeDetector:
    """
    基于2024-2025研究的高级市场区制检测器
    实现多维度特征的马尔科夫区制转换模型
    """
    
    def __init__(self, n_regimes: int = 3, lookback_window: int = 252):
        self.n_regimes = n_regimes
        self.lookback_window = lookback_window
        self.regime_model = None
        self.scaler = StandardScaler()
        self.regime_names = {
            0: "low_volatility_trending",
            1: "high_volatility_mean_reverting", 
            2: "crisis_regime"
        }
        
    def extract_regime_features(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        提取多维度市场特征用于区制识别
        基于Lopez-Lira 2025框架的特征工程
        """
        features = pd.DataFrame(index=prices_df.index)
        
        # 价格相关特征
        returns = prices_df['close'].pct_change()
        features['returns'] = returns
        features['log_returns'] = np.log(1 + returns)
        
        # 波动率特征 (多时间尺度)
        features['volatility_5d'] = returns.rolling(5).std()
        features['volatility_21d'] = returns.rolling(21).std()
        features['volatility_63d'] = returns.rolling(63).std()
        features['volatility_ratio'] = features['volatility_5d'] / features['volatility_21d']
        
        # 趋势特征
        features['price_ma_ratio_20'] = prices_df['close'] / prices_df['close'].rolling(20).mean()
        features['price_ma_ratio_50'] = prices_df['close'] / prices_df['close'].rolling(50).mean()
        features['ma_slope_20'] = prices_df['close'].rolling(20).mean().pct_change(5)
        
        # 动量特征
        features['momentum_5d'] = returns.rolling(5).sum()
        features['momentum_21d'] = returns.rolling(21).sum()
        features['rsi'] = self._calculate_rsi(prices_df['close'])
        
        # 成交量特征 (如果有成交量数据)
        if 'volume' in prices_df.columns:
            features['volume_ma_ratio'] = prices_df['volume'] / prices_df['volume'].rolling(20).mean()
            features['price_volume_trend'] = (returns * np.log(1 + prices_df['volume'].pct_change())).rolling(10).mean()
        
        # 市场微观结构特征
        features['high_low_ratio'] = (prices_df['high'] - prices_df['low']) / prices_df['close']
        features['close_position'] = (prices_df['close'] - prices_df['low']) / (prices_df['high'] - prices_df['low'])
        
        # 跳跃检测 (基于Barndorff-Nielsen & Shephard测试)
        features['jump_indicator'] = self._detect_jumps(returns)
        
        # 长记忆性特征 (Hurst指数)
        features['hurst_21d'] = returns.rolling(21).apply(lambda x: self._calculate_hurst(x) if len(x) >= 10 else np.nan)
        
        return features.dropna()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _detect_jumps(self, returns: pd.Series, threshold: float = 3.0) -> pd.Series:
        """检测价格跳跃"""
        rolling_std = returns.rolling(21).std()
        standardized_returns = returns / rolling_std
        return (np.abs(standardized_returns) > threshold).astype(int)
    
    def _calculate_hurst(self, ts: pd.Series) -> float:
        """计算Hurst指数"""
        try:
            if len(ts) < 10:
                return 0.5
            
            lags = range(2, min(len(ts)//2, 20))
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        except:
            return 0.5
    
    def fit_regime_model(self, features: pd.DataFrame) -> Dict:
        """
        拟合高斯混合模型进行区制识别
        """
        # 选择关键特征进行建模
        key_features = [
            'returns', 'volatility_21d', 'volatility_ratio', 
            'price_ma_ratio_20', 'momentum_21d', 'rsi',
            'high_low_ratio', 'jump_indicator', 'hurst_21d'
        ]
        
        # 过滤存在的特征
        available_features = [f for f in key_features if f in features.columns]
        model_data = features[available_features].dropna()
        
        if len(model_data) < 50:
            return {"error": "Insufficient data for regime modeling"}
        
        # 标准化特征
        scaled_features = self.scaler.fit_transform(model_data)
        
        # 拟合高斯混合模型
        self.regime_model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            random_state=42,
            max_iter=200
        )
        
        self.regime_model.fit(scaled_features)
        
        # 预测区制
        regime_probs = self.regime_model.predict_proba(scaled_features)
        regime_labels = self.regime_model.predict(scaled_features)
        
        # 计算区制特征
        regime_stats = self._analyze_regime_characteristics(model_data, regime_labels)
        
        return {
            "regime_probabilities": regime_probs,
            "regime_labels": regime_labels,
            "regime_characteristics": regime_stats,
            "model_score": self.regime_model.score(scaled_features),
            "feature_names": available_features
        }
    
    def _analyze_regime_characteristics(self, data: pd.DataFrame, labels: np.ndarray) -> Dict:
        """分析各个区制的特征"""
        regime_chars = {}
        
        for regime in range(self.n_regimes):
            regime_mask = labels == regime
            regime_data = data[regime_mask]
            
            if len(regime_data) > 0:
                regime_chars[regime] = {
                    "avg_return": float(regime_data['returns'].mean()),
                    "volatility": float(regime_data['volatility_21d'].mean()),
                    "momentum": float(regime_data['momentum_21d'].mean()),
                    "frequency": float(np.sum(regime_mask) / len(labels)),
                    "avg_duration": self._calculate_avg_duration(regime_mask),
                    "regime_name": self._classify_regime(regime_data)
                }
        
        return regime_chars
    
    def _calculate_avg_duration(self, regime_mask: np.ndarray) -> float:
        """计算区制平均持续时间"""
        durations = []
        current_duration = 0
        
        for is_regime in regime_mask:
            if is_regime:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0
        
        if current_duration > 0:
            durations.append(current_duration)
        
        return np.mean(durations) if durations else 0
    
    def _classify_regime(self, regime_data: pd.DataFrame) -> str:
        """基于特征自动分类区制类型"""
        avg_vol = regime_data['volatility_21d'].mean()
        avg_momentum = regime_data['momentum_21d'].mean()
        avg_return = regime_data['returns'].mean()
        
        if avg_vol > regime_data['volatility_21d'].quantile(0.7):
            if avg_return < -0.001:
                return "crisis_regime"
            else:
                return "high_volatility_mean_reverting"
        else:
            if abs(avg_momentum) > 0.02:
                return "low_volatility_trending"
            else:
                return "low_volatility_consolidation"
    
    def predict_current_regime(self, features: pd.DataFrame) -> Dict:
        """预测当前市场区制"""
        if self.regime_model is None:
            return {"error": "Model not fitted"}
        
        # 获取最新特征
        latest_features = features.iloc[-1:][self.regime_model.feature_names_in_]
        scaled_features = self.scaler.transform(latest_features)
        
        # 预测区制概率
        regime_probs = self.regime_model.predict_proba(scaled_features)[0]
        predicted_regime = np.argmax(regime_probs)
        
        return {
            "predicted_regime": int(predicted_regime),
            "regime_name": self.regime_names.get(predicted_regime, f"regime_{predicted_regime}"),
            "regime_probabilities": {f"regime_{i}": float(prob) for i, prob in enumerate(regime_probs)},
            "confidence": float(np.max(regime_probs))
        }

def adaptive_signal_aggregation(signals: Dict, regime_info: Dict, confidence_threshold: float = 0.6) -> Dict:
    """
    基于FLAG-Trader 2025研究的自适应信号聚合
    根据市场区制动态调整信号权重
    """
    regime_name = regime_info.get("regime_name", "unknown")
    regime_confidence = regime_info.get("confidence", 0.5)
    
    # 基础权重 (来自FINSABER 2024研究)
    base_weights = {
        'technical': 0.25,
        'fundamental': 0.20,
        'sentiment': 0.15,
        'valuation': 0.15,
        'ai_model': 0.15,
        'macro': 0.10
    }
    
    # 区制特定权重调整 (基于Lopez-Lira 2025框架)
    regime_adjustments = {
        "low_volatility_trending": {
            'technical': 1.3,  # 增强技术分析权重
            'ai_model': 1.2,   # AI模型在趋势市场表现更好
            'sentiment': 0.8,  # 降低情绪权重
            'fundamental': 0.9
        },
        "high_volatility_mean_reverting": {
            'fundamental': 1.4,  # 基本面在震荡市场更重要
            'valuation': 1.3,    # 估值回归
            'technical': 0.8,    # 降低技术分析权重
            'sentiment': 0.7     # 情绪噪音较大
        },
        "crisis_regime": {
            'macro': 1.5,       # 宏观因素主导
            'sentiment': 1.2,   # 恐慌情绪重要
            'ai_model': 0.7,    # AI模型在危机中表现较差
            'technical': 0.8,
            'fundamental': 0.9
        }
    }
    
    # 应用区制调整
    adjusted_weights = base_weights.copy()
    if regime_name in regime_adjustments and regime_confidence > confidence_threshold:
        adjustments = regime_adjustments[regime_name]
        for signal_type in adjusted_weights:
            if signal_type in adjustments:
                adjusted_weights[signal_type] *= adjustments[signal_type]
    
    # 归一化权重
    total_weight = sum(adjusted_weights.values())
    adjusted_weights = {k: v/total_weight for k, v in adjusted_weights.items()}
    
    # 计算加权信号
    weighted_signal = 0
    weighted_confidence = 0
    signal_contributions = {}
    
    for signal_type, weight in adjusted_weights.items():
        if signal_type in signals:
            signal_data = signals[signal_type]
            signal_value = signal_data.get('signal', 0)
            signal_conf = signal_data.get('confidence', 0.5)
            
            # 使用置信度加权的信号值
            contribution = weight * signal_value * signal_conf
            weighted_signal += contribution
            weighted_confidence += weight * signal_conf
            
            signal_contributions[signal_type] = {
                'weight': weight,
                'signal': signal_value,
                'confidence': signal_conf,
                'contribution': contribution
            }
    
    # 应用动态阈值 (基于RLMF 2024技术)
    dynamic_threshold = 0.3 if regime_name == "crisis_regime" else 0.2
    
    # 信号强度调整
    if abs(weighted_signal) < dynamic_threshold:
        weighted_signal *= 0.5  # 弱信号进一步衰减
    
    return {
        "aggregated_signal": weighted_signal,
        "aggregated_confidence": weighted_confidence,
        "regime_adjusted_weights": adjusted_weights,
        "signal_contributions": signal_contributions,
        "regime_info": regime_info,
        "dynamic_threshold": dynamic_threshold
    } 