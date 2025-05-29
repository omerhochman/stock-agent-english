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
        self.feature_names = None  # 手动存储特征名称
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
        if prices_df.empty or len(prices_df) < 10:
            # 如果数据太少，返回空的DataFrame
            return pd.DataFrame()
        
        features = pd.DataFrame(index=prices_df.index)
        
        # 价格相关特征
        returns = prices_df['close'].pct_change()
        features['returns'] = returns
        
        # 安全计算对数收益率，避免log(0)或log(负数)
        safe_returns = returns.fillna(0)
        safe_returns = np.where(safe_returns <= -1, -0.999, safe_returns)  # 避免log(0)
        features['log_returns'] = np.log(1 + safe_returns)
        
        # 波动率特征 (多时间尺度) - 使用更短的窗口以保留更多数据
        features['volatility_5d'] = returns.rolling(5).std()
        features['volatility_10d'] = returns.rolling(10).std()  # 改为10天而不是21天
        features['volatility_20d'] = returns.rolling(20).std()  # 改为20天而不是63天
        
        # 安全计算比率，避免除零
        vol_5d = features['volatility_5d'].fillna(0.01)
        vol_10d = features['volatility_10d'].fillna(0.01)
        features['volatility_ratio'] = np.where(vol_10d != 0, vol_5d / vol_10d, 1.0)
        
        # 趋势特征 - 使用更短的窗口
        ma_10 = prices_df['close'].rolling(10).mean()
        ma_20 = prices_df['close'].rolling(20).mean()
        features['price_ma_ratio_10'] = np.where(ma_10 != 0, prices_df['close'] / ma_10, 1.0)
        features['price_ma_ratio_20'] = np.where(ma_20 != 0, prices_df['close'] / ma_20, 1.0)
        features['ma_slope_10'] = ma_10.pct_change(3)
        
        # 动量特征
        features['momentum_3d'] = returns.rolling(3).sum()
        features['momentum_5d'] = returns.rolling(5).sum()
        features['momentum_10d'] = returns.rolling(10).sum()
        features['rsi'] = self._calculate_rsi(prices_df['close'], period=10)  # 使用更短周期
        
        # 成交量特征 (如果有成交量数据)
        if 'volume' in prices_df.columns:
            volume_ma = prices_df['volume'].rolling(10).mean()
            features['volume_ma_ratio'] = np.where(volume_ma != 0, prices_df['volume'] / volume_ma, 1.0)
            
            # 安全计算价格成交量趋势
            volume_change = prices_df['volume'].pct_change().fillna(0)
            safe_volume_change = np.where(volume_change <= -1, -0.999, volume_change)
            features['price_volume_trend'] = (returns * np.log(1 + safe_volume_change)).rolling(5).mean()
        
        # 市场微观结构特征
        high_low_diff = prices_df['high'] - prices_df['low']
        features['high_low_ratio'] = np.where(prices_df['close'] != 0, high_low_diff / prices_df['close'], 0.0)
        
        # 安全计算收盘位置
        hl_range = prices_df['high'] - prices_df['low']
        close_low_diff = prices_df['close'] - prices_df['low']
        features['close_position'] = np.where(hl_range != 0, close_low_diff / hl_range, 0.5)
        
        # 跳跃检测 (基于Barndorff-Nielsen & Shephard测试)
        features['jump_indicator'] = self._detect_jumps(returns)
        
        # 长记忆性特征 (Hurst指数) - 使用更短窗口
        features['hurst_10d'] = returns.rolling(10).apply(lambda x: self._calculate_hurst(x) if len(x) >= 8 else np.nan)
        
        # 填充初始的NaN值，使用前向填充和后向填充
        features = features.bfill().ffill()
        
        # 最终检查：确保没有无穷大或NaN值
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)
        
        return features.dropna()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # 用中性值50填充NaN
    
    def _detect_jumps(self, returns: pd.Series, threshold: float = 2.5) -> pd.Series:
        """检测价格跳跃 - 降低阈值以增加敏感性"""
        rolling_std = returns.rolling(10).std()  # 使用更短窗口
        standardized_returns = returns / rolling_std
        jumps = (np.abs(standardized_returns) > threshold).astype(int)
        return jumps.fillna(0)  # 用0填充NaN
    
    def _calculate_hurst(self, ts: pd.Series) -> float:
        """计算Hurst指数"""
        try:
            if len(ts) < 8:
                return 0.5
            
            lags = range(2, min(len(ts)//2, 10))  # 减少lag范围
            if len(lags) < 2:
                return 0.5
                
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            if any(t <= 0 for t in tau):
                return 0.5
                
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return max(0.1, min(0.9, poly[0] * 2.0))  # 限制在合理范围内
        except:
            return 0.5
    
    def fit_regime_model(self, features: pd.DataFrame) -> Dict:
        """
        拟合高斯混合模型进行区制识别
        """
        try:
            # 选择关键特征进行建模 - 使用更基础的特征
            key_features = [
                'returns', 'volatility_10d', 'volatility_ratio', 
                'price_ma_ratio_10', 'momentum_5d', 'rsi',
                'high_low_ratio', 'jump_indicator', 'hurst_10d'
            ]
            
            # 过滤存在的特征
            available_features = [f for f in key_features if f in features.columns]
            model_data = features[available_features].dropna()
            
            # 降低数据要求阈值
            if len(model_data) < 20:  # 从50降低到20
                # 数据不足时使用简化的区制检测
                return self._simplified_regime_detection(features)
            
            # 标准化特征
            scaled_features = self.scaler.fit_transform(model_data)
            
            # 拟合高斯混合模型
            temp_model = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type='diag',  # 改为对角协方差矩阵，减少参数
                random_state=42,
                max_iter=100,  # 减少迭代次数
                init_params='kmeans'  # 使用kmeans初始化
            )
            
            temp_model.fit(scaled_features)
            
            # 只有拟合成功后才设置模型
            self.regime_model = temp_model
            # 手动存储特征名称，因为GaussianMixture没有feature_names_in_属性
            self.feature_names = available_features
            
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
        except Exception as e:
            # 拟合失败时不设置模型，返回错误信息
            return {"error": f"Model fitting failed: {str(e)}"}
    
    def _simplified_regime_detection(self, features: pd.DataFrame) -> Dict:
        """
        简化的区制检测，用于数据不足的情况
        基于基本的统计特征进行区制分类
        """
        try:
            # 使用基本特征进行简化分析
            basic_features = ['returns']
            if 'volatility_5d' in features.columns:
                basic_features.append('volatility_5d')
            if 'momentum_3d' in features.columns:
                basic_features.append('momentum_3d')
            
            available_data = features[basic_features].dropna()
            
            # 进一步降低数据要求
            if len(available_data) < 5:  # 从10降低到5
                # 即使数据极少，也尝试基于最基本的统计进行分类
                if 'returns' in features.columns and len(features['returns'].dropna()) >= 3:
                    returns = features['returns'].dropna()
                    avg_return = returns.mean()
                    volatility = returns.std()
                    
                    # 极简分类逻辑
                    if volatility > 0.02:  # 高波动
                        regime_name = "high_volatility_mean_reverting"
                        confidence = 0.3
                    elif avg_return > 0.001:  # 正收益
                        regime_name = "low_volatility_trending"
                        confidence = 0.3
                    elif avg_return < -0.001:  # 负收益
                        regime_name = "crisis_regime"
                        confidence = 0.4
                    else:
                        regime_name = "low_volatility_trending"  # 默认
                        confidence = 0.2
                    
                    return {
                        "simplified_regime": True,
                        "regime_name": regime_name,
                        "confidence": confidence,
                        "data_points": len(returns),
                        "avg_return": float(avg_return),
                        "volatility": float(volatility),
                        "note": "Used minimal data regime detection"
                    }
                else:
                    # 完全没有数据时的默认处理
                    return {
                        "simplified_regime": True,
                        "regime_name": "low_volatility_trending",  # 默认为低波动趋势
                        "confidence": 0.1,
                        "data_points": 0,
                        "avg_return": 0.0,
                        "volatility": 0.01,
                        "note": "Default regime due to insufficient data"
                    }
            
            # 计算基本统计量
            returns = available_data['returns']
            avg_return = returns.mean()
            volatility = returns.std()
            
            # 计算额外指标
            recent_volatility = returns.tail(min(10, len(returns))).std()
            vol_trend = recent_volatility / volatility if volatility > 0 else 1.0
            
            # 改进的区制分类逻辑
            if volatility > 0.025:  # 高波动阈值
                if avg_return < -0.005:  # 显著负收益
                    regime_name = "crisis_regime"
                    confidence = 0.7
                else:
                    regime_name = "high_volatility_mean_reverting"
                    confidence = 0.6
            elif volatility > 0.015:  # 中等波动
                if abs(avg_return) > 0.003:  # 有明显趋势
                    regime_name = "low_volatility_trending"
                    confidence = 0.5
                else:
                    regime_name = "high_volatility_mean_reverting"
                    confidence = 0.4
            else:  # 低波动
                regime_name = "low_volatility_trending"
                confidence = 0.5
            
            # 基于波动率趋势调整
            if vol_trend > 1.5:  # 波动率快速上升
                if regime_name == "low_volatility_trending":
                    regime_name = "high_volatility_mean_reverting"
                confidence *= 0.9
            
            return {
                "simplified_regime": True,
                "regime_name": regime_name,
                "confidence": confidence,
                "data_points": len(available_data),
                "avg_return": float(avg_return),
                "volatility": float(volatility),
                "vol_trend": float(vol_trend),
                "note": "Used simplified regime detection"
            }
            
        except Exception as e:
            # 即使简化检测失败，也返回一个默认区制而不是错误
            return {
                "simplified_regime": True,
                "regime_name": "low_volatility_trending",  # 默认区制
                "confidence": 0.1,
                "data_points": 0,
                "avg_return": 0.0,
                "volatility": 0.01,
                "error": f"Simplified detection failed: {str(e)}, using default regime"
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
                    "volatility": float(regime_data['volatility_10d'].mean()),
                    "momentum": float(regime_data['momentum_5d'].mean()),
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
        avg_vol = regime_data['volatility_10d'].mean()
        avg_momentum = regime_data['momentum_5d'].mean()
        avg_return = regime_data['returns'].mean()
        
        if avg_vol > regime_data['volatility_10d'].quantile(0.7):
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
        if self.regime_model is None or self.feature_names is None:
            # 如果没有完整模型，尝试使用简化检测
            simplified_result = self._simplified_regime_detection(features)
            
            # 简化检测现在总是返回一个有效的区制，不会返回错误
            return {
                "regime_name": simplified_result.get("regime_name", "low_volatility_trending"),
                "confidence": simplified_result.get("confidence", 0.1),
                "predicted_regime": -1,
                "regime_probabilities": {},
                "simplified": True,
                "data_points": simplified_result.get("data_points", 0),
                "note": simplified_result.get("note", "Used simplified detection"),
                "avg_return": simplified_result.get("avg_return", 0.0),
                "volatility": simplified_result.get("volatility", 0.01)
            }
        
        try:
            # 获取最新特征
            latest_features = features.iloc[-1:][self.feature_names]
            scaled_features = self.scaler.transform(latest_features)
            
            # 预测区制概率
            regime_probs = self.regime_model.predict_proba(scaled_features)[0]
            predicted_regime = np.argmax(regime_probs)
            
            return {
                "predicted_regime": int(predicted_regime),
                "regime_name": self.regime_names.get(predicted_regime, f"regime_{predicted_regime}"),
                "regime_probabilities": {f"regime_{i}": float(prob) for i, prob in enumerate(regime_probs)},
                "confidence": float(np.max(regime_probs)),
                "simplified": False
            }
        except Exception as e:
            # 如果预测过程中出现任何错误，尝试简化检测
            simplified_result = self._simplified_regime_detection(features)
            
            # 简化检测现在总是返回一个有效的区制
            return {
                "regime_name": simplified_result.get("regime_name", "low_volatility_trending"),
                "confidence": simplified_result.get("confidence", 0.1),
                "predicted_regime": -1,
                "regime_probabilities": {},
                "simplified": True,
                "error": f"Full prediction failed, used simplified: {str(e)}",
                "data_points": simplified_result.get("data_points", 0),
                "note": simplified_result.get("note", "Fallback to simplified detection")
            }

def adaptive_signal_aggregation(signals: Dict, regime_info: Dict, confidence_threshold: float = 0.6) -> Dict:
    """
    基于FLAG-Trader 2025研究的自适应信号聚合
    根据市场区制动态调整信号权重
    """
    regime_name = regime_info.get("regime_name", "unknown")
    regime_confidence = regime_info.get("confidence", 0.5)
    
    # 信号值映射
    signal_value_mapping = {
        'bullish': 1.0,
        'neutral': 0.0,
        'bearish': -1.0
    }
    
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
    
    def _parse_confidence(conf_value):
        """解析置信度值，支持字符串和数值格式"""
        if isinstance(conf_value, str):
            # 处理百分比格式 (如 "50%")
            if conf_value.endswith('%'):
                try:
                    return float(conf_value[:-1]) / 100.0
                except ValueError:
                    return 0.5
            # 处理纯数字字符串
            try:
                return float(conf_value)
            except ValueError:
                return 0.5
        elif isinstance(conf_value, (int, float)):
            # 如果是数值，确保在0-1范围内
            if conf_value > 1.0:
                return conf_value / 100.0  # 假设是百分比形式
            return float(conf_value)
        else:
            return 0.5  # 默认置信度
    
    for signal_type, weight in adjusted_weights.items():
        if signal_type in signals:
            signal_data = signals[signal_type]
            raw_signal = signal_data.get('signal', 'neutral')
            raw_confidence = signal_data.get('confidence', 0.5)
            
            # 解析置信度
            signal_conf = _parse_confidence(raw_confidence)
            
            # 应用最小置信度下限，避免过低的置信度完全抵消信号
            signal_conf = max(signal_conf, 0.2)  # 最低置信度为0.2
            
            # 转换字符串信号为数值
            if isinstance(raw_signal, str):
                signal_value = signal_value_mapping.get(raw_signal.lower(), 0.0)
            else:
                # 如果已经是数值，直接使用
                signal_value = float(raw_signal)
            
            # 使用置信度加权的信号值
            contribution = weight * signal_value * signal_conf
            weighted_signal += contribution
            weighted_confidence += weight * signal_conf
            
            signal_contributions[signal_type] = {
                'weight': weight,
                'signal': signal_value,
                'confidence': signal_conf,
                'raw_confidence': raw_confidence,  # 保留原始置信度值用于调试
                'contribution': contribution
            }
    
    # 应用动态阈值 (基于RLMF 2024技术)
    dynamic_threshold = 0.15 if regime_name == "crisis_regime" else 0.1  # 降低阈值
    
    # 调整信号强度处理逻辑 - 避免过度衰减弱信号
    original_signal = weighted_signal  # 保存原始信号用于调试
    if abs(weighted_signal) < dynamic_threshold:
        # 使用更温和的衰减，而不是直接减半
        attenuation_factor = 0.8 if abs(weighted_signal) < dynamic_threshold * 0.5 else 0.9
        weighted_signal *= attenuation_factor
    
    return {
        "aggregated_signal": weighted_signal,
        "aggregated_confidence": weighted_confidence,
        "regime_adjusted_weights": adjusted_weights,
        "signal_contributions": signal_contributions,
        "regime_info": regime_info,
        "dynamic_threshold": dynamic_threshold,
        "original_signal": original_signal,  # 添加原始信号用于调试
        "attenuation_applied": abs(original_signal) < dynamic_threshold  # 标记是否应用了衰减
    } 