import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
from src.utils.logging_config import setup_logger
from typing import Dict, Tuple, List, Optional, Any

# 设置日志
logger = setup_logger('deep_learning')

# 默认预测的未来天数 (可被函数参数覆盖)
DEFAULT_FORECAST_DAYS = 10 

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)


class LSTMModel(nn.Module):
    """通用LSTM模型，用于加载保存的模型"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int, dropout: float = 0.2):
        """
        初始化模型
        Args:
            input_dim: 输入特征数量
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            output_dim: 输出维度 (通常为预测天数)
            dropout: Dropout概率
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """前向传播"""
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM前向传播
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # 使用最后一个时间步的隐藏状态
        out = self.fc(lstm_out[:, -1, :])
        return out

# 为了向后兼容，保留原类名别名
StockLSTM = LSTMModel


class DeepLearningModule:
    """深度学习管理类，负责模型训练、评估和预测"""
    
    def __init__(self, model_dir: str = 'models'):
        """
        初始化深度学习模块
        
        Args:
            model_dir: 模型保存目录
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"深度学习模块初始化完成，使用设备: {self.device}")
        
        # 保存数据处理器和模型
        self.price_scaler = None
        self.feature_scaler = None
        self.lstm_model = None
        self.rf_model = None
    
    def _create_lstm_sequences(self, data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建LSTM输入序列
        
        Args:
            data: 输入特征数据
            seq_length: 序列长度(时间窗口)
            
        Returns:
            X: 输入序列
            y: 目标值 (下一个时间点)
        """
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length, 0])  # 只预测第一列 (通常是收盘价)
        return np.array(X), np.array(y).reshape(-1, 1)
    
    def train_lstm_model(self, 
                         price_data: pd.DataFrame, 
                         target_col: str = 'close',
                         feature_cols: Optional[List[str]] = None,
                         seq_length: int = 10,
                         forecast_days: int = 10,
                         hidden_dim: int = 128,
                         num_layers: int = 3,
                         epochs: int = 50,
                         batch_size: int = 16,
                         learning_rate: float = 0.001):
        """
        训练LSTM模型用于价格预测
        
        Args:
            price_data: 包含价格数据的DataFrame
            target_col: 目标列名(通常是收盘价)
            feature_cols: 特征列名列表(为None时使用target_col)
            seq_length: 输入序列长度
            forecast_days: 预测天数 (默认为10)
            hidden_dim: LSTM隐藏层维度 (默认为128)
            num_layers: LSTM层数 (默认为3)
            epochs: 训练轮数
            batch_size: 批量大小
            learning_rate: 学习率
        
        Returns:
            训练后的模型
        """
        try:
            logger.info("开始训练LSTM模型...")
            
            # 如果feature_cols为None，只使用target_col
            if feature_cols is None:
                feature_cols = [target_col]
            
            # 记录使用的特征列
            logger.info(f"LSTM训练使用特征列: {feature_cols}")
            
            # 确保所有特征都在DataFrame中
            missing_features = [col for col in feature_cols if col not in price_data.columns]
            if missing_features:
                raise ValueError(f"以下特征列在数据中不存在: {missing_features}")
            
            # 提取特征数据
            data = price_data[feature_cols].values
            
            # 数据归一化
            self.price_scaler = MinMaxScaler()
            data_scaled = self.price_scaler.fit_transform(data)
            
            # 创建序列数据
            X, y = self._create_lstm_sequences(data_scaled, seq_length)
            
            # 划分训练集和验证集
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            # 转换为PyTorch张量
            X_train = torch.FloatTensor(X_train)
            y_train = torch.FloatTensor(y_train)
            X_val = torch.FloatTensor(X_val)
            y_val = torch.FloatTensor(y_val)
            
            # 创建数据加载器
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # 初始化模型
            input_dim = len(feature_cols)
            logger.info(f"初始化LSTM模型，输入维度: {input_dim}, 隐藏层维度: {hidden_dim}, 层数: {num_layers}")
            self.lstm_model = LSTMModel(input_dim, hidden_dim, num_layers, forecast_days)
            self.lstm_model.to(self.device)
            
            # 定义优化器和损失函数
            optimizer = optim.Adam(self.lstm_model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            
            # 训练模型
            self.lstm_model.train()
            for epoch in range(epochs):
                total_loss = 0
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    # 前向传播
                    optimizer.zero_grad()
                    outputs = self.lstm_model(batch_X)
                    
                    # 计算损失
                    # 使用第一个预测值与实际值比较
                    loss = criterion(outputs[:, 0].unsqueeze(1), batch_y)
                    
                    # 反向传播
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                # 验证
                if epoch % 5 == 0:
                    self.lstm_model.eval()
                    with torch.no_grad():
                        val_outputs = self.lstm_model(X_val.to(self.device))
                        val_loss = criterion(val_outputs[:, 0].unsqueeze(1), y_val.to(self.device))
                    
                    logger.info(f"Epoch {epoch+1}/{epochs}, Training Loss: {total_loss/len(train_loader):.4f}, Validation Loss: {val_loss.item():.4f}")
                    self.lstm_model.train()
            
            # 保存模型
            torch.save(self.lstm_model.state_dict(), f"{self.model_dir}/lstm_model.pth")
            joblib.dump(self.price_scaler, f"{self.model_dir}/price_scaler.pkl")
            
            logger.info("LSTM模型训练完成")
            return self.lstm_model
        
        except Exception as e:
            logger.error(f"训练LSTM模型时出错: {str(e)}")
            raise
    
    def predict_lstm(self, price_data: pd.DataFrame, feature_cols: Optional[List[str]] = None,
                    seq_length: int = 10, target_col: str = 'close') -> np.ndarray:
        """
        使用LSTM模型预测未来价格
        
        Args:
            price_data: 包含历史价格数据的DataFrame
            feature_cols: 特征列名列表
            seq_length: 输入序列长度
            target_col: 目标列名
            
        Returns:
            未来价格预测结果
        """
        if self.lstm_model is None:
            raise ValueError("LSTM模型未训练，请先调用train_lstm_model方法")
        
        try:
            # 如果feature_cols为None，获取price_scaler的特征数量
            if feature_cols is None:
                # 确定price_scaler中的特征数量
                n_features = self.price_scaler.n_features_in_
                logger.info(f"从price_scaler检测到特征数量: {n_features}")
                
                # 根据特征数量确定要使用的特征列
                if n_features == 1:
                    feature_cols = [target_col]
                elif n_features == 6:
                    feature_cols = ['close', 'ma5', 'ma10', 'ma20', 'rsi', 'macd']
                else:
                    # 如果不确定使用哪些特征，尝试从DataFrame中选择足够的特征
                    all_numeric_cols = price_data.select_dtypes(include=np.number).columns.tolist()
                    feature_cols = all_numeric_cols[:n_features]
                    logger.info(f"自动选择特征列: {feature_cols}")
            
            # 确保所有特征都存在
            missing_features = [col for col in feature_cols if col not in price_data.columns]
            if missing_features:
                logger.warning(f"以下特征在数据中缺失: {missing_features}")
                # 尝试生成缺失的特征
                for col in missing_features:
                    if col == 'ma5':
                        price_data['ma5'] = price_data['close'].rolling(window=5).mean()
                    elif col == 'ma10':
                        price_data['ma10'] = price_data['close'].rolling(window=10).mean()
                    elif col == 'ma20':
                        price_data['ma20'] = price_data['close'].rolling(window=20).mean()
                    elif col == 'rsi':
                        delta = price_data['close'].diff()
                        gain = (delta.where(delta > 0, 0)).fillna(0)
                        loss = (-delta.where(delta < 0, 0)).fillna(0)
                        avg_gain = gain.rolling(window=14).mean()
                        avg_loss = loss.rolling(window=14).mean()
                        rs = avg_gain / avg_loss
                        price_data['rsi'] = 100 - (100 / (1 + rs))
                    elif col == 'macd':
                        ema12 = price_data['close'].ewm(span=12, adjust=False).mean()
                        ema26 = price_data['close'].ewm(span=26, adjust=False).mean()
                        price_data['macd'] = ema12 - ema26
            
            # 填充NaN值
            price_data = price_data.ffill().bfill()
            
            # 提取最新的序列数据
            data = price_data[feature_cols].values[-seq_length:]
            
            # 数据归一化
            data_scaled = self.price_scaler.transform(data)
            
            # 转换为模型输入格式
            X = torch.FloatTensor(data_scaled).unsqueeze(0).to(self.device)
            
            # 预测
            self.lstm_model.eval()
            with torch.no_grad():
                predictions = self.lstm_model(X)
            
            # 转换回原始尺度
            # 创建一个与原始特征数量相同的零数组
            zeros = np.zeros((predictions.shape[1], len(feature_cols)))
            # 将预测值放入第一列(收盘价列)
            zeros[:, 0] = predictions.cpu().numpy()[0]
            predictions_rescaled = self.price_scaler.inverse_transform(zeros)
            
            # 仅返回收盘价列的预测结果
            return predictions_rescaled[:, 0]
            
        except Exception as e:
            logger.error(f"LSTM预测时出错: {str(e)}")
            raise
    
    def train_stock_classifier(self, features: pd.DataFrame, labels: pd.Series,
                              n_estimators: int = 200, random_state: int = 42):
        """
        训练股票分类器，用于选股
        
        Args:
            features: 特征DataFrame
            labels: 标签Series (1表示上涨，0表示下跌)
            n_estimators: 决策树数量
            random_state: 随机种子
            
        Returns:
            训练后的模型
        """
        try:
            logger.info("开始训练随机森林分类器...")
            
            # 检查类别分布
            class_distribution = labels.value_counts(normalize=True)
            logger.info(f"原始类别分布: {class_distribution.to_dict()}")
            
            # 确定是否需要平衡类别
            class_weight = None
            if len(class_distribution) > 1:
                ratio_0 = class_distribution.get(0, 0)
                ratio_1 = class_distribution.get(1, 0)
                
                if ratio_0 > 0.65 or ratio_1 > 0.65:
                    logger.info("检测到类别严重不平衡，将应用平衡权重")
                    # 自定义类别权重，确保模型不会总是预测多数类
                    class_weight = {
                        0: 1.0 / (ratio_0 if ratio_0 > 0 else 0.5),
                        1: 1.0 / (ratio_1 if ratio_1 > 0 else 0.5)
                    }
                    logger.info(f"应用的类别权重: {class_weight}")
            
            # 数据标准化
            self.feature_scaler = StandardScaler()
            features_scaled = self.feature_scaler.fit_transform(features)
            
            # 划分训练集和验证集
            X_train, X_val, y_train, y_val = train_test_split(
                features_scaled, labels, test_size=0.2, random_state=random_state, stratify=labels
            )
            
            # 训练随机森林分类器
            self.rf_model = RandomForestClassifier(
                n_estimators=n_estimators, 
                random_state=random_state, 
                n_jobs=-1,
                max_depth=6,  # 限制树的深度，减少过拟合
                min_samples_leaf=5,  # 确保叶节点有足够的样本
                class_weight=class_weight,
                max_features='sqrt'  # 使用特征子集，增加多样性
            )
            self.rf_model.fit(X_train, y_train)
            
            # 验证
            train_accuracy = self.rf_model.score(X_train, y_train)
            val_accuracy = self.rf_model.score(X_val, y_val)
            
            # 验证集上的类别预测比例
            y_val_pred = self.rf_model.predict(X_val)
            val_pred_distribution = pd.Series(y_val_pred).value_counts(normalize=True)
            logger.info(f"验证集预测类别分布: {val_pred_distribution.to_dict()}")
            
            logger.info(f"分类器训练完成，训练准确率: {train_accuracy:.4f}, 验证准确率: {val_accuracy:.4f}")
            
            # 特征重要性
            feature_importance = {
                features.columns[i]: importance 
                for i, importance in enumerate(self.rf_model.feature_importances_)
            }
            logger.info(f"特征重要性: {sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]}")
            
            # 保存模型
            joblib.dump(self.rf_model, f"{self.model_dir}/rf_classifier.pkl")
            joblib.dump(self.feature_scaler, f"{self.model_dir}/feature_scaler.pkl")
            
            return self.rf_model
            
        except Exception as e:
            logger.error(f"训练股票分类器时出错: {str(e)}")
            raise
    
    def predict_stock_returns(self, features: pd.DataFrame) -> Dict[str, float]:
        """
        预测股票分类结果和概率
        
        Args:
            features: 特征DataFrame
            
        Returns:
            包含预测分类和概率的字典
        """
        if self.rf_model is None:
            raise ValueError("随机森林模型未训练，请先调用train_stock_classifier方法")
        
        try:
            # 数据标准化
            features_scaled = self.feature_scaler.transform(features)
            
            # 预测类别和概率
            predictions = self.rf_model.predict(features_scaled)
            probabilities = self.rf_model.predict_proba(features_scaled)
            
            # 获取原始概率
            raw_probability_up = probabilities[0][1]  # 上涨概率
            raw_probability_down = probabilities[0][0]  # 下跌概率
            
            # 调整极端概率，防止过于自信的预测
            calibrated_prob_up = min(max(raw_probability_up, 0.05), 0.95)
            calibrated_prob_down = min(max(raw_probability_down, 0.05), 0.95)
            
            # 重新归一化概率
            total = calibrated_prob_up + calibrated_prob_down
            calibrated_prob_up = calibrated_prob_up / total
            calibrated_prob_down = calibrated_prob_down / total
            
            # 记录原始和校准后的概率
            logger.info(f"原始预测概率 - 上涨: {raw_probability_up:.4f}, 下跌: {raw_probability_down:.4f}")
            logger.info(f"校准后概率 - 上涨: {calibrated_prob_up:.4f}, 下跌: {calibrated_prob_down:.4f}")
            
            # 返回预测结果
            return {
                "prediction": predictions[0],  # 1表示上涨，0表示下跌
                "probability": calibrated_prob_up if predictions[0] == 1 else calibrated_prob_down,
                "up_probability": calibrated_prob_up,
                "down_probability": calibrated_prob_down,
                "expected_return": (calibrated_prob_up - 0.5) * 2  # 转换为-1到1的范围
            }
            
        except Exception as e:
            logger.error(f"预测股票分类时出错: {str(e)}")
            raise
    
    def load_models(self, lstm_path: Optional[str] = None, rf_path: Optional[str] = None,
                   price_scaler_path: Optional[str] = None, feature_scaler_path: Optional[str] = None):
        """
        加载已训练的模型
        
        Args:
            lstm_path: LSTM模型路径
            rf_path: 随机森林模型路径
            price_scaler_path: 价格标准化器路径
            feature_scaler_path: 特征标准化器路径
        """
        try:
            # 设置默认路径
            if lstm_path is None:
                lstm_path = f"{self.model_dir}/lstm_model.pth"
            if rf_path is None:
                rf_path = f"{self.model_dir}/rf_classifier.pkl"
            if price_scaler_path is None:
                price_scaler_path = f"{self.model_dir}/price_scaler.pkl"
            if feature_scaler_path is None:
                feature_scaler_path = f"{self.model_dir}/feature_scaler.pkl"
            
            # 加载LSTM模型
            if os.path.exists(lstm_path) and os.path.exists(price_scaler_path):
                # 先加载标准化器以获取特征维度
                self.price_scaler = joblib.load(price_scaler_path)
                input_dim = getattr(self.price_scaler, 'n_features_in_', 1)
                
                logger.info(f"从price_scaler检测到特征维度: {input_dim}")
                
                # 需要先定义模型结构，使用正确的输入维度
                self.lstm_model = LSTMModel(input_dim=input_dim, hidden_dim=128, num_layers=3, output_dim=10)
                
                # 添加尝试加载模型的代码
                try:
                    state_dict = torch.load(lstm_path, map_location=self.device)
                    logger.info(f"开始尝试加载模型参数，检测到以下参数: {list(state_dict.keys())}")
                    
                    # 手动创建具有正确结构的新模型
                    # 首先确定模型的关键参数
                    # 检查第一层参数的形状来确定真实的隐藏层大小
                    if 'lstm.weight_ih_l0' in state_dict:
                        # LSTM参数的形状是 [4*hidden_size, input_dim]
                        lstm_param_shape = state_dict['lstm.weight_ih_l0'].shape
                        actual_hidden_size = lstm_param_shape[0] // 4
                        actual_input_dim = lstm_param_shape[1]
                        logger.info(f"从模型参数推断: 隐藏层大小={actual_hidden_size}, 输入维度={actual_input_dim}")
                        
                        # 检查输出层大小
                        if 'fc.weight' in state_dict:
                            fc_param_shape = state_dict['fc.weight'].shape
                            actual_output_dim = fc_param_shape[0]
                            logger.info(f"从模型参数推断: 输出维度={actual_output_dim}")
                        else:
                            actual_output_dim = 10  # 默认值
                        
                        # 检查层数
                        actual_num_layers = 1
                        while f'lstm.weight_ih_l{actual_num_layers}' in state_dict:
                            actual_num_layers += 1
                        logger.info(f"从模型参数推断: LSTM层数={actual_num_layers}")
                        
                        # 使用正确的参数重新创建模型
                        logger.info(f"使用推断的参数重新创建模型: input_dim={actual_input_dim}, hidden_dim={actual_hidden_size}, num_layers={actual_num_layers}, output_dim={actual_output_dim}")
                        self.lstm_model = LSTMModel(
                            input_dim=actual_input_dim,
                            hidden_dim=actual_hidden_size,
                            num_layers=actual_num_layers,
                            output_dim=actual_output_dim
                        )
                        
                        # 尝试加载现在应该匹配的状态字典
                        self.lstm_model.load_state_dict(state_dict)
                        logger.info("✓ LSTM模型加载成功，使用从参数推断的模型结构")
                    else:
                        # 如果找不到关键参数，则使用默认值重试
                        logger.warning("无法从参数推断模型结构，使用默认参数")
                        self.lstm_model = LSTMModel(input_dim=input_dim, hidden_dim=128, num_layers=3, output_dim=10)
                        self.lstm_model.load_state_dict(state_dict)
                        logger.info("使用默认参数加载模型成功")
                except Exception as e:
                    logger.error(f"加载模型最终失败: {str(e)}")
                    logger.warning("将使用默认模型初始化，但不会加载预训练权重")
                    self.lstm_model = LSTMModel(input_dim=input_dim, hidden_dim=128, num_layers=3, output_dim=10)
                
                self.lstm_model.to(self.device)
                self.lstm_model.eval()
                
                logger.info("LSTM模型加载处理完成")
            else:
                logger.warning("LSTM模型或标准化器文件不存在")
            
            # 加载随机森林模型
            if os.path.exists(rf_path) and os.path.exists(feature_scaler_path):
                self.rf_model = joblib.load(rf_path)
                self.feature_scaler = joblib.load(feature_scaler_path)
                logger.info("随机森林模型加载成功")
            else:
                logger.warning("随机森林模型或标准化器文件不存在")
                
        except Exception as e:
            logger.error(f"加载模型时出错: {str(e)}")
            raise

    def prepare_features(self, price_data: pd.DataFrame, technical_indicators: Dict[str, pd.Series] = None) -> pd.DataFrame:
        """
        准备模型特征
        
        Args:
            price_data: 价格数据DataFrame
            technical_indicators: 技术指标字典
            
        Returns:
            特征DataFrame
        """
        # 创建特征DataFrame
        features = pd.DataFrame()
        
        # 添加价格特征
        if 'close' in price_data.columns:
            # 价格变化率 (1天、3天、5天)
            features['price_change_1d'] = price_data['close'].pct_change(1)
            features['price_change_3d'] = price_data['close'].pct_change(3)
            features['price_change_5d'] = price_data['close'].pct_change(5)
            
            # 波动率 (10天、20天)
            features['volatility_10d'] = price_data['close'].pct_change().rolling(10).std()
            features['volatility_20d'] = price_data['close'].pct_change().rolling(20).std()
        
        # 添加量价特征
        if 'volume' in price_data.columns:
            # 成交量变化率
            features['volume_change_1d'] = price_data['volume'].pct_change(1)
            features['volume_change_5d'] = price_data['volume'].pct_change(5)
            
            # 量价关系
            if 'close' in price_data.columns:
                features['price_volume_corr'] = price_data['close'].rolling(10).corr(price_data['volume'])
        
        # 添加技术指标
        if technical_indicators:
            for name, indicator in technical_indicators.items():
                features[name] = indicator
        
        # 处理缺失值
        features = features.dropna()
        
        return features
    
    def generate_training_labels(self, price_data: pd.DataFrame, forward_days: int = 5,
                                return_threshold: float = 0.01) -> pd.Series:
        """
        生成训练标签
        
        Args:
            price_data: 价格数据DataFrame
            forward_days: 未来天数
            return_threshold: 收益率阈值
            
        Returns:
            标签Series (1表示上涨超过阈值，0表示下跌低于阈值的负值)
        """
        # 计算未来收益率
        future_returns = price_data['close'].shift(-forward_days) / price_data['close'] - 1
        
        # 生成标签 (1表示上涨超过阈值，0表示下跌低于负的阈值)
        # 使用对称的阈值，使得上涨和下跌的机会更均衡
        labels = (future_returns > return_threshold).astype(int)
        
        # 统计标签分布并输出日志
        label_counts = labels.value_counts()
        up_ratio = label_counts.get(1, 0) / len(labels) if len(labels) > 0 else 0
        down_ratio = label_counts.get(0, 0) / len(labels) if len(labels) > 0 else 0
        logger.info(f"标签分布 - 上涨: {up_ratio:.2%}, 下跌: {down_ratio:.2%}")
        
        return labels

# 模型包装器，用于集成到Agent系统中
class MLAgent:
    """机器学习Agent，负责提供预测信号给Portfolio Manager"""
    
    def __init__(self, model_dir: str = 'models'):
        """初始化ML Agent"""
        self.dl_module = DeepLearningModule(model_dir)
        self.is_trained = False
        self.logger = setup_logger('ml_agent')
    
    def train_models(self, price_data: pd.DataFrame, technical_indicators: Dict[str, pd.Series] = None,
                  seq_length: int = 10, feature_cols: List[str] = None, hidden_dim: int = 128,
                  num_layers: int = 3, forecast_days: int = 10, epochs: int = 50,
                  batch_size: int = 16, learning_rate: float = 0.001):
        """
        训练所有模型
        
        Args:
            price_data: 价格数据
            technical_indicators: 技术指标
            seq_length: 序列长度
            feature_cols: 特征列名列表
            hidden_dim: LSTM隐藏层维度 (默认为128)
            num_layers: LSTM层数 (默认为3)
            forecast_days: 预测天数 (默认为10)
            epochs: 训练轮数
            batch_size: 批量大小
            learning_rate: 学习率
        """
        try:
            # 如果feature_cols为None，使用默认值
            if feature_cols is None:
                feature_cols = ['close', 'ma5', 'ma10', 'ma20', 'rsi', 'macd']
                self.logger.info(f"未指定特征列，使用默认特征: {feature_cols}")
            
            # 确保特征列在数据中存在，否则添加它们
            missing_features = [col for col in feature_cols if col not in price_data.columns]
            if missing_features:
                self.logger.warning(f"数据中缺少以下特征，将自动添加: {missing_features}")
                for col in missing_features:
                    if col == 'ma5':
                        price_data['ma5'] = price_data['close'].rolling(window=5).mean()
                    elif col == 'ma10':
                        price_data['ma10'] = price_data['close'].rolling(window=10).mean()
                    elif col == 'ma20':
                        price_data['ma20'] = price_data['close'].rolling(window=20).mean()
                    elif col == 'rsi':
                        delta = price_data['close'].diff()
                        gain = (delta.where(delta > 0, 0)).fillna(0)
                        loss = (-delta.where(delta < 0, 0)).fillna(0)
                        avg_gain = gain.rolling(window=14).mean()
                        avg_loss = loss.rolling(window=14).mean()
                        rs = avg_gain / avg_loss
                        price_data['rsi'] = 100 - (100 / (1 + rs))
                    elif col == 'macd':
                        ema12 = price_data['close'].ewm(span=12, adjust=False).mean()
                        ema26 = price_data['close'].ewm(span=26, adjust=False).mean()
                        price_data['macd'] = ema12 - ema26
            
            # 填充NaN值
            price_data = price_data.ffill().bfill()
            
            self.logger.info(f"开始训练模型，使用参数: seq_length={seq_length}, hidden_dim={hidden_dim}, "
                           f"num_layers={num_layers}, epochs={epochs}, forecast_days={forecast_days}")
            self.logger.info(f"使用特征列: {feature_cols}")
            
            # 准备LSTM训练数据，使用传入的参数
            self.dl_module.train_lstm_model(
                price_data=price_data,
                target_col='close',
                feature_cols=feature_cols,
                seq_length=seq_length,
                forecast_days=forecast_days,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            
            # 准备分类器训练数据
            features = self.dl_module.prepare_features(price_data, technical_indicators)
            labels = self.dl_module.generate_training_labels(price_data, forward_days=forecast_days)
            
            # 确保特征和标签具有相同的索引
            common_idx = features.index.intersection(labels.index)
            if len(common_idx) > 0:
                self.logger.info(f"训练随机森林分类器，使用{len(common_idx)}个样本")
                self.dl_module.train_stock_classifier(
                    features=features.loc[common_idx],
                    labels=labels.loc[common_idx],
                    n_estimators=200
                )
                self.is_trained = True
            else:
                self.logger.warning("特征和标签没有共同的索引，无法训练分类器")
            
        except Exception as e:
            self.logger.error(f"训练模型时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def load_models(self):
        """加载已训练的模型"""
        try:
            self.dl_module.load_models()
            self.is_trained = True
        except Exception as e:
            self.logger.error(f"加载模型时出错: {str(e)}")
    
    def generate_signals(self, price_data: pd.DataFrame, 
                         technical_indicators: Dict[str, pd.Series] = None) -> Dict[str, Any]:
        """
        生成交易信号
        
        Args:
            price_data: 价格数据
            technical_indicators: 技术指标
            
        Returns:
            包含交易信号的字典
        """
        signals = {}
        
        try:
            # 加载模型(如果尚未加载)
            if not self.is_trained:
                self.load_models()
            
            # 使用LSTM预测未来价格
            lstm_predictions = None
            if self.dl_module.lstm_model is not None:
                # 确定预测天数
                forecast_days = self.dl_module.lstm_model.fc.out_features
                self.logger.info(f"使用LSTM模型预测未来{forecast_days}天价格")
                
                # 获取price_scaler的特征数量以确定正确的feature_cols
                n_features = self.dl_module.price_scaler.n_features_in_
                if n_features == 1:
                    lstm_feature_cols = ['close']
                elif n_features == 6:
                    lstm_feature_cols = ['close', 'ma5', 'ma10', 'ma20', 'rsi', 'macd']
                else:
                    # 尝试根据数据选择合适的特征
                    all_numeric_cols = price_data.select_dtypes(include=np.number).columns.tolist()
                    lstm_feature_cols = all_numeric_cols[:n_features]
                
                self.logger.info(f"LSTM预测将使用以下特征: {lstm_feature_cols}")
                
                lstm_predictions = self.dl_module.predict_lstm(
                    price_data=price_data,
                    feature_cols=lstm_feature_cols,
                    seq_length=10,
                    target_col='close'
                )
                
                # 计算预期收益率
                current_price = price_data['close'].iloc[-1]
                future_prices = lstm_predictions
                expected_returns = [price/current_price - 1 for price in future_prices]
                
                signals['lstm_predictions'] = {
                    'future_prices': future_prices.tolist(),
                    'expected_returns': expected_returns,
                    'forecast_days': forecast_days
                }
                
                # 计算预测增长率的统计信息
                avg_return = np.mean(expected_returns)
                median_return = np.median(expected_returns)
                positive_count = sum(1 for r in expected_returns if r > 0)
                negative_count = sum(1 for r in expected_returns if r < 0)
                
                self.logger.info(f"LSTM预测 - 平均收益率: {avg_return:.4f}, 中位数收益率: {median_return:.4f}")
                self.logger.info(f"LSTM预测 - 正收益天数: {positive_count}, 负收益天数: {negative_count}")
                
                # 基于LSTM预测生成信号，调整阈值和置信度计算
                positive_ratio = positive_count / len(expected_returns) if len(expected_returns) > 0 else 0.5
                
                # 使用更平衡的预测逻辑
                if avg_return > 0.02 and positive_ratio > 0.6:  # 正向信号阈值
                    signals['lstm_signal'] = 'bullish'
                    signals['lstm_confidence'] = min(0.5 + positive_ratio * 0.4, 0.9)  # 更温和的置信度
                elif avg_return < -0.02 and positive_ratio < 0.4:  # 负向信号阈值
                    signals['lstm_signal'] = 'bearish'
                    signals['lstm_confidence'] = min(0.5 + (1 - positive_ratio) * 0.4, 0.9)
                else:
                    signals['lstm_signal'] = 'neutral'
                    signals['lstm_confidence'] = 0.5
            
            # 使用分类器预测上涨/下跌概率
            rf_prediction = None
            if self.dl_module.rf_model is not None:
                # 准备特征
                features = self.dl_module.prepare_features(price_data, technical_indicators)
                if not features.empty:
                    # 获取最新的特征行
                    latest_features = features.iloc[[-1]]
                    rf_prediction = self.dl_module.predict_stock_returns(latest_features)
                    
                    signals['rf_prediction'] = rf_prediction
                    
                    # 获取上涨和下跌概率
                    up_probability = rf_prediction.get('up_probability', 0.5)
                    down_probability = rf_prediction.get('down_probability', 0.5)
                    
                    # 基于随机森林预测生成信号，应用更保守的阈值
                    if rf_prediction['prediction'] == 1:
                        signals['rf_signal'] = 'bullish'
                        signals['rf_confidence'] = rf_prediction.get('probability', up_probability)
                    else:
                        signals['rf_signal'] = 'bearish'
                        signals['rf_confidence'] = rf_prediction.get('probability', down_probability)
                    
                    # 记录预测结果
                    self.logger.info(f"RF预测 - 信号: {signals['rf_signal']}, 置信度: {signals['rf_confidence']:.4f}")
            
            # 结合两个模型生成最终信号
            if 'lstm_signal' in signals and 'rf_signal' in signals:
                # 简单加权平均，但调整权重以减少偏差
                lstm_weight = 0.5  # LSTM和随机森林权重相等
                rf_weight = 0.5
                
                lstm_score = {'bullish': 1, 'neutral': 0, 'bearish': -1}[signals['lstm_signal']] * signals['lstm_confidence']
                rf_score = {'bullish': 1, 'neutral': 0, 'bearish': -1}[signals['rf_signal']] * signals['rf_confidence']
                
                combined_score = lstm_weight * lstm_score + rf_weight * rf_score
                
                # 使用更保守的阈值
                if combined_score > 0.3:
                    signals['signal'] = 'bullish'
                    signals['confidence'] = min(abs(combined_score), 0.85)
                elif combined_score < -0.3:
                    signals['signal'] = 'bearish'
                    signals['confidence'] = min(abs(combined_score), 0.85)
                else:
                    signals['signal'] = 'neutral'
                    signals['confidence'] = 0.5
                
                self.logger.info(f"最终信号 - {signals['signal']}, 置信度: {signals['confidence']:.4f}, 组合得分: {combined_score:.4f}")
            elif 'lstm_signal' in signals:
                signals['signal'] = signals['lstm_signal']
                signals['confidence'] = signals['lstm_confidence']
            elif 'rf_signal' in signals:
                signals['signal'] = signals['rf_signal']
                signals['confidence'] = signals['rf_confidence']
            else:
                signals['signal'] = 'neutral'
                signals['confidence'] = 0.5
                
            # 添加处理逻辑
            signals['reasoning'] = self._generate_reasoning(signals)
            
        except Exception as e:
            self.logger.error(f"生成交易信号时出错: {str(e)}")
            signals['signal'] = 'neutral'
            signals['confidence'] = 0.5
            signals['error'] = str(e)
        
        return signals
    
    def _generate_reasoning(self, signals: Dict[str, Any]) -> str:
        """
        根据信号生成决策理由
        
        Args:
            signals: 信号字典
            
        Returns:
            决策理由
        """
        reasoning_parts = []
        
        # LSTM模型理由
        if 'lstm_predictions' in signals:
            future_prices = signals['lstm_predictions']['future_prices']
            expected_returns = signals['lstm_predictions']['expected_returns']
            forecast_days = signals['lstm_predictions'].get('forecast_days', DEFAULT_FORECAST_DAYS)
            
            avg_return = np.mean(expected_returns)
            positive_count = sum(1 for r in expected_returns if r > 0)
            negative_count = sum(1 for r in expected_returns if r < 0)
            
            positive_ratio = positive_count / len(expected_returns) if len(expected_returns) > 0 else 0
            
            short_term_return = expected_returns[0] if expected_returns else 0
            medium_term_avg = np.mean(expected_returns[1:min(5, len(expected_returns))]) if len(expected_returns) > 1 else avg_return
            long_term_avg = np.mean(expected_returns[min(5, len(expected_returns)):]) if len(expected_returns) > 5 else avg_return
            
            reasoning_parts.append(
                f"LSTM模型预测未来{forecast_days}天: 短期收益率({short_term_return:.2%}), 中期收益率({medium_term_avg:.2%}), 长期收益率({long_term_avg:.2%}). "
                f"预期正收益天数占比: {positive_ratio:.2%}"
            )
        
        # 随机森林模型理由
        if 'rf_prediction' in signals:
            pred = signals['rf_prediction']['prediction']
            up_prob = signals['rf_prediction'].get('up_probability', 0.5)
            down_prob = signals['rf_prediction'].get('down_probability', 0.5)
            
            if pred == 1:
                reasoning_parts.append(f"随机森林模型预测上涨概率: {up_prob:.2%}, 下跌概率: {down_prob:.2%}")
            else:
                reasoning_parts.append(f"随机森林模型预测下跌概率: {down_prob:.2%}, 上涨概率: {up_prob:.2%}")
        
        # 技术指标评估
        if 'lstm_signal' in signals:
            reasoning_parts.append(f"LSTM技术分析结果: {signals['lstm_signal']}, 置信度: {signals['lstm_confidence']:.2%}")
        
        if 'rf_signal' in signals:
            reasoning_parts.append(f"随机森林技术分析结果: {signals['rf_signal']}, 置信度: {signals['rf_confidence']:.2%}")
        
        # 组合策略理由
        if 'signal' in signals:
            if signals['signal'] == 'bullish':
                reasoning_parts.append(
                    f"综合分析产生看多信号，置信度: {signals['confidence']:.2%}. "
                    f"请注意，即使在看多信号下，市场仍存在风险，建议设置止损。"
                )
            elif signals['signal'] == 'bearish':
                reasoning_parts.append(
                    f"综合分析产生看空信号，置信度: {signals['confidence']:.2%}. "
                    f"请注意，即使在看空信号下，市场仍有可能反弹，建议适时止盈。"
                )
            else:
                reasoning_parts.append(
                    f"综合分析产生中性信号，置信度: {signals['confidence']:.2%}. "
                    f"市场方向不明确，建议谨慎操作或持币观望。"
                )
        
        return "; ".join(reasoning_parts)


def preprocess_stock_data(price_df: pd.DataFrame, technical_indicators: Optional[Dict] = None) -> pd.DataFrame:
    """
    预处理股票数据以供深度学习模型使用
    
    Args:
        price_df: 价格数据DataFrame
        technical_indicators: 技术指标字典
        
    Returns:
        处理后的DataFrame
    """
    # 确保数据为DataFrame
    if not isinstance(price_df, pd.DataFrame):
        raise ValueError("价格数据必须是pandas DataFrame")
    
    # 确保有必要的列
    required_cols = ['close']
    if not all(col in price_df.columns for col in required_cols):
        raise ValueError(f"价格数据必须包含以下列: {required_cols}")
    
    # 复制数据，避免修改原始数据
    df = price_df.copy()

    df.set_index('date', inplace=True)
    
    # 计算收益率
    df['returns'] = df['close'].pct_change()
    
    # 计算技术指标
    # 移动平均线
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    
    # MACD
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 添加外部技术指标
    if technical_indicators:
        for name, indicator in technical_indicators.items():
            if isinstance(indicator, pd.Series):
                df[name] = indicator
    
    # 添加滞后特征
    for lag in [1, 2, 3, 5, 10]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
    
    # 添加滚动统计量
    for window in [5, 10, 20]:
        df[f'volatility_{window}d'] = df['returns'].rolling(window=window).std()
        df[f'max_{window}d'] = df['close'].rolling(window=window).max()
        df[f'min_{window}d'] = df['close'].rolling(window=window).min()
    
    # 删除缺失值
    df = df.dropna()
    
    return df
