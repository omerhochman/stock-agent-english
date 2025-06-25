# 投资分析系统代理模块 (src/agents)

## 概述

本模块是基于多代理系统的 AI 投资分析框架的核心组件，实现了一个完整的投资研究和决策流水线。该系统采用现代投资组合理论、行为金融学和机器学习技术，提供全面的股票分析和投资建议。

## 系统架构

系统采用分布式代理架构，每个代理专注于特定的分析领域，最终通过智能聚合形成投资决策：

```
数据收集 → 多维分析 → AI模型预测 → 辩论评估 → 风险管理 → 投资组合决策
```

## 核心代理模块

### 1. 市场数据代理 (market_data.py)

**功能**: 负责收集和预处理所有市场相关数据

- **核心特性**:
  - 支持单一和多资产数据收集
  - 获取股价历史、财务指标、市场数据
  - 数据验证和清洗
  - 统一数据格式化

**主要方法**:

```python
@agent_endpoint("market_data", "市场数据收集，负责获取股价历史、财务指标和市场信息")
def market_data_agent(state: AgentState):
    # 收集多个股票的完整市场数据
    # 返回标准化的数据结构
```

### 2. 技术分析代理 (technicals.py)

**功能**: 基于 2024-2025 年研究的区制感知技术分析系统

- **核心算法**:
  - 趋势跟踪策略 (EMA、ADX、一目均衡)
  - 均值回归策略 (RSI、布林带、KD 指标)
  - 动量策略 (多周期动量、OBV)
  - GARCH 波动率预测模型
  - 统计套利信号

**高级特性**:

- **区制检测**: 使用高斯混合模型自动识别市场状态
- **动态权重**: 根据市场区制调整策略权重
- **信号增强**: 基于区制特性过滤和增强信号

```python
# 核心信号组合逻辑
regime_adjusted_weights = _calculate_regime_adjusted_weights(current_regime)
combined_signal = weighted_signal_combination({
    'trend': trend_signals,
    'mean_reversion': mean_reversion_signals,
    'momentum': momentum_signals,
    'volatility': volatility_signals
}, regime_adjusted_weights)
```

### 3. 基本面分析代理 (fundamentals.py)

**功能**: 全面的财务指标分析

- **分析维度**:
  - 盈利能力分析 (ROE、净利润率、营业利润率)
  - 增长性分析 (收入增长、盈利增长、账面价值增长)
  - 财务健康度 (流动比率、负债率、现金流)
  - 估值比率 (PE、PB、PS 比率)

**决策逻辑**:

```python
# 多维度信号聚合
signals = [profitability_signal, growth_signal, health_signal, valuation_signal]
overall_signal = determine_signal_by_majority(signals)
confidence = calculate_weighted_confidence(signals)
```

### 4. 估值分析代理 (valuation.py)

**功能**: 使用多种估值模型评估内在价值

- **估值方法**:
  - **DCF 模型**: 三阶段现金流折现
  - **所有者收益法**: 改进的巴菲特估值法
  - **相对估值**: 行业比较分析
  - **剩余收益模型**: 基于 ROE 的估值

**核心算法**:

```python
# 加权估值组合
all_valuations = [
    {"method": "DCF", "value": dcf_value, "weight": 0.35},
    {"method": "Owner Earnings", "value": owner_earnings_value, "weight": 0.35},
    {"method": "Relative Valuation", "value": relative_value, "weight": 0.15},
    {"method": "Residual Income", "value": residual_income_value, "weight": 0.15}
]
```

### 5. 情感分析代理 (sentiment.py)

**功能**: 基于新闻和市场情绪的分析

- **数据源**: 7 天内的新闻数据
- **分析方法**: NLP 情感分析
- **信号生成**: 情感分数到交易信号的转换

### 6. 宏观分析代理 (macro_analyst.py)

**功能**: 宏观经济环境对个股影响分析

- **分析因子**:
  - 货币政策 (利率、准备金率)
  - 财政政策 (政府支出、税收)
  - 产业政策 (行业规划、监管)
  - 国际环境 (全球经济、贸易关系)

### 7. AI 模型分析代理 (ai_model_analyst.py)

**功能**: 集成深度学习、强化学习和遗传编程模型

- **模型类型**:
  - **深度学习**: LSTM + 随机森林组合
  - **强化学习**: PPO 算法优化交易策略
  - **遗传编程**: 自动化因子挖掘

**信号聚合**:

```python
# 多模型信号组合
weights = {
    'deep_learning': 0.35,
    'reinforcement_learning': 0.35,
    'genetic_programming': 0.30
}
combined_signal = combine_ai_signals(ml_signals, rl_signals, factor_signals)
```

### 8. 辩论室代理 (debate_room.py)

**功能**: 基于 2024-2025 研究的自适应信号聚合系统

- **核心技术**:
  - FLAG-Trader 框架的区制感知聚合
  - FINSABER 权重优化
  - Lopez-Lira 动态阈值

**聚合算法**:

```python
def adaptive_signal_aggregation(signals: Dict, regime_info: Dict, confidence_threshold: float = 0.6):
    # 根据市场区制动态调整信号权重
    regime_adjustments = {
        "low_volatility_trending": {'technical': 1.3, 'ai_model': 1.2},
        "high_volatility_mean_reverting": {'fundamental': 1.4, 'valuation': 1.3},
        "crisis_regime": {'macro': 1.5, 'sentiment': 1.2}
    }
```

### 9. 风险管理代理 (risk_manager.py)

**功能**: 基于现代投资组合理论的综合风险评估

- **风险指标**:
  - VaR 和 CVaR 计算
  - 最大回撤分析
  - GARCH 波动率预测
  - 压力测试
  - 区制风险评估

**头寸规模优化**:

```python
# 区制感知的凯利准则
kelly_fraction = win_rate - ((1 - win_rate) / win_loss_ratio)
# 根据市场区制调整保守系数
conservative_factor = 0.3 if regime_name == "crisis_regime" else 0.5
```

### 10. 投资组合分析代理 (portfolio_analyzer.py)

**功能**: 多资产投资组合优化和风险评估

- **分析功能**:
  - 有效前沿生成
  - 夏普比率优化
  - 相关性分析
  - Beta 系数计算
  - 尾部风险测量

### 11. 投资组合管理代理 (portfolio_manager.py)

**功能**: 最终交易决策和投资组合管理

- **决策整合**: 综合所有分析师建议
- **现代投资组合理论**: 使用 MPT 优化决策
- **LLM 增强**: 使用大语言模型进行最终决策

**决策权重**:

```python
# 信号权重分配
weights = {
    'ai_models': 0.15,      # AI模型预测
    'valuation': 0.35,      # 估值分析 (主要驱动)
    'fundamental': 0.30,    # 基本面分析
    'technical': 0.25,      # 技术分析
    'macro': 0.15,          # 宏观分析
    'sentiment': 0.10       # 情感分析
}
```

### 12. 多方/空方研究员 (researcher_bull.py / researcher_bear.py)

**功能**: 从不同角度分析市场，提供多元化观点

- **多方研究员**: 寻找投资机会和积极因素
- **空方研究员**: 识别风险和消极因素
- **风险调整**: 基于市场整体环境调整置信度

## 高级特性

### 区制检测系统 (regime_detector.py)

基于 2024-2025 年研究的高级市场区制检测：

```python
class AdvancedRegimeDetector:
    def __init__(self, n_regimes: int = 3):
        self.regime_names = {
            0: "low_volatility_trending",
            1: "high_volatility_mean_reverting",
            2: "crisis_regime"
        }
```

**特征工程**:

- 多时间尺度波动率
- 趋势强度指标
- 动量特征
- 市场微观结构
- Hurst 指数 (长记忆性)

### 状态管理系统 (state.py)

统一的代理状态管理：

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[Dict[str, Any], merge_dicts]
    metadata: Annotated[Dict[str, Any], merge_dicts]
```

## 数据流架构

1. **数据收集阶段**: market_data_agent 收集原始数据
2. **分析阶段**: 各专业代理并行分析
3. **AI 预测阶段**: ai_model_analyst 生成机器学习预测
4. **整合阶段**: debate_room_agent 智能聚合信号
5. **风险评估**: risk_manager 评估风险并设置约束
6. **决策阶段**: portfolio_manager 生成最终交易决策

## 性能优化

- **缓存机制**: 宏观分析等耗时操作使用缓存
- **并行处理**: 多代理可并行执行
- **动态阈值**: 基于市场条件自适应调整
- **错误处理**: 完整的异常处理和降级机制

## 配置和扩展

### 添加新代理

```python
@agent_endpoint("new_agent", "新代理描述")
def new_agent(state: AgentState):
    # 实现代理逻辑
    return {
        "messages": [message],
        "data": updated_data,
        "metadata": metadata
    }
```

### 自定义权重

系统支持根据市场条件动态调整各代理权重，可通过配置文件或运行时参数调整。

## 研究基础

本系统基于 2024-2025 年的最新金融 AI 研究：

- **FLAG-Trader**: 区制感知的信号聚合
- **FINSABER**: 多因子信号整合框架
- **Lopez-Lira**: 动态阈值和权重调整
- **RLMF**: 强化学习在金融中的应用

## 使用示例

```python
# 初始化系统状态
state = {
    "messages": [],
    "data": {"ticker": "000001", "tickers": ["000001", "000002"]},
    "metadata": {"show_reasoning": True}
}

# 执行分析流水线
result = market_data_agent(state)
result = technical_analyst_agent(result)
result = fundamentals_agent(result)
# ... 其他代理
final_decision = portfolio_management_agent(result)
```

## 输出格式

每个代理返回标准化的 JSON 格式结果，包含：

- `signal`: 投资信号 (bullish/bearish/neutral)
- `confidence`: 置信度 (0-1)
- `reasoning`: 详细分析推理
- `metrics`: 相关计算指标

最终的投资组合管理代理输出完整的交易决策，包括具体的买卖建议和仓位大小。
