from datetime import datetime, timedelta
import json
import time
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from src.tools.api import get_price_data
from src.main import run_hedge_fund
import sys
import matplotlib
import os
import uuid  # 添加uuid模块用于生成run_id

from src.tools.factor_data_api import get_risk_free_rate
from src.utils.logging_config import setup_logger

# 根据操作系统配置中文字体
if sys.platform.startswith('win'):
    # Windows系统
    matplotlib.rc('font', family='Microsoft YaHei')
elif sys.platform.startswith('linux'):
    # Linux系统
    matplotlib.rc('font', family='WenQuanYi Micro Hei')
else:
    # macOS系统
    matplotlib.rc('font', family='PingFang SC')

# 用来正常显示负号
matplotlib.rcParams['axes.unicode_minus'] = False


class Backtester:
    def __init__(self, agent, ticker, start_date, end_date, initial_capital, num_of_news, tickers=None):
        self.agent = agent
        self.ticker = ticker  # 主要交易的股票代码
        self.tickers = tickers if tickers else None  # 多个股票代码
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.portfolio = {"cash": initial_capital, "stock": 0}
        self.portfolio_values = []
        self.num_of_news = num_of_news
        # 设置回测日志
        self.setup_backtest_logging()
        self.logger = setup_logger('Backtester')

        # 初始化 API 调用管理
        self._api_call_count = 0
        self._api_window_start = time.time()
        self._last_api_call = 0

        # 回测配置参数
        self.trading_cost = 0.001  # 0.1%交易成本
        self.slippage = 0.001  # 0.1%滑点成本
        self.benchmark_ticker = "000300"  # 沪深300指数作为基准
        self.rebalance_frequency = "day"  # 日频回测
        
        # 性能跟踪字段
        self.trade_history = []
        self.daily_returns = []
        self.benchmark_returns = []
        self.drawdowns = []

        # 验证输入参数
        self.validate_inputs()

    def validate_inputs(self):
        """验证输入参数的有效性"""
        try:
            start = datetime.strptime(self.start_date, "%Y-%m-%d")
            end = datetime.strptime(self.end_date, "%Y-%m-%d")
            if start >= end:
                raise ValueError("开始日期必须早于结束日期")
            if self.initial_capital <= 0:
                raise ValueError("初始资金必须大于0")
            if not isinstance(self.ticker, str) or len(self.ticker) != 6:
                raise ValueError("无效的股票代码格式")
            self.logger.info("输入参数验证通过")
        except Exception as e:
            self.logger.error(f"输入参数验证失败: {str(e)}")
            raise

    def get_agent_decision(self, current_date, lookback_start, portfolio):
        """获取智能体决策，包含 API 限制处理和增强的错误处理"""
        max_retries = 3

        # 记录请求参数，以便调试
        self.logger.info(f"请求Agent决策: 日期={current_date}, 回溯开始={lookback_start}")
        self.logger.info(f"当前组合状态: 现金={portfolio['cash']:.2f}, 持股={portfolio.get('stock', 0)}")

        # 检查并重置 API 时间窗口
        current_time = time.time()
        if current_time - self._api_window_start >= 60:
            self._api_call_count = 0
            self._api_window_start = current_time

        # 放宽 API 限制
        if self._api_call_count >= 12:
            wait_time = 60 - (current_time - self._api_window_start)
            if wait_time > 0:
                self.logger.info(f"达到API限制，等待 {wait_time:.1f} 秒")
                time.sleep(wait_time)
                self._api_call_count = 0
                self._api_window_start = time.time()

        for attempt in range(max_retries):
            try:
                # 确保调用间隔
                if self._last_api_call:
                    time_since_last_call = time.time() - self._last_api_call
                    if time_since_last_call < 3:
                        sleep_time = 3 - time_since_last_call
                        time.sleep(sleep_time)

                # 更新调用时间和计数
                self._last_api_call = time.time()
                self._api_call_count += 1

                # 生成run_id - 这是关键修复
                run_id = str(uuid.uuid4())
                
                # 调用智能体并解析结果，添加run_id参数
                result = self.agent(
                    run_id=run_id,  # 添加缺失的run_id参数
                    ticker=self.ticker,
                    start_date=lookback_start,
                    end_date=current_date,
                    portfolio=portfolio,
                    num_of_news=self.num_of_news
                )
                
                # 记录原始返回结果的类型和内容
                self.logger.info(f"智能体返回结果类型: {type(result)}")
                if isinstance(result, str) and len(result) > 200:
                    self.logger.info(f"智能体返回结果 (前200字符): {result[:200]}...")
                else:
                    self.logger.info(f"智能体返回结果: {result}")
                
                # 根据返回结果类型不同处理
                if isinstance(result, dict):
                    # 如果返回的是字典，直接使用
                    self.logger.info("智能体返回了字典格式结果，直接处理")
                    
                    # 确保结果具有正确的结构
                    if "decision" not in result:
                        self.logger.warning("结果字典中没有decision字段，使用默认决策")
                        result["decision"] = {"action": "hold", "quantity": 0}
                    
                    # 如果没有分析师信号，添加空字典
                    if "analyst_signals" not in result:
                        result["analyst_signals"] = {}
                    
                    return result
                    
                elif isinstance(result, str):
                    # 尝试解析返回的字符串为JSON
                    self.logger.info("智能体返回了字符串，尝试解析为JSON")
                    
                    # 清理可能的markdown标记
                    cleaned_result = result.replace('```json\n', '').replace('\n```', '').strip()
                    
                    # 只打印部分清理后的结果
                    if len(cleaned_result) > 200:
                        self.logger.info(f"清理后的结果 (前200字符): {cleaned_result[:200]}...")
                    else:
                        self.logger.info(f"清理后的结果: {cleaned_result}")
                    
                    try:
                        parsed_result = json.loads(cleaned_result)
                        self.logger.info("JSON解析成功")
                        
                        # 检查结果是否有action和quantity字段
                        has_action = False
                        if isinstance(parsed_result, dict):
                            if "action" in parsed_result and "quantity" in parsed_result:
                                has_action = True
                                # 直接返回决策
                                self.logger.info(f"发现直接的决策: action={parsed_result['action']}, quantity={parsed_result['quantity']}")
                                return {
                                    "decision": parsed_result,
                                    "analyst_signals": {}
                                }
                            elif "decision" in parsed_result:
                                has_action = True
                                self.logger.info(f"发现决策字段: {parsed_result['decision']}")
                        
                        # 构建标准格式的结果
                        formatted_result = {
                            "decision": {"action": "hold", "quantity": 0},  # 默认决策
                            "analyst_signals": {}
                        }
                        
                        # 检查并处理决策字段
                        if "decision" in parsed_result:
                            if isinstance(parsed_result["decision"], dict):
                                formatted_result["decision"] = parsed_result["decision"]
                            elif isinstance(parsed_result["decision"], str):
                                # 如果决策是字符串，尝试解析内容
                                decision_text = parsed_result["decision"].lower()
                                if "buy" in decision_text:
                                    formatted_result["decision"] = {"action": "buy", "quantity": 100}
                                elif "sell" in decision_text:
                                    formatted_result["decision"] = {"action": "sell", "quantity": 100}
                        
                        # 处理智能体信号
                        if "agent_signals" in parsed_result and isinstance(parsed_result["agent_signals"], list):
                            for signal in parsed_result["agent_signals"]:
                                if isinstance(signal, dict) and "agent" in signal:
                                    agent_name = signal.get("agent_name", signal.get("agent", "unknown"))
                                    formatted_result["analyst_signals"][agent_name] = {
                                        "signal": signal.get("signal", "unknown"),
                                        "confidence": signal.get("confidence", 0)
                                    }
                                    
                                    # 复制其他可能的字段
                                    for key, value in signal.items():
                                        if key not in ["agent", "agent_name", "signal", "confidence"]:
                                            formatted_result["analyst_signals"][agent_name][key] = value
                        
                        self.logger.info(f"最终处理后的决策: {formatted_result['decision']}")
                        return formatted_result
                    
                    except json.JSONDecodeError as e:
                        # JSON解析失败，尝试从文本中提取决策
                        self.logger.warning(f"JSON解析错误: {str(e)}")
                        self.logger.info("尝试从文本中提取决策")
                        
                        # 使用简单的文本分析判断决策
                        decision = self.parse_decision_from_text(result)
                        self.logger.info(f"从文本提取的决策: {decision}")
                        
                        return {
                            "decision": decision,
                            "analyst_signals": {}
                        }
                
                # 如果都不是字典也不是字符串，返回默认决策
                self.logger.warning(f"未知的结果类型: {type(result)}")
                return {
                    "decision": {"action": "hold", "quantity": 0},
                    "analyst_signals": {}
                }
                    
            except Exception as e:
                # 捕获所有异常
                if "AFC is enabled" in str(e):
                    self.logger.warning(f"触发 AFC 限制，等待 60 秒后重试...")
                    time.sleep(60)
                    self._api_call_count = 0
                    self._api_window_start = time.time()
                    continue

                self.logger.warning(f"获取Agent决策失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    # 最后一次尝试，返回固定的买入决策
                    self.logger.info("所有尝试失败，返回测试用的买入决策")
                    return {"decision": {"action": "buy", "quantity": 50}, "analyst_signals": {}}
                time.sleep(2 ** attempt)
        
        # 如果所有尝试都失败，返回默认决策
        self.logger.warning("无法获取有效决策，返回默认决策")
        return {"decision": {"action": "hold", "quantity": 0}, "analyst_signals": {}}

    def parse_decision_from_text(self, text):
        """从文本中解析交易决策"""
        text = text.lower()

        # 默认决策
        decision = {"action": "hold", "quantity": 0}

        # 检查是否包含决策关键词
        if "buy" in text or "bullish" in text:
            decision["action"] = "buy"
            decision["quantity"] = 100  # 默认购买数量
        elif "sell" in text or "bearish" in text:
            decision["action"] = "sell"
            decision["quantity"] = 100  # 默认卖出数量

        return decision

    def execute_trade(self, action, quantity, current_price):
        """执行交易，验证组合约束"""
        if action == "buy" and quantity > 0:
            cost = quantity * current_price
            if cost <= self.portfolio["cash"]:
                self.portfolio["stock"] += quantity
                self.portfolio["cash"] -= cost
                return quantity
            else:
                # 计算最大可买数量
                max_quantity = int(self.portfolio["cash"] // current_price)
                if max_quantity > 0:
                    self.portfolio["stock"] += max_quantity
                    self.portfolio["cash"] -= max_quantity * current_price
                    return max_quantity
                return 0
        elif action == "sell" and quantity > 0:
            quantity = min(quantity, self.portfolio["stock"])
            if quantity > 0:
                self.portfolio["cash"] += quantity * current_price
                self.portfolio["stock"] -= quantity
                return quantity
            return 0
        return 0

    def setup_backtest_logging(self):
        """设置回测日志"""
        # 创建日志目录
        log_dir = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)

        # 创建回测日志记录器
        self.backtest_logger = logging.getLogger('backtest')
        self.backtest_logger.setLevel(logging.INFO)

        # 清除已存在的处理器
        if self.backtest_logger.handlers:
            self.backtest_logger.handlers.clear()

        # 设置文件处理器
        current_date = datetime.now().strftime('%Y%m%d')
        backtest_period = f"{self.start_date.replace('-', '')}_{self.end_date.replace('-', '')}"
        log_file = os.path.join(
            log_dir, f"backtest_{self.ticker}_{current_date}_{backtest_period}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # 设置日志格式
        formatter = logging.Formatter('%(message)s')  # 简化格式，只显示消息
        file_handler.setFormatter(formatter)

        # 添加处理器
        self.backtest_logger.addHandler(file_handler)

        # 写入回测初始信息
        self.backtest_logger.info(
            f"回测开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.backtest_logger.info(f"股票代码: {self.ticker}")
        self.backtest_logger.info(f"回测区间: {self.start_date} 至 {self.end_date}")
        self.backtest_logger.info(f"初始资金: {self.initial_capital:,.2f}\n")
        self.backtest_logger.info("-" * 100)

    def run_backtest(self):
        """运行回测"""
        # 回测精度：每天
        dates = pd.date_range(self.start_date, self.end_date, freq="B")
        # 回测精度：每周取一天
        # dates = pd.date_range(self.start_date, self.end_date, freq="W-MON")

        self.logger.info("\n开始回测...")
        print(f"{'日期':<12} {'代码':<6} {'操作':<6} {'数量':>8} {'价格':>8} {'现金':>12} {'持仓':>8} {'总值':>12} {'看多':>8} {'看空':>8} {'中性':>8}")
        print("-" * 110)

        for current_date in dates:
            lookback_start = (current_date - timedelta(days=30)
                              ).strftime("%Y-%m-%d")
            current_date_str = current_date.strftime("%Y-%m-%d")

            # 获取Agent决策
            output = self.get_agent_decision(
                current_date_str, lookback_start, self.portfolio)

            # 记录每个Agent的信号和分析结果
            self.backtest_logger.info(f"\n交易日期: {current_date_str}")
            if "analyst_signals" in output:
                self.backtest_logger.info("\n各Agent分析结果:")
                for agent_name, signal in output["analyst_signals"].items():
                    self.backtest_logger.info(f"\n{agent_name}:")

                    # 记录信号和置信度
                    signal_str = f"- 信号: {signal.get('signal', 'unknown')}"
                    if 'confidence' in signal:
                        signal_str += f", 置信度: {signal.get('confidence', 0)*100:.0f}%"
                    self.backtest_logger.info(signal_str)

                    # 记录分析结果
                    if 'analysis' in signal:
                        self.backtest_logger.info("- 分析结果:")
                        analysis = signal['analysis']
                        if isinstance(analysis, dict):
                            for key, value in analysis.items():
                                self.backtest_logger.info(f"  {key}: {value}")
                        elif isinstance(analysis, list):
                            for item in analysis:
                                self.backtest_logger.info(f"  • {item}")
                        else:
                            self.backtest_logger.info(f"  {analysis}")

                    # 记录理由
                    if 'reason' in signal:
                        self.backtest_logger.info("- 决策理由:")
                        reason = signal['reason']
                        if isinstance(reason, list):
                            for item in reason:
                                self.backtest_logger.info(f"  • {item}")
                        else:
                            self.backtest_logger.info(f"  • {reason}")

                    # 记录其他可能的指标
                    for key, value in signal.items():
                        if key not in ['signal', 'confidence', 'analysis', 'reason']:
                            self.backtest_logger.info(f"- {key}: {value}")

                self.backtest_logger.info("\n综合决策:")

            agent_decision = output.get(
                "decision", {"action": "hold", "quantity": 0})
            action, quantity = agent_decision.get(
                "action", "hold"), agent_decision.get("quantity", 0)

            # 记录决策详情
            self.backtest_logger.info(f"行动: {action.upper()}")
            self.backtest_logger.info(f"数量: {quantity}")
            if "reason" in agent_decision:
                self.backtest_logger.info(f"决策理由: {agent_decision['reason']}")

            # 获取当前价格并执行交易
            df = get_price_data(self.ticker, lookback_start, current_date_str)
            if df is None or df.empty:
                continue

            current_price = df.iloc[-1]['open']
            executed_quantity = self.execute_trade(
                action, quantity, current_price)

            # 更新组合总值
            total_value = self.portfolio["cash"] + \
                self.portfolio["stock"] * current_price
            self.portfolio["portfolio_value"] = total_value

            # 计算当日收益率
            if len(self.portfolio_values) > 0:
                daily_return = (
                    total_value / self.portfolio_values[-1]["Portfolio Value"] - 1) * 100
            else:
                daily_return = 0

            # 记录组合价值和收益率
            self.portfolio_values.append({
                "Date": current_date,
                "Portfolio Value": total_value,
                "Daily Return": daily_return
            })

    def analyze_performance(self):
        """性能分析函数"""
        performance_df = pd.DataFrame(self.portfolio_values).set_index("Date")

        # 计算累计收益率
        performance_df["Cumulative Return"] = (
            performance_df["Portfolio Value"] / self.initial_capital - 1) * 100

        # 将金额转换为千元
        performance_df["Portfolio Value (K)"] = performance_df["Portfolio Value"] / 1000
        
        # 计算高级性能指标
        self._calculate_advanced_metrics(performance_df)
        
        # 创建增强图表
        self._create_enhanced_charts(performance_df)
        
        # 返回性能数据
        return performance_df
    
    def _calculate_advanced_metrics(self, performance_df):
        """计算高级性能指标"""
        # 从每日收益计算
        daily_returns = performance_df["Daily Return"] / 100  # 转换为小数
        
        # 基础风险指标
        mean_daily_return = daily_returns.mean()
        std_daily_return = daily_returns.std()
        
        # 年化收益和风险
        annual_return = mean_daily_return * 252
        annual_volatility = std_daily_return * np.sqrt(252)
        
        # 获取无风险利率
        try:
            # 使用get_risk_free_rate函数获取日度无风险利率
            risk_free_rate_series = get_risk_free_rate(
                start_date=self.start_date, 
                end_date=self.end_date, 
                freq='D'  # 获取日频数据
            )
            # 计算平均日度无风险利率
            risk_free_rate = risk_free_rate_series.mean()
            self.logger.info(f"成功获取无风险利率，平均值：{risk_free_rate:.6f}")
        except Exception as e:
            # 如果获取失败，使用默认值
            risk_free_rate = 0.03 / 252  # 默认年化3%无风险利率的日度值
            self.logger.warning(f"获取无风险利率失败: {str(e)}，使用默认值 {risk_free_rate:.6f}")
        
        # 夏普比率
        sharpe_ratio = (mean_daily_return - risk_free_rate) / std_daily_return * np.sqrt(252) if std_daily_return != 0 else 0
        
        # 索提诺比率 (只考虑下行风险)
        downside_returns = daily_returns[daily_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if not downside_returns.empty else 0
        sortino_ratio = (mean_daily_return - risk_free_rate) / downside_deviation * np.sqrt(252) if downside_deviation != 0 else 0
        
        # 最大回撤
        rolling_max = performance_df["Portfolio Value"].cummax()
        drawdown = (performance_df["Portfolio Value"] / rolling_max - 1) * 100
        max_drawdown = drawdown.min()
        
        # 卡玛比率 (考虑回撤风险的收益率指标)
        calmar_ratio = annual_return / (abs(max_drawdown) / 100) if max_drawdown != 0 else 0
        
        # 胜率统计
        winning_days = (daily_returns > 0).sum()
        losing_days = (daily_returns < 0).sum()
        win_rate = winning_days / (winning_days + losing_days) if (winning_days + losing_days) > 0 else 0
        
        # 平均盈亏比
        avg_gain = daily_returns[daily_returns > 0].mean() if not daily_returns[daily_returns > 0].empty else 0
        avg_loss = daily_returns[daily_returns < 0].mean() if not daily_returns[daily_returns < 0].empty else 0
        profit_loss_ratio = abs(avg_gain / avg_loss) if avg_loss != 0 else 0
        
        # 最大连续盈利和亏损天数
        profit_streaks = []
        loss_streaks = []
        current_streak = 0
        for ret in daily_returns:
            if ret > 0:
                if current_streak > 0:
                    current_streak += 1
                else:
                    if current_streak < 0:
                        loss_streaks.append(abs(current_streak))
                    current_streak = 1
            elif ret < 0:
                if current_streak < 0:
                    current_streak -= 1
                else:
                    if current_streak > 0:
                        profit_streaks.append(current_streak)
                    current_streak = -1
            else:  # ret == 0
                if current_streak > 0:
                    profit_streaks.append(current_streak)
                elif current_streak < 0:
                    loss_streaks.append(abs(current_streak))
                current_streak = 0
                
        if current_streak > 0:
            profit_streaks.append(current_streak)
        elif current_streak < 0:
            loss_streaks.append(abs(current_streak))
            
        max_profit_streak = max(profit_streaks) if profit_streaks else 0
        max_loss_streak = max(loss_streaks) if loss_streaks else 0
        
        # 添加使用的无风险利率值到指标中
        self.metrics = {
            "总收益率": f"{(performance_df['Portfolio Value'].iloc[-1] / self.initial_capital - 1) * 100:.2f}%",
            "年化收益率": f"{annual_return * 100:.2f}%",
            "年化波动率": f"{annual_volatility * 100:.2f}%",
            "无风险利率": f"{risk_free_rate * 252 * 100:.2f}%",  # 显示年化无风险利率
            "夏普比率": f"{sharpe_ratio:.2f}",
            "索提诺比率": f"{sortino_ratio:.2f}",
            "最大回撤": f"{max_drawdown:.2f}%",
            "卡玛比率": f"{calmar_ratio:.2f}",
            "胜率": f"{win_rate * 100:.2f}%",
            "盈亏比": f"{profit_loss_ratio:.2f}",
            "最大连续盈利天数": max_profit_streak,
            "最大连续亏损天数": max_loss_streak
        }
        
        # 打印性能指标
        print("\n=== 回测性能指标 ===")
        for name, value in self.metrics.items():
            print(f"{name}: {value}")
    
    def _create_enhanced_charts(self, performance_df):
        """创建增强的分析图表"""
        # 图表目录
        img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets', 'img')
        os.makedirs(img_dir, exist_ok=True)
        
        # 生成基础文件名
        current_date = datetime.now().strftime('%Y%m%d')
        backtest_period = f"{self.start_date.replace('-', '')}_{self.end_date.replace('-', '')}"
        img_base = os.path.join(img_dir, f"backtest_{self.ticker}_{current_date}_{backtest_period}")
        
        # 创建多个图表
        
        # 1. 收益与回撤图
        self._create_returns_drawdown_chart(performance_df, f"{img_base}_returns_drawdown.png")
        
        # 2. 交易分析图
        self._create_trade_analysis_chart(performance_df, f"{img_base}_trades.png")
        
        # 3. 月度收益热图
        self._create_monthly_returns_heatmap(performance_df, f"{img_base}_monthly.png")
        
        # 4. 风险收益散点图 (如果有基准数据)
        if hasattr(self, 'benchmark_returns') and len(self.benchmark_returns) > 0:
            self._create_risk_return_chart(performance_df, f"{img_base}_risk_return.png")
    
    def _create_returns_drawdown_chart(self, performance_df, filename):
        """创建收益与回撤组合图表"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # 绘制累计收益曲线
        ax1.plot(performance_df.index, performance_df["Cumulative Return"], 
                label="投资组合", color='#4CAF50', linewidth=2)
        
        # 如果有基准数据，添加基准曲线
        if hasattr(self, 'benchmark_returns') and len(self.benchmark_returns) > 0:
            benchmark_df = pd.DataFrame(self.benchmark_returns)
            ax1.plot(benchmark_df.index, benchmark_df["Cumulative Return"], 
                    label="基准", color='#2196F3', linewidth=1.5, linestyle='--')
        
        # 添加交易标记
        buy_dates = [trade['date'] for trade in self.trade_history if trade['action'] == 'buy']
        sell_dates = [trade['date'] for trade in self.trade_history if trade['action'] == 'sell']
        
        for date in buy_dates:
            if date in performance_df.index:
                value = performance_df.loc[date, "Cumulative Return"]
                ax1.scatter(date, value, color='green', marker='^', s=100)
                
        for date in sell_dates:
            if date in performance_df.index:
                value = performance_df.loc[date, "Cumulative Return"]
                ax1.scatter(date, value, color='red', marker='v', s=100)
        
        # 美化收益图
        ax1.set_title("累计收益 & 交易", fontsize=14, fontweight='bold')
        ax1.set_ylabel("累计收益率 (%)", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # 计算回撤并绘制
        rolling_max = performance_df["Portfolio Value"].cummax()
        drawdown = (performance_df["Portfolio Value"] / rolling_max - 1) * 100
        
        ax2.fill_between(performance_df.index, 0, drawdown, color='#FFA726', alpha=0.3)
        ax2.plot(performance_df.index, drawdown, color='#E65100', linewidth=1)
        
        # 美化回撤图
        ax2.set_title("回撤", fontsize=14, fontweight='bold')
        ax2.set_ylabel("回撤 (%)", fontsize=12)
        ax2.set_xlabel("日期", fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"收益与回撤图表已保存至: {filename}")
    
    def _create_trade_analysis_chart(self, performance_df, filename):
        """创建交易分析图表"""
        if not self.trade_history:
            return
            
        # 准备交易数据
        trades_df = pd.DataFrame(self.trade_history)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 绘制每笔交易盈亏
        if 'profit_loss' in trades_df.columns:
            colors = ['green' if pl >= 0 else 'red' for pl in trades_df['profit_loss']]
            ax1.bar(range(len(trades_df)), trades_df['profit_loss'], color=colors)
            ax1.set_title("每笔交易盈亏", fontsize=14, fontweight='bold')
            ax1.set_ylabel("盈亏金额", fontsize=12)
            ax1.set_xticks(range(len(trades_df)))
            ax1.set_xticklabels(trades_df.index, rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # 添加累计盈亏线
            cumulative_pl = trades_df['profit_loss'].cumsum()
            ax_twin = ax1.twinx()
            ax_twin.plot(range(len(trades_df)), cumulative_pl, color='blue', linewidth=2)
            ax_twin.set_ylabel("累计盈亏", fontsize=12, color='blue')
        
        # 绘制持仓时间分布直方图
        if 'holding_days' in trades_df.columns:
            ax2.hist(trades_df['holding_days'], bins=20, color='#2196F3', alpha=0.7)
            ax2.set_title("持仓时间分布", fontsize=14, fontweight='bold')
            ax2.set_xlabel("持仓天数", fontsize=12)
            ax2.set_ylabel("交易次数", fontsize=12)
            ax2.grid(True, alpha=0.3)
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"交易分析图表已保存至: {filename}")
    
    def _create_monthly_returns_heatmap(self, performance_df, filename):
        """创建月度收益热图"""
        # 确保有日期索引
        if not isinstance(performance_df.index, pd.DatetimeIndex):
            performance_df.index = pd.to_datetime(performance_df.index)
            
        # 计算月度收益
        monthly_returns = performance_df['Daily Return'].resample('M').apply(
            lambda x: ((1 + x/100).prod() - 1) * 100
        )
        
        # 创建年/月交叉表
        returns_table = pd.DataFrame(monthly_returns)
        returns_table['Year'] = returns_table.index.year
        returns_table['Month'] = returns_table.index.month
        returns_pivot = returns_table.pivot_table(
            values='Daily Return', index='Year', columns='Month'
        )
        
        # 添加年度合计
        returns_pivot['Annual'] = returns_pivot.apply(
            lambda x: ((1 + x/100).prod() - 1) * 100, axis=1
        )
        
        # 绘制热图
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 调整颜色映射，使用发散型颜色映射
        cmap = plt.cm.RdYlGn  # Red-Yellow-Green
        norm = plt.Normalize(vmin=-10, vmax=10)  # 假设最大值为±10%
        
        # 绘制热图
        sns.heatmap(returns_pivot, cmap=cmap, norm=norm, annot=True, 
                   fmt=".2f", cbar_kws={'label': '月收益率 (%)'}, ax=ax)
        
        # 美化图表
        month_names = ['', '1月', '2月', '3月', '4月', '5月', '6月', 
                      '7月', '8月', '9月', '10月', '11月', '12月', '全年']
        ax.set_xticklabels(month_names[:len(returns_pivot.columns)+1])
        ax.set_title("月度收益热图", fontsize=14, fontweight='bold')
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"月度收益热图已保存至: {filename}")


if __name__ == "__main__":
    import argparse

    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='运行回测模拟')
    parser.add_argument('--ticker', type=str, required=True,
                        help='主要股票代码 (例如: 600519)')
    parser.add_argument('--tickers', type=str,
                        help='多个股票代码，逗号分隔 (例如: "600519,000858,601398")')
    parser.add_argument('--end-date', type=str,
                        default=datetime.now().strftime('%Y-%m-%d'), help='结束日期，格式：YYYY-MM-DD')
    parser.add_argument('--start-date', type=str, default=(datetime.now() -
                        timedelta(days=90)).strftime('%Y-%m-%d'), help='开始日期，格式：YYYY-MM-DD')
    parser.add_argument('--initial-capital', type=float,
                        default=100000, help='初始资金 (默认: 100000)')
    parser.add_argument('--num-of-news', type=int, default=5,
                        help='Number of news articles to analyze for sentiment (default: 5)')

    args = parser.parse_args()

    # 创建回测器实例
    backtester = Backtester(
        agent=run_hedge_fund,
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        num_of_news=args.num_of_news,
        tickers=args.tickers  # 传入多个股票代码
    )

    # 运行回测
    backtester.run_backtest()

    # 分析性能
    performance_df = backtester.analyze_performance()