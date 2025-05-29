from langchain_core.messages import HumanMessage
import json
import pandas as pd

from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.utils.api_utils import agent_endpoint
from src.calc.portfolio_optimization import (
    portfolio_return, portfolio_volatility, portfolio_sharpe_ratio, 
    optimize_portfolio, efficient_frontier
)
from src.calc.tail_risk_measures import (
    calculate_historical_var, calculate_conditional_var
)
from src.tools.factor_data_api import (
    get_multi_stock_returns, get_stock_covariance_matrix, calculate_rolling_beta
)
from src.utils.logging_config import setup_logger

# 设置日志记录器
logger = setup_logger('portfolio_analyzer_agent')

@agent_endpoint("portfolio_analyzer", "投资组合分析师，分析多资产组合表现、进行组合优化和风险评估")
def portfolio_analyzer_agent(state: AgentState):
    """负责多资产投资组合分析、优化和风险评估"""
    show_workflow_status("Portfolio Analyzer")
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    
    # 从数据中获取资产列表
    tickers = data.get("tickers", [])
    if isinstance(tickers, str):
        tickers = [ticker.strip() for ticker in tickers.split(',')]
    
    # 如果只有一个资产或没有资产，静默跳过投资组合分析
    if not tickers or len(tickers) < 2:
        # 对于单资产分析，不显示警告，直接返回简单的分析结果
        if len(tickers) == 1:
            logger.info(f"单资产分析模式: {tickers[0]}")
            message_content = {
                "analysis_type": "single_asset",
                "ticker": tickers[0],
                "note": "单资产分析，跳过投资组合优化",
                "portfolio_analysis": None
            }
        else:
            logger.info("未提供资产代码，跳过投资组合分析")
            message_content = {
                "analysis_type": "no_assets",
                "note": "未提供资产代码，跳过投资组合分析",
                "portfolio_analysis": None
            }
        
        message = HumanMessage(
            content=json.dumps(message_content),
            name="portfolio_analyzer_agent",
        )
        return {
            "messages": [message],
            "data": data,
            "metadata": state["metadata"],
        }
    
    # 获取时间范围
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    
    # 调用factor_data_api获取多个股票的收益率
    try:
        logger.info(f"获取多个股票收益率: {tickers}")
        returns_df = get_multi_stock_returns(tickers, start_date, end_date)
        
        if returns_df.empty:
            raise ValueError("无法获取股票收益率数据")
            
        # 计算协方差矩阵和预期收益率
        logger.info("计算协方差矩阵和预期收益率")
        cov_matrix, expected_returns = get_stock_covariance_matrix(tickers, start_date, end_date)
        
        # 当前市场情况下的无风险利率
        risk_free_rate = 0.03  # 可以根据实际情况调整
        
        # 投资组合优化分析
        portfolio_analysis = analyze_portfolio(tickers, returns_df, cov_matrix, expected_returns, risk_free_rate)
        
        # 风险分析
        risk_analysis = analyze_portfolio_risk(returns_df)
        
        # 滚动Beta分析
        beta_analysis = analyze_rolling_betas(tickers, start_date, end_date)
        
        # 生成有效前沿
        ef_results = generate_efficient_frontier(expected_returns, cov_matrix, risk_free_rate)
        
        # 组合分析结果
        message_content = {
            "tickers": tickers,
            "portfolio_analysis": portfolio_analysis,
            "risk_analysis": risk_analysis,
            "beta_analysis": beta_analysis,
            "efficient_frontier": ef_results,
            "summary": generate_summary(portfolio_analysis, risk_analysis, beta_analysis)
        }
        
    except Exception as e:
        logger.error(f"投资组合分析失败: {str(e)}")
        message_content = {
            "error": f"投资组合分析过程中发生错误: {str(e)}",
            "tickers": tickers
        }
    
    # 创建消息
    message = HumanMessage(
        content=json.dumps(message_content),
        name="portfolio_analyzer_agent",
    )
    
    # 显示推理过程
    if show_reasoning:
        show_agent_reasoning(message_content, "Portfolio Analyzer Agent")
        # 保存推理信息到metadata供API使用
        state["metadata"]["agent_reasoning"] = message_content
    
    show_workflow_status("Portfolio Analyzer", "completed")
    return {
        "messages": [message],
        "data": {
            **data,
            "portfolio_analysis": message_content
        },
        "metadata": state["metadata"],
    }

def analyze_portfolio(tickers, returns_df, cov_matrix, expected_returns, risk_free_rate=0.03):
    """
    分析投资组合的各种组合情况
    """
    # 计算等权重投资组合
    equal_weights = pd.Series(1/len(tickers), index=tickers)
    equal_return = portfolio_return(equal_weights.values, expected_returns.values)
    equal_volatility = portfolio_volatility(equal_weights.values, cov_matrix.values)
    equal_sharpe = portfolio_sharpe_ratio(equal_weights.values, expected_returns.values, cov_matrix.values, risk_free_rate)
    
    # 最大夏普比率投资组合
    try:
        max_sharpe_portfolio = optimize_portfolio(
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            risk_free_rate=risk_free_rate,
            objective='sharpe'
        )
        
        # 最小风险投资组合
        min_risk_portfolio = optimize_portfolio(
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            risk_free_rate=risk_free_rate,
            objective='min_risk'
        )
    except Exception as e:
        logger.error(f"投资组合优化失败: {e}")
        max_sharpe_portfolio = {'weights': equal_weights, 'return': equal_return, 'risk': equal_volatility, 'sharpe_ratio': equal_sharpe}
        min_risk_portfolio = {'weights': equal_weights, 'return': equal_return, 'risk': equal_volatility, 'sharpe_ratio': equal_sharpe}
    
    # 计算相关系数矩阵
    correlation = returns_df.corr()
    
    # 返回分析结果
    return {
        "equal_weight": {
            "weights": equal_weights.to_dict(),
            "return": float(equal_return),
            "risk": float(equal_volatility),
            "sharpe_ratio": float(equal_sharpe)
        },
        "max_sharpe": {
            "weights": max_sharpe_portfolio['weights'].to_dict(),
            "return": float(max_sharpe_portfolio['return']),
            "risk": float(max_sharpe_portfolio['risk']),
            "sharpe_ratio": float(max_sharpe_portfolio['sharpe_ratio'])
        },
        "min_risk": {
            "weights": min_risk_portfolio['weights'].to_dict(),
            "return": float(min_risk_portfolio['return']),
            "risk": float(min_risk_portfolio['risk']),
            "sharpe_ratio": float(min_risk_portfolio['sharpe_ratio'])
        },
        "correlation_matrix": correlation.to_dict(),
        "risk_free_rate": risk_free_rate
    }

def analyze_portfolio_risk(returns_df):
    """
    分析投资组合的风险指标
    """
    # 计算等权重投资组合收益率
    portfolio_returns = returns_df.mean(axis=1)
    
    # 计算风险指标
    var_95 = calculate_historical_var(portfolio_returns, confidence_level=0.95)
    cvar_95 = calculate_conditional_var(portfolio_returns, confidence_level=0.95)
    
    # 计算最大回撤
    cum_returns = (1 + portfolio_returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max - 1)
    max_drawdown = drawdown.min()
    
    # 计算偏度和峰度
    skewness = portfolio_returns.skew()
    kurtosis = portfolio_returns.kurt()
    
    # 计算投资组合范围内的最佳和最差资产
    mean_returns = returns_df.mean()
    best_asset = mean_returns.idxmax()
    worst_asset = mean_returns.idxmin()
    
    return {
        "var_95": float(var_95),
        "cvar_95": float(cvar_95),
        "max_drawdown": float(max_drawdown),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
        "best_asset": {
            "ticker": best_asset,
            "return": float(mean_returns.max())
        },
        "worst_asset": {
            "ticker": worst_asset,
            "return": float(mean_returns.min())
        }
    }

def analyze_rolling_betas(tickers, start_date, end_date, window=60):
    """
    计算资产对应市场的滚动Beta系数
    """
    beta_results = {}
    for ticker in tickers:
        try:
            # 使用factor_data_api计算滚动Beta
            rolling_beta = calculate_rolling_beta(ticker, window, start_date, end_date)
            
            if not rolling_beta.empty:
                # 计算Beta的统计数据
                beta_avg = rolling_beta.mean()
                beta_std = rolling_beta.std()
                beta_min = rolling_beta.min()
                beta_max = rolling_beta.max()
                
                beta_results[ticker] = {
                    "average_beta": float(beta_avg),
                    "beta_volatility": float(beta_std),
                    "min_beta": float(beta_min),
                    "max_beta": float(beta_max),
                    "latest_beta": float(rolling_beta.iloc[-1]) if len(rolling_beta) > 0 else float(beta_avg)
                }
        except Exception as e:
            logger.error(f"计算{ticker}滚动Beta失败: {e}")
            beta_results[ticker] = {"error": str(e)}
    
    return beta_results

def generate_efficient_frontier(expected_returns, cov_matrix, risk_free_rate=0.03, points=20):
    """
    生成有效前沿
    """
    try:
        ef = efficient_frontier(
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            risk_free_rate=risk_free_rate,
            points=points
        )
        
        # 转换为可序列化的字典
        ef_dict = {
            "returns": ef['return'].tolist(),
            "risks": ef['risk'].tolist(),
            "sharpe_ratios": ef['sharpe_ratio'].tolist()
        }
        
        return ef_dict
    except Exception as e:
        logger.error(f"生成有效前沿失败: {e}")
        return {"error": str(e)}

def generate_summary(portfolio_analysis, risk_analysis, beta_analysis):
    """
    生成投资组合分析总结
    """
    # 最佳投资组合
    max_sharpe = portfolio_analysis["max_sharpe"]
    
    # 风险度量
    var_95 = risk_analysis["var_95"]
    max_drawdown = risk_analysis["max_drawdown"]
    
    # 构建总结
    summary = []
    
    # 最优配置建议
    summary.append("最优投资组合配置 (最大夏普比率):")
    for ticker, weight in max_sharpe["weights"].items():
        summary.append(f"- {ticker}: {weight*100:.2f}%")
    
    summary.append(f"该配置的预期年化收益率为 {max_sharpe['return']*100:.2f}%，波动率为 {max_sharpe['risk']*100:.2f}%，夏普比率为 {max_sharpe['sharpe_ratio']:.2f}")
    
    # 风险评估
    summary.append(f"风险评估: 95%置信水平下的VaR为 {var_95*100:.2f}%，最大回撤为 {max_drawdown*100:.2f}%")
    
    # Beta分析
    summary.append("各资产对市场的Beta系数:")
    for ticker, beta_data in beta_analysis.items():
        if "average_beta" in beta_data:
            summary.append(f"- {ticker}: {beta_data['average_beta']:.2f} (最近: {beta_data['latest_beta']:.2f})")
    
    return "\n".join(summary)