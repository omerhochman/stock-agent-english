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

# Setup logger
logger = setup_logger('portfolio_analyzer_agent')

@agent_endpoint("portfolio_analyzer", "Portfolio analyst, analyzing multi-asset portfolio performance, conducting portfolio optimization and risk assessment")
def portfolio_analyzer_agent(state: AgentState):
    """Responsible for multi-asset portfolio analysis, optimization and risk assessment"""
    show_workflow_status("Portfolio Analyzer")
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    
    # Get asset list from data
    tickers = data.get("tickers", [])
    if isinstance(tickers, str):
        tickers = [ticker.strip() for ticker in tickers.split(',')]
    
    # If only one asset or no assets, silently skip portfolio analysis
    if not tickers or len(tickers) < 2:
        # For single asset analysis, don't show warning, directly return simple analysis result
        if len(tickers) == 1:
            logger.info(f"Single asset analysis mode: {tickers[0]}")
            message_content = {
                "analysis_type": "single_asset",
                "ticker": tickers[0],
                "note": "Single asset analysis, skipping portfolio optimization",
                "portfolio_analysis": None
            }
        else:
            logger.info("No asset codes provided, skipping portfolio analysis")
            message_content = {
                "analysis_type": "no_assets",
                "note": "No asset codes provided, skipping portfolio analysis",
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
    
    # Get time range
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    
    # Call factor_data_api to get returns for multiple stocks
    try:
        logger.info(f"Getting returns for multiple stocks: {tickers}")
        returns_df = get_multi_stock_returns(tickers, start_date, end_date)
        
        if returns_df.empty:
            raise ValueError("Unable to get stock returns data")
            
        # Calculate covariance matrix and expected returns
        logger.info("Calculating covariance matrix and expected returns")
        cov_matrix, expected_returns = get_stock_covariance_matrix(tickers, start_date, end_date)
        
        # Risk-free rate under current market conditions
        risk_free_rate = 0.03  # Can be adjusted based on actual situation
        
        # Portfolio optimization analysis
        portfolio_analysis = analyze_portfolio(tickers, returns_df, cov_matrix, expected_returns, risk_free_rate)
        
        # Risk analysis
        risk_analysis = analyze_portfolio_risk(returns_df)
        
        # Rolling Beta analysis
        beta_analysis = analyze_rolling_betas(tickers, start_date, end_date)
        
        # Generate efficient frontier
        ef_results = generate_efficient_frontier(expected_returns, cov_matrix, risk_free_rate)
        
        # Portfolio analysis results
        message_content = {
            "tickers": tickers,
            "portfolio_analysis": portfolio_analysis,
            "risk_analysis": risk_analysis,
            "beta_analysis": beta_analysis,
            "efficient_frontier": ef_results,
            "summary": generate_summary(portfolio_analysis, risk_analysis, beta_analysis)
        }
        
    except Exception as e:
        logger.error(f"Portfolio analysis failed: {str(e)}")
        message_content = {
            "error": f"Error occurred during portfolio analysis: {str(e)}",
            "tickers": tickers
        }
    
    # Create message
    message = HumanMessage(
        content=json.dumps(message_content),
        name="portfolio_analyzer_agent",
    )
    
    # Show reasoning process
    if show_reasoning:
        show_agent_reasoning(message_content, "Portfolio Analyzer Agent")
        # Save reasoning information to metadata for API use
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
    Analyze various portfolio combinations
    """
    # Calculate equal weight portfolio
    equal_weights = pd.Series(1/len(tickers), index=tickers)
    equal_return = portfolio_return(equal_weights.values, expected_returns.values)
    equal_volatility = portfolio_volatility(equal_weights.values, cov_matrix.values)
    equal_sharpe = portfolio_sharpe_ratio(equal_weights.values, expected_returns.values, cov_matrix.values, risk_free_rate)
    
    # Maximum Sharpe ratio portfolio
    try:
        max_sharpe_portfolio = optimize_portfolio(
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            risk_free_rate=risk_free_rate,
            objective='sharpe'
        )
        
        # Minimum risk portfolio
        min_risk_portfolio = optimize_portfolio(
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            risk_free_rate=risk_free_rate,
            objective='min_risk'
        )
    except Exception as e:
        logger.error(f"Portfolio optimization failed: {e}")
        max_sharpe_portfolio = {'weights': equal_weights, 'return': equal_return, 'risk': equal_volatility, 'sharpe_ratio': equal_sharpe}
        min_risk_portfolio = {'weights': equal_weights, 'return': equal_return, 'risk': equal_volatility, 'sharpe_ratio': equal_sharpe}
    
    # Calculate correlation matrix
    correlation = returns_df.corr()
    
    # Return analysis results
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
    Analyze portfolio risk metrics
    """
    # Calculate equal weight portfolio returns
    portfolio_returns = returns_df.mean(axis=1)
    
    # Calculate risk metrics
    var_95 = calculate_historical_var(portfolio_returns, confidence_level=0.95)
    cvar_95 = calculate_conditional_var(portfolio_returns, confidence_level=0.95)
    
    # Calculate maximum drawdown
    cum_returns = (1 + portfolio_returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max - 1)
    max_drawdown = drawdown.min()
    
    # Calculate skewness and kurtosis
    skewness = portfolio_returns.skew()
    kurtosis = portfolio_returns.kurt()
    
    # Calculate best and worst assets in the portfolio
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
    Calculate rolling Beta coefficients for assets relative to market
    """
    beta_results = {}
    for ticker in tickers:
        try:
            # Use factor_data_api to calculate rolling Beta
            rolling_beta = calculate_rolling_beta(ticker, window, start_date, end_date)
            
            if not rolling_beta.empty:
                # Calculate Beta statistics
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
            logger.error(f"Failed to calculate rolling Beta for {ticker}: {e}")
            beta_results[ticker] = {"error": str(e)}
    
    return beta_results

def generate_efficient_frontier(expected_returns, cov_matrix, risk_free_rate=0.03, points=20):
    """
    Generate efficient frontier
    """
    try:
        ef = efficient_frontier(
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            risk_free_rate=risk_free_rate,
            points=points
        )
        
        # Convert to serializable dictionary
        ef_dict = {
            "returns": ef['return'].tolist(),
            "risks": ef['risk'].tolist(),
            "sharpe_ratios": ef['sharpe_ratio'].tolist()
        }
        
        return ef_dict
    except Exception as e:
        logger.error(f"Failed to generate efficient frontier: {e}")
        return {"error": str(e)}

def generate_summary(portfolio_analysis, risk_analysis, beta_analysis):
    """
    Generate portfolio analysis summary
    """
    # Best portfolio
    max_sharpe = portfolio_analysis["max_sharpe"]
    
    # Risk metrics
    var_95 = risk_analysis["var_95"]
    max_drawdown = risk_analysis["max_drawdown"]
    
    # Build summary
    summary = []
    
    # Optimal allocation recommendation
    summary.append("Optimal Portfolio Allocation (Maximum Sharpe Ratio):")
    for ticker, weight in max_sharpe["weights"].items():
        summary.append(f"- {ticker}: {weight*100:.2f}%")
    
    summary.append(f"This allocation has expected annualized return of {max_sharpe['return']*100:.2f}%, volatility of {max_sharpe['risk']*100:.2f}%, and Sharpe ratio of {max_sharpe['sharpe_ratio']:.2f}")
    
    # Risk assessment
    summary.append(f"Risk Assessment: 95% confidence level VaR is {var_95*100:.2f}%, maximum drawdown is {max_drawdown*100:.2f}%")
    
    # Beta analysis
    summary.append("Beta coefficients of each asset relative to market:")
    for ticker, beta_data in beta_analysis.items():
        if "average_beta" in beta_data:
            summary.append(f"- {ticker}: {beta_data['average_beta']:.2f} (latest: {beta_data['latest_beta']:.2f})")
    
    return "\n".join(summary)