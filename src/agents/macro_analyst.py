from langchain_core.messages import HumanMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.tools.news_crawler import get_stock_news
from src.utils.logging_config import setup_logger
from src.utils.api_utils import agent_endpoint
import json
from datetime import datetime, timedelta
from src.tools.openrouter_config import get_chat_completion

# Setup logging
logger = setup_logger('macro_analyst_agent')


@agent_endpoint("macro_analyst", "Macro analyst, analyzing the impact of macroeconomic environment on target stocks")
def macro_analyst_agent(state: AgentState):
    """Responsible for macroeconomic analysis"""
    show_workflow_status("Macro Analyst")
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    
    # Get asset list
    tickers = data.get("tickers", [])
    if isinstance(tickers, str):
        tickers = [ticker.strip() for ticker in tickers.split(',')]
    
    # If no tickers provided but ticker is provided, use single ticker
    if not tickers and data.get("ticker"):
        tickers = [data["ticker"]]
    
    primary_ticker = tickers[0] if tickers else ""
    symbol = primary_ticker
    logger.info(f"Performing macro analysis: {symbol}")
    
    # Multi-asset macro analysis
    multi_asset_analysis = None
    if len(tickers) > 1:
        try:
            logger.info(f"Executing multi-asset macro analysis: {tickers}")
            # Collect news for all assets
            all_assets_news = []
            for ticker in tickers:
                asset_news = get_stock_news(ticker, max_news=20)
                all_assets_news.extend(asset_news)
            
            # Perform macro analysis on all news
            if all_assets_news:
                multi_asset_analysis = get_macro_news_analysis(all_assets_news)
                logger.info(f"Multi-asset macro analysis completed: {multi_asset_analysis.get('macro_environment', 'neutral')}")
        except Exception as e:
            logger.error(f"Multi-asset macro analysis failed: {e}")
    
    # Get large amount of news data (up to 100 articles)
    news_list = get_stock_news(symbol, max_news=100)  # Try to get 100 news articles
    
    # Filter news from seven days ago
    cutoff_date = datetime.now() - timedelta(days=7)
    recent_news = [news for news in news_list
                  if datetime.strptime(news['publish_time'], '%Y-%m-%d %H:%M:%S') > cutoff_date]
    
    logger.info(f"Retrieved {len(recent_news)} news articles from the past seven days")
    
    # If no news retrieved, use multi-asset analysis results or return default result
    if not recent_news:
        logger.warning(f"No recent news retrieved for {symbol}")
        if multi_asset_analysis:
            logger.info("Using multi-asset macro analysis results")
            macro_analysis = multi_asset_analysis
        else:
            logger.warning("Unable to perform macro analysis")
            macro_analysis = {
                "macro_environment": "neutral",
                "impact_on_stock": "neutral",
                "key_factors": [],
                "reasoning": "No recent news retrieved, unable to perform macro analysis"
            }
    else:
        # Get macro analysis results
        macro_analysis = get_macro_news_analysis(recent_news)
    
    # If reasoning process needs to be displayed
    if show_reasoning:
        show_agent_reasoning(macro_analysis, "Macro Analysis Agent")
        # Save reasoning information to metadata for API use
        state["metadata"]["agent_reasoning"] = macro_analysis
    
    # Add multi-asset analysis information
    if multi_asset_analysis and len(tickers) > 1:
        message_content = {
            "signal": macro_analysis.get("impact_on_stock", "neutral"),
            "confidence": 0.7 if macro_analysis.get("impact_on_stock") == "positive" else (
                        0.3 if macro_analysis.get("impact_on_stock") == "negative" else 0.5),
            "macro_environment": macro_analysis.get("macro_environment", "neutral"),
            "impact_on_stock": macro_analysis.get("impact_on_stock", "neutral"),
            "key_factors": macro_analysis.get("key_factors", []),
            "reasoning": macro_analysis.get("reasoning", "No macro analysis reasoning provided"),
            "multi_asset_analysis": multi_asset_analysis,
            "tickers_analyzed": tickers,
            "summary": "\n".join([
                f"Macro Environment: {macro_analysis.get('macro_environment', 'neutral')}",
                f"Impact on Stock: {macro_analysis.get('impact_on_stock', 'neutral')}",
                "Key Factors:",
                *[f"- {factor}" for factor in macro_analysis.get("key_factors", [])]
            ])
        }
    else:
        message_content = {
            "signal": macro_analysis.get("impact_on_stock", "neutral"),
            "confidence": 0.7 if macro_analysis.get("impact_on_stock") == "positive" else (
                        0.3 if macro_analysis.get("impact_on_stock") == "negative" else 0.5),
            "macro_environment": macro_analysis.get("macro_environment", "neutral"),
            "impact_on_stock": macro_analysis.get("impact_on_stock", "neutral"),
            "key_factors": macro_analysis.get("key_factors", []),
            "reasoning": macro_analysis.get("reasoning", "No macro analysis reasoning provided"),
            "summary": "\n".join([
                f"Macro Environment: {macro_analysis.get('macro_environment', 'neutral')}",
                f"Impact on Stock: {macro_analysis.get('impact_on_stock', 'neutral')}",
                "Key Factors:",
                *[f"- {factor}" for factor in macro_analysis.get("key_factors", [])]
            ])
        }
    
    # Create message
    message = HumanMessage(
        content=json.dumps(message_content),
        name="macro_analyst_agent",
    )
    
    show_workflow_status("Macro Analyst", "completed")
    return {
        "messages": [message],
        "data": {
            **data,
            "macro_analysis": message_content
        },
        "metadata": state["metadata"],
    }


def get_macro_news_analysis(news_list: list) -> dict:
    """Analyze the impact of macroeconomic news on stocks
    
    Args:
        news_list (list): News list
        
    Returns:
        dict: Macro analysis results, including environment assessment, impact on stocks, key factors and detailed reasoning
    """
    if not news_list:
        return {
            "macro_environment": "neutral",
            "impact_on_stock": "neutral",
            "key_factors": [],
            "reasoning": "Insufficient news data for macro analysis"
        }
    
    # Check cache
    import os
    cache_file = "src/data/macro_analysis_cache.json"
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    
    # Generate unique identifier for news content
    news_key = "|".join([
        f"{news['title']}|{news['publish_time']}"
        for news in news_list[:20]  # Use first 20 news items as identifier
    ])
    
    # Check cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
                if news_key in cache:
                    logger.info("Using cached macro analysis results")
                    return cache[news_key]
        except Exception as e:
            logger.error(f"Error reading macro analysis cache: {e}")
            cache = {}
    else:
        logger.info("Macro analysis cache file not found, will create new file")
        cache = {}
    
    # Prepare system message
    system_message = {
        "role": "system",
        "content": """You are a professional macroeconomic analyst specializing in analyzing the impact of macroeconomic environment on A-share individual stocks.
        Please analyze the provided news, assess the current economic environment from a macro perspective, and analyze the potential impact of these macro factors on target stocks.
        
        Please focus on the following macro factors:
        1. Monetary policy: interest rates, reserve requirements, open market operations, etc.
        2. Fiscal policy: government spending, tax policies, subsidies, etc.
        3. Industrial policy: industry planning, regulatory policies, environmental requirements, etc.
        4. International environment: global economic situation, trade relations, geopolitics, etc.
        5. Market sentiment: investor confidence, market liquidity, risk appetite, etc.
        
        Your analysis should include:
        1. Macro environment assessment: positive, neutral, or negative
        2. Impact on target stock: positive, neutral, or negative
        3. Key influencing factors: list 3-5 most important macro factors
        4. Detailed reasoning: explain why these factors would affect the target stock
        
        Please ensure your analysis:
        1. Is based on facts and data, not speculation
        2. Considers industry characteristics and company features
        3. Focuses on medium to long-term impact, not short-term fluctuations
        4. Provides specific, actionable insights"""
    }
    
    # Prepare news content
    news_content = "\n\n".join([
        f"Title: {news['title']}\n"
        f"Source: {news['source']}\n"
        f"Time: {news['publish_time']}\n"
        f"Content: {news['content']}"
        for news in news_list[:50]  # Use first 50 news items for analysis
    ])
    
    user_message = {
        "role": "user",
        "content": f"Please analyze the following news, assess the current macroeconomic environment and its impact on related A-share listed companies:\n\n{news_content}\n\nPlease return results in JSON format with the following fields: macro_environment (macro environment: positive/neutral/negative), impact_on_stock (impact on stock: positive/neutral/negative), key_factors (key factors array), reasoning (detailed reasoning)."
    }
    
    try:
        # Get LLM analysis results
        logger.info("Calling LLM for macro analysis...")
        result = get_chat_completion([system_message, user_message])
        if result is None:
            logger.error("LLM analysis failed, unable to get macro analysis results")
            return {
                "macro_environment": "neutral",
                "impact_on_stock": "neutral",
                "key_factors": [],
                "reasoning": "LLM analysis failed, unable to get macro analysis results"
            }
        
        # Parse JSON results
        try:
            # Try direct parsing
            analysis_result = json.loads(result.strip())
            logger.info("Successfully parsed LLM returned JSON results")
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON part
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
            if json_match:
                try:
                    analysis_result = json.loads(json_match.group(1).strip())
                    logger.info("Successfully extracted and parsed JSON results from code block")
                except:
                    # If still fails, return default result
                    logger.error("Unable to parse JSON results in code block")
                    return {
                        "macro_environment": "neutral",
                        "impact_on_stock": "neutral",
                        "key_factors": [],
                        "reasoning": "Unable to parse LLM returned JSON results"
                    }
            else:
                # If no JSON found, return default result
                logger.error("LLM did not return valid JSON format results")
                return {
                    "macro_environment": "neutral",
                    "impact_on_stock": "neutral",
                    "key_factors": [],
                    "reasoning": "LLM did not return valid JSON format results"
                }
        
        # Cache results
        cache[news_key] = analysis_result
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            logger.info("Macro analysis results cached")
        except Exception as e:
            logger.error(f"Error writing macro analysis cache: {e}")
        
        return analysis_result
    
    except Exception as e:
        logger.error(f"Macro analysis error: {e}")
        return {
            "macro_environment": "neutral",
            "impact_on_stock": "neutral",
            "key_factors": [],
            "reasoning": f"Error during analysis process: {str(e)}"
        }