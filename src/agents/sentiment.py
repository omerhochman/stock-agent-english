from langchain_core.messages import HumanMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.tools.news_crawler import get_stock_news, get_news_sentiment
from src.utils.logging_config import setup_logger
from src.utils.api_utils import agent_endpoint
import json
from datetime import datetime, timedelta

# Setup logging
logger = setup_logger('sentiment_agent')


@agent_endpoint("sentiment", "Sentiment analyst, analyzing market news and social media sentiment")
def sentiment_agent(state: AgentState):
    """Responsible for sentiment analysis"""
    show_workflow_status("Sentiment Analyst")
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    symbol = data["ticker"]
    logger.info(f"Analyzing stock: {symbol}")
    # Get number of news from command line arguments, default to 5
    num_of_news = data.get("num_of_news", 5)

    # Get news data and analyze sentiment
    news_list = get_stock_news(symbol, max_news=num_of_news)  # Ensure getting enough news

    # Filter news within 7 days
    cutoff_date = datetime.now() - timedelta(days=7)
    recent_news = [news for news in news_list
                   if datetime.strptime(news['publish_time'], '%Y-%m-%d %H:%M:%S') > cutoff_date]

    sentiment_score = get_news_sentiment(recent_news, num_of_news=num_of_news)

    # Generate trading signal and confidence based on sentiment score
    if sentiment_score >= 0.5:
        signal = "bullish"
        confidence = str(round(abs(sentiment_score) * 100)) + "%"
    elif sentiment_score <= -0.5:
        signal = "bearish"
        confidence = str(round(abs(sentiment_score) * 100)) + "%"
    else:
        signal = "neutral"
        confidence = str(round((1 - abs(sentiment_score)) * 100)) + "%"

    # Generate analysis results
    message_content = {
        "signal": signal,
        "confidence": confidence,
        "reasoning": f"Based on {len(recent_news)} recent news articles, sentiment score: {sentiment_score:.2f}"
    }

    # If reasoning process needs to be displayed
    if show_reasoning:
        show_agent_reasoning(message_content, "Sentiment Analysis Agent")
        # Save reasoning information to metadata for API use
        state["metadata"]["agent_reasoning"] = message_content

    # Create message
    message = HumanMessage(
        content=json.dumps(message_content),
        name="sentiment_agent",
    )

    show_workflow_status("Sentiment Analyst", "completed")
    return {
        "messages": [message],
        "data": {
            **data,
            "sentiment_analysis": sentiment_score
        },
        "metadata": state["metadata"],
    }
