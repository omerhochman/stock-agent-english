import json
import os
import sys
import time
from datetime import datetime

import akshare as ak
import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.tools.openrouter_config import get_chat_completion
from src.tools.openrouter_config import logger as api_logger


def get_stock_news(symbol: str, max_news: int = 10) -> list:
    """Get and process individual stock news

    Args:
        symbol (str): Stock code, e.g. "300059"
        max_news (int, optional): Number of news items to fetch, default 10. Maximum supported is 100.

    Returns:
        list: News list, each news item contains title, content, publish time and other information
    """

    # Set pandas display options to ensure complete content display
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", None)

    # Limit maximum number of news items
    max_news = min(max_news, 100)

    # Get current date
    today = datetime.now().strftime("%Y-%m-%d")

    # Build news file path
    # project_root = os.path.dirname(os.path.dirname(
    #     os.path.dirname(os.path.abspath(__file__))))
    news_dir = os.path.join("src", "data", "stock_news")
    print(f"News save directory: {news_dir}")

    # Ensure directory exists
    try:
        os.makedirs(news_dir, exist_ok=True)
        print(f"Successfully created or confirmed directory exists: {news_dir}")
    except Exception as e:
        print(f"Failed to create directory: {e}")
        return []

    news_file = os.path.join(news_dir, f"{symbol}_news.json")
    print(f"News file path: {news_file}")

    # Check if news needs to be updated
    need_update = True
    if os.path.exists(news_file):
        try:
            with open(news_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("date") == today:
                    cached_news = data.get("news", [])
                    if len(cached_news) >= max_news:
                        print(f"Using cached news data: {news_file}")
                        return cached_news[:max_news]
                    else:
                        print(
                            f"Cached news count ({len(cached_news)}) insufficient, need to fetch more news ({max_news} items)"
                        )
        except Exception as e:
            print(f"Failed to read cache file: {e}")

    print(f"Starting to fetch news data for {symbol}...")

    try:
        # Get news list
        news_df = ak.stock_news_em(symbol=symbol)
        if news_df is None or len(news_df) == 0:
            print(f"No news data obtained for {symbol}")
            return []

        print(f"Successfully obtained {len(news_df)} news items")

        # Actual available news count
        available_news_count = len(news_df)
        if available_news_count < max_news:
            print(
                f"Warning: Actual available news count ({available_news_count}) is less than requested ({max_news})"
            )
            max_news = available_news_count

        # Get specified number of news items (considering some news content might be empty, fetch 50% more)
        news_list = []
        for _, row in news_df.head(int(max_news * 1.5)).iterrows():
            try:
                # Get news content
                content = (
                    row["news_content"]
                    if "news_content" in row and not pd.isna(row["news_content"])
                    else ""
                )
                if not content:
                    content = row["news_title"]

                # Only remove leading and trailing whitespace
                content = content.strip()
                if len(content) < 10:  # Skip content that's too short
                    continue

                # Get keywords
                keyword = (
                    row["keywords"]
                    if "keywords" in row and not pd.isna(row["keywords"])
                    else ""
                )

                # Add news
                news_item = {
                    "title": row["news_title"].strip(),
                    "content": content,
                    "publish_time": row["publish_time"],
                    "source": row["news_source"].strip(),
                    "url": row["news_url"].strip(),
                    "keyword": keyword.strip(),
                }
                news_list.append(news_item)
                print(f"Successfully added news: {news_item['title']}")

            except Exception as e:
                print(f"Error processing individual news item: {e}")
                continue

        # Sort by publish time
        news_list.sort(key=lambda x: x["publish_time"], reverse=True)

        # Keep only the specified number of valid news items
        news_list = news_list[:max_news]

        # Save to file
        try:
            save_data = {"date": today, "news": news_list}
            with open(news_file, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            print(
                f"Successfully saved {len(news_list)} news items to file: {news_file}"
            )
        except Exception as e:
            print(f"Error saving news data to file: {e}")

        return news_list

    except Exception as e:
        print(f"Error getting news data: {e}")
        return []


def get_news_sentiment(news_list: list, num_of_news: int = 5) -> float:
    """Analyze news sentiment score

    Args:
        news_list (list): News list
        num_of_news (int): Number of news items for analysis, default 5

    Returns:
        float: Sentiment score, range [-1, 1], -1 most negative, 1 most positive
    """
    if not news_list:
        return 0.0

    # # Get project root directory
    # project_root = os.path.dirname(os.path.dirname(
    #     os.path.dirname(os.path.abspath(__file__))))

    # Check if there are cached sentiment analysis results
    # Check if there are cached sentiment analysis results
    cache_file = "src/data/sentiment_cache.json"
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    # Generate unique identifier for news content
    news_key = "|".join(
        [
            f"{news['title']}|{news['content'][:100]}|{news['publish_time']}"
            for news in news_list[:num_of_news]
        ]
    )

    # Check cache
    if os.path.exists(cache_file):
        print("Found sentiment analysis cache file")
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)
                if news_key in cache:
                    print("Using cached sentiment analysis results")
                    return cache[news_key]
                print("No matching sentiment analysis cache found")
        except Exception as e:
            print(f"Error reading sentiment analysis cache: {e}")
            cache = {}
    else:
        print("No sentiment analysis cache file found, will create new file")
        cache = {}

    # Prepare system message
    system_message = {
        "role": "system",
        "content": """You are a professional A-share market analyst, skilled at interpreting the impact of news on stock price movements. You need to analyze the sentiment tendency of a group of news and give a score between -1 and 1:
        - 1 means extremely positive (e.g., major positive news, better-than-expected performance, industry policy support)
        - 0.5 to 0.9 means positive (e.g., performance growth, new project launch, order acquisition)
        - 0.1 to 0.4 means slightly positive (e.g., small contract signing, normal daily operations)
        - 0 means neutral (e.g., daily announcements, personnel changes, news with no significant impact)
        - -0.1 to -0.4 means slightly negative (e.g., small lawsuits, non-core business losses)
        - -0.5 to -0.9 means negative (e.g., performance decline, important customer loss, industry policy tightening)
        - -1 means extremely negative (e.g., major violations, severe core business losses, regulatory penalties)

        Focus on the following when analyzing:
        1. Performance-related: financial reports, performance forecasts, revenue and profit, etc.
        2. Policy impact: industry policies, regulatory policies, local policies, etc.
        3. Market performance: market share, competitive situation, business model, etc.
        4. Capital operations: M&A, equity incentives, private placements, etc.
        5. Risk events: litigation, penalties, debt, etc.
        6. Industry position: technological innovation, patents, market share, etc.
        7. Public opinion environment: media evaluation, social impact, etc.

        Please ensure analysis considers:
        1. Authenticity and reliability of news
        2. Timeliness and scope of impact of news
        3. Actual impact on company fundamentals
        4. Special reaction patterns of A-share market""",
    }

    # Prepare news content
    news_content = "\n\n".join(
        [
            f"Title: {news['title']}\n"
            f"Source: {news['source']}\n"
            f"Time: {news['publish_time']}\n"
            f"Content: {news['content']}"
            for news in news_list[:num_of_news]  # Use specified number of news items
        ]
    )

    user_message = {
        "role": "user",
        "content": f"Please analyze the sentiment tendency of the following A-share listed company related news:\n\n{news_content}\n\nPlease directly return a number, range -1 to 1, no explanation needed.",
    }

    try:
        # Get LLM analysis results
        result = get_chat_completion([system_message, user_message])
        if result is None:
            print("Error: PI error occurred, LLM returned None")
            return 0.0

        # Extract numeric result
        try:
            sentiment_score = float(result.strip())
        except ValueError as e:
            print(f"Error parsing sentiment score: {e}")
            print(f"Raw result: {result}")
            return 0.0

        # Ensure score is between -1 and 1
        sentiment_score = max(-1.0, min(1.0, sentiment_score))

        # Cache result
        cache[news_key] = sentiment_score
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error writing cache: {e}")

        return sentiment_score

    except Exception as e:
        print(f"Error analyzing news sentiment: {e}")
        return 0.0  # Return neutral score when error occurs
