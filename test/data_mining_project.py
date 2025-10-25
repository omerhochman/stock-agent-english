import base64
import io
import json
import os
import re
import threading
import warnings
from datetime import datetime, timedelta

import baostock as bs
import jieba
import jieba.analyse
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import pymysql
import requests
import seaborn as sns
import streamlit as st
import talib
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from wordcloud import WordCloud

warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(
    page_title="Financial Data Analysis and Prediction System",
    page_icon="💹",
    layout="wide",
)

# Plotting settings
plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "SimSun",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# Ensure matplotlib can correctly display Chinese
import matplotlib

matplotlib.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "SimSun"]
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun"]
matplotlib.rcParams["axes.unicode_minus"] = False

# AI model API configuration
LLM_API_CONFIG = {
    "api_key": st.secrets.get(
        "LLM_API_KEY", "sk-4DKrMT5Du1BR6e1eYvru5kjb7u7FBzPR59cVyChZrC7SJZcg"
    ),
    "base_url": st.secrets.get("LLM_API_BASE_URL", "https://yunwu.ai/v1"),
    "model_name": st.secrets.get("LLM_MODEL_NAME", "gpt-4o"),
    "timeout": 30,
}

# Configure MySQL database connection
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "db": "financial_analysis",
    "charset": "utf8mb4",
}


# Create database connection
def create_connection():
    try:
        conn = pymysql.connect(
            host=DB_CONFIG["host"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            db=DB_CONFIG["db"],
            charset="utf8mb4",
            use_unicode=True,
        )
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None


# Initialize database
def init_database():
    try:
        conn = pymysql.connect(
            host=DB_CONFIG["host"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            charset=DB_CONFIG["charset"],
        )
        cursor = conn.cursor()

        # Create database
        cursor.execute("CREATE DATABASE IF NOT EXISTS financial_analysis")
        cursor.execute("USE financial_analysis")

        # Create stock data table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS stock_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
            open FLOAT,
            high FLOAT,
            low FLOAT,
            close FLOAT,
            volume BIGINT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX (symbol, date)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
        )

        # Create news data table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS news_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            title VARCHAR(255) NOT NULL,
            content TEXT,
            source VARCHAR(100),
            date DATE,
            sentiment FLOAT,
            keywords VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX (date)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
        )

        # Create prediction results table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS prediction_results (
            id INT AUTO_INCREMENT PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            model_name VARCHAR(50) NOT NULL,
            prediction_date DATE NOT NULL,
            predicted_value FLOAT,
            actual_value FLOAT,
            accuracy FLOAT,
            model_params TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX (symbol, prediction_date)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
        )

        # Create index data table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS index_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            index_code VARCHAR(20) NOT NULL,
            index_name VARCHAR(50) NOT NULL,
            date DATE NOT NULL,
            open FLOAT,
            high FLOAT,
            low FLOAT,
            close FLOAT,
            volume BIGINT,
            amount FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX (index_code, date)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
        )

        # Create financial data table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS financial_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            year INT NOT NULL,
            quarter INT NOT NULL,
            report_type VARCHAR(20) NOT NULL,
            report_date DATE NOT NULL,
            data_type VARCHAR(50) NOT NULL,
            data_value FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX (symbol, year, quarter, report_type)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
        )

        conn.commit()
        cursor.close()
        conn.close()

        st.success("Database initialization successful!")
    except Exception as e:
        st.error(f"Database initialization failed: {e}")


# Use generic LLM API for sentiment analysis
def analyze_sentiment_with_llm(text):
    """
    Use generic LLM API for text sentiment analysis
    Returns value range from -1 (extremely negative) to 1 (extremely positive)
    """
    try:
        # Truncate text to avoid excessive length
        if len(text) > 1000:
            text = text[:1000]

        # API request headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLM_API_CONFIG['api_key']}",
        }

        # API request body
        payload = {
            "model": LLM_API_CONFIG["model_name"],
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional financial sentiment analysis AI. Please analyze the sentiment tendency of the following financial news and score its impact on the stock market/enterprise. Score range from -1 (extremely negative) to 1 (extremely positive), only return a number, no explanation.",
                },
                {
                    "role": "user",
                    "content": f"Perform sentiment analysis scoring (-1 to 1) on the following financial news, only return a number:\n\n{text}",
                },
            ],
            "temperature": 0.3,
        }

        # Send request
        response = requests.post(
            f"{LLM_API_CONFIG['base_url']}/chat/completions",
            headers=headers,
            json=payload,
            timeout=LLM_API_CONFIG["timeout"],
        )

        # Check response status
        if response.status_code == 200:
            response_data = response.json()
            sentiment_score = response_data["choices"][0]["message"]["content"].strip()

            # Ensure result is a number
            try:
                sentiment_score = float(sentiment_score)
                # Ensure within -1 to 1 range
                sentiment_score = max(-1, min(1, sentiment_score))
            except ValueError:
                # If parsing fails, extract number from text
                import re

                match = re.search(r"(-?\d+(\.\d+)?)", sentiment_score)
                if match:
                    sentiment_score = float(match.group(1))
                    sentiment_score = max(-1, min(1, sentiment_score))
                else:
                    # Default to neutral
                    sentiment_score = 0.0
        else:
            st.warning(
                f"API request failed, status code: {response.status_code}, using neutral score"
            )
            sentiment_score = 0.0

        return sentiment_score

    except Exception as e:
        st.warning(f"Sentiment analysis API call failed: {e}, using neutral score")
        return 0.0  # Return neutral score on error


# News summary and key information extraction through LLM API
def extract_news_insights_with_llm(news_df):
    """
    Use LLM API to perform summary analysis on a group of news
    Returns key insights and market trend analysis
    """
    try:
        # Select the latest 10 news titles
        recent_news = news_df.sort_values("date", ascending=False).head(10)
        titles = recent_news["title"].tolist()

        # Build news summary request
        titles_text = "\n".join([f"{i+1}. {title}" for i, title in enumerate(titles)])

        # API request headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLM_API_CONFIG['api_key']}",
        }

        # API request body
        payload = {
            "model": LLM_API_CONFIG["model_name"],
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional financial analyst. Please analyze the following financial news titles and extract key market trends and insights.",
                },
                {
                    "role": "user",
                    "content": f"Please analyze the following financial news titles, identify main market trends, potential investment opportunities and risks. Provide concise analysis (within 200 words):\n\n{titles_text}",
                },
            ],
            "temperature": 0.7,
        }

        # Send request
        response = requests.post(
            f"{LLM_API_CONFIG['base_url']}/chat/completions",
            headers=headers,
            json=payload,
            timeout=LLM_API_CONFIG["timeout"],
        )

        # Check response status
        if response.status_code == 200:
            response_data = response.json()
            return response_data["choices"][0]["message"]["content"].strip()
        else:
            st.warning(f"API request failed, status code: {response.status_code}")
            return "Unable to get news analysis results, please check API connection."

    except Exception as e:
        st.warning(f"News analysis API call failed: {e}")
        return "Unable to get news analysis results, please check API connection."


# Use LLM API to generate prediction result interpretation
def interpret_prediction_with_llm(stock_name, prediction_data, model_name, metrics):
    """
    Use LLM API to interpret prediction results
    Generate professional analysis report
    """
    try:
        # Format prediction data
        prediction_text = f"Model: {model_name}\n"
        prediction_text += f"Prediction accuracy/error: {metrics}\n"
        prediction_text += f"Latest prediction value: {prediction_data[-1]:.4f}\n"

        # API request headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLM_API_CONFIG['api_key']}",
        }

        # API request body
        payload = {
            "model": LLM_API_CONFIG["model_name"],
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional financial analyst, skilled at interpreting stock prediction model results.",
                },
                {
                    "role": "user",
                    "content": f"Please interpret the following {stock_name} stock prediction results, provide professional analysis insights and possible investment advice (around 200 words):\n\n{prediction_text}",
                },
            ],
            "temperature": 0.7,
        }

        # Send request
        response = requests.post(
            f"{LLM_API_CONFIG['base_url']}/chat/completions",
            headers=headers,
            json=payload,
            timeout=LLM_API_CONFIG["timeout"],
        )

        # Check response status
        if response.status_code == 200:
            response_data = response.json()
            return response_data["choices"][0]["message"]["content"].strip()
        else:
            st.warning(f"API request failed, status code: {response.status_code}")
            return "Unable to get prediction analysis results, please check API connection."

    except Exception as e:
        st.warning(f"Prediction interpretation API call failed: {e}")
        return "Unable to get prediction analysis results, please check API connection."


# Get stock data from BaoStock
def fetch_stock_data(symbol, start_date, end_date):
    """Get stock data from BaoStock"""
    try:
        # Check if data already exists in database
        conn = create_connection()
        cursor = conn.cursor()

        cursor.execute(
            f"""
        SELECT * FROM stock_data 
        WHERE symbol = '{symbol}' 
        AND date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY date ASC
        """
        )

        result = cursor.fetchall()

        # If database already has complete data, return directly
        if len(result) > 0:
            df = pd.DataFrame(
                result,
                columns=[
                    "id",
                    "symbol",
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "created_at",
                ],
            )
            df = df.drop(["id", "created_at"], axis=1)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            return df

        # Use BaoStock to get data
        # Login to BaoStock
        lg = bs.login()
        if lg.error_code != "0":
            st.error(f"BaoStock login failed: {lg.error_msg}")
            return None

        # Build complete stock code
        if symbol.startswith("6"):
            bs_symbol = f"sh.{symbol}"
        elif symbol.startswith(("0", "3")):
            bs_symbol = f"sz.{symbol}"
        else:
            bs_symbol = symbol

        # Get stock data
        rs = bs.query_history_k_data_plus(
            bs_symbol,
            "date,open,high,low,close,volume,amount,adjustflag",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3",  # Adjustment type: 3-forward adjustment 2-backward adjustment 1-no adjustment
        )

        if rs.error_code != "0":
            st.error(f"Failed to get stock data: {rs.error_msg}")
            bs.logout()
            return None

        # Process data
        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())

        # Logout from BaoStock
        bs.logout()

        if not data_list:
            st.warning(f"No data found for {symbol} from {start_date} to {end_date}")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(
            data_list,
            columns=[
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "adjustflag",
            ],
        )

        # Convert data types
        df["date"] = pd.to_datetime(df["date"])
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)

        # Set date as index
        df.set_index("date", inplace=True)

        # Save to database
        for idx, row in df.iterrows():
            sql = """
            INSERT INTO stock_data (symbol, date, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(
                sql,
                (
                    symbol,
                    idx.strftime("%Y-%m-%d"),
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row["volume"]),
                ),
            )

        conn.commit()
        cursor.close()
        conn.close()

        return df
    except Exception as e:
        st.error(f"Failed to get stock data: {e}")
        try:
            bs.logout()  # Ensure logout from BaoStock
        except:
            pass
        return None


def get_stock_basic_info(symbol):
    """Get stock basic information"""
    result = [None]  # Use list to store result for modification in thread
    done_event = threading.Event()  # Used to mark if operation is complete

    def _get_info():
        try:
            # Format stock code
            if symbol.startswith("6"):
                bs_symbol = f"sh.{symbol}"
            elif symbol.startswith(("0", "3")):
                bs_symbol = f"sz.{symbol}"
            else:
                bs_symbol = symbol

            # Login to BaoStock
            lg = bs.login()
            if lg.error_code != "0":
                st.error(f"BaoStock login failed: {lg.error_msg}")
                done_event.set()
                return

            # Get stock basic information
            rs = bs.query_stock_basic(code=bs_symbol)

            if rs is None or rs.error_code != "0":
                error_msg = rs.error_msg if rs is not None else "Query failed"
                st.error(f"Failed to get stock information: {error_msg}")
                bs.logout()
                done_event.set()
                return

            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())

            # Logout from BaoStock
            bs.logout()

            if data_list:
                result[0] = data_list[0]

            # Mark operation as complete
            done_event.set()
        except Exception as e:
            st.error(f"Failed to get stock basic information: {e}")
            try:
                bs.logout()
            except:
                pass
            done_event.set()

    # Create and start thread
    thread = threading.Thread(target=_get_info)
    thread.daemon = True
    thread.start()

    # Wait for operation to complete or timeout
    if not done_event.wait(timeout=10):
        st.error("Timeout getting stock basic information (10 seconds)")
        try:
            bs.logout()
        except:
            pass

    return result[0]


# Get all stock list
def get_stock_list():
    """Get all A-share stock list"""
    try:
        # Login to BaoStock
        lg = bs.login()
        if lg.error_code != "0":
            st.error(f"BaoStock login failed: {lg.error_msg}")
            return []

        # Get A-share code list
        rs = bs.query_stock_basic()

        if rs.error_code != "0":
            st.error(f"Failed to get stock list: {rs.error_msg}")
            bs.logout()
            return []

        data_list = []
        while rs.next():
            row = rs.get_row_data()
            # Only select A-shares with listing status
            if (
                row[4] == "1" and row[5] == "1"
            ):  # type=1 means A-share, status=1 means listed
                data_list.append({"code": row[0], "name": row[1]})

        # Logout from BaoStock
        bs.logout()

        return data_list
    except Exception as e:
        st.error(f"Failed to get stock list: {e}")
        try:
            bs.logout()
        except:
            pass
        return []


# Get industry classification data
def get_industry_data():
    """Get industry classification data"""
    try:
        # Login to BaoStock
        lg = bs.login()
        if lg.error_code != "0":
            st.error(f"BaoStock login failed: {lg.error_msg}")
            return None

        # Get industry classification data
        rs = bs.query_stock_industry()

        if rs.error_code != "0":
            st.error(f"Failed to get industry classification: {rs.error_msg}")
            bs.logout()
            return None

        industry_list = []
        while rs.next():
            row_data = rs.get_row_data()
            # Ensure data column count is consistent
            if len(row_data) == 5:  # If API returns 5 columns
                # Adjust based on actual API situation, may need to remove extra columns or merge columns
                # Assume first column is index column, we remove it
                industry_list.append(row_data[1:])
            else:
                industry_list.append(row_data)

        # Logout from BaoStock
        bs.logout()

        if industry_list:
            # Dynamically get column names
            first_row = industry_list[0]
            if len(first_row) == 4:
                columns = ["code", "code_name", "industry", "industry_classification"]
            else:
                # If data format changes, need to adapt new column names
                columns = ["code", "code_name", "industry", "industry_classification"]
                # If actual column count doesn't match, need to adjust
                columns = columns[: len(first_row)]

            df = pd.DataFrame(industry_list, columns=columns)
            return df
        else:
            return None
    except Exception as e:
        st.error(f"Failed to get industry classification data: {e}")
        try:
            bs.logout()
        except:
            pass
        return None


# Get index data
def fetch_index_data(index_code, start_date, end_date):
    """Get index data"""
    try:
        # Check if data already exists in database
        conn = create_connection()
        cursor = conn.cursor()

        index_name = {
            "sh.000001": "Shanghai Composite Index",
            "sz.399001": "Shenzhen Component Index",
            "sz.399006": "ChiNext Index",
            "sh.000016": "SSE 50",
            "sh.000300": "CSI 300",
            "sz.399905": "CSI 500",
        }.get(index_code, index_code)

        cursor.execute(
            f"""
        SELECT * FROM index_data 
        WHERE index_code = '{index_code}' 
        AND date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY date ASC
        """
        )

        result = cursor.fetchall()

        # If database already has complete data, return directly
        if len(result) > 0:
            df = pd.DataFrame(
                result,
                columns=[
                    "id",
                    "index_code",
                    "index_name",
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "amount",
                    "created_at",
                ],
            )
            df = df.drop(["id", "index_code", "index_name", "created_at"], axis=1)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            return df

        # Login to BaoStock
        lg = bs.login()
        if lg.error_code != "0":
            st.error(f"BaoStock login failed: {lg.error_msg}")
            return None

        # Get index data
        rs = bs.query_history_k_data_plus(
            index_code,
            "date,open,high,low,close,volume,amount",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
        )

        if rs.error_code != "0":
            st.error(f"Failed to get index data: {rs.error_msg}")
            bs.logout()
            return None

        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())

        # Logout from BaoStock
        bs.logout()

        if not data_list:
            st.warning(
                f"No data found for index {index_code} from {start_date} to {end_date}"
            )
            return None

        # Convert to DataFrame
        df = pd.DataFrame(
            data_list,
            columns=["date", "open", "high", "low", "close", "volume", "amount"],
        )

        # Convert data types
        df["date"] = pd.to_datetime(df["date"])
        numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Set date as index
        df.set_index("date", inplace=True)

        # Save to database
        for idx, row in df.iterrows():
            sql = """
            INSERT INTO index_data (index_code, index_name, date, open, high, low, close, volume, amount)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(
                sql,
                (
                    index_code,
                    index_name,
                    idx.strftime("%Y-%m-%d"),
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row["volume"]),
                    float(row["amount"]),
                ),
            )

        conn.commit()
        cursor.close()
        conn.close()

        return df
    except Exception as e:
        st.error(f"Failed to get index data: {e}")
        try:
            bs.logout()
        except:
            pass
        return None


# Get trading calendar
def get_trade_calendar(start_date, end_date):
    """Get trading calendar"""
    try:
        # Login to BaoStock
        lg = bs.login()
        if lg.error_code != "0":
            st.error(f"BaoStock login failed: {lg.error_msg}")
            return None

        # Get trading calendar
        rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)

        if rs.error_code != "0":
            st.error(f"Failed to get trading calendar: {rs.error_msg}")
            bs.logout()
            return None

        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())

        # Logout from BaoStock
        bs.logout()

        if data_list:
            df = pd.DataFrame(data_list, columns=["calendar_date", "is_trading_day"])
            df["calendar_date"] = pd.to_datetime(df["calendar_date"])
            df["is_trading_day"] = df["is_trading_day"].apply(
                lambda x: True if x == "1" else False
            )
            return df
        else:
            return None
    except Exception as e:
        st.error(f"Failed to get trading calendar: {e}")
        try:
            bs.logout()
        except:
            pass
        return None


# Crawl news data
def fetch_news_data(keyword, num_pages=2):
    """Crawl Sina Finance news data"""
    try:
        all_news = []

        for page in range(1, num_pages + 1):
            url = f"https://search.sina.com.cn/?q={keyword}&c=news&from=&col=&range=&source=&country=&size=&time=&a=&page={page}&pf=0&ps=0&dpc=1"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")

            news_items = soup.select(".box-result")

            for item in news_items:
                try:
                    title_elem = item.select_one("h2 a")
                    if not title_elem:
                        continue

                    title = title_elem.text.strip()
                    link = title_elem["href"]

                    # Extract date
                    time_elem = item.select_one(".fgray_time")
                    news_date = datetime.now().strftime("%Y-%m-%d")
                    if time_elem:
                        date_str = time_elem.text.strip()
                        if (
                            "年" in date_str and "月" in date_str and "日" in date_str
                        ):  # Check for Chinese date format (year, month, day)
                            date_match = re.search(
                                r"(\d{4})年(\d{1,2})月(\d{1,2})日", date_str
                            )  # Parse Chinese date format: YYYY年MM月DD日
                            if date_match:
                                year, month, day = date_match.groups()
                                news_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"

                    # Get news content
                    content = ""
                    try:
                        news_response = requests.get(link, headers=headers, timeout=5)
                        news_soup = BeautifulSoup(news_response.text, "html.parser")
                        content_elem = news_soup.select_one(
                            ".article-content"
                        ) or news_soup.select_one("#artibody")
                        if content_elem:
                            paras = content_elem.select("p")
                            content = "\n".join([p.text.strip() for p in paras])
                    except Exception as e:
                        content = "Failed to get content"

                    # Extract keywords
                    keywords = jieba.analyse.extract_tags(
                        title + " " + content, topK=5, withWeight=False
                    )
                    keywords_str = ",".join(keywords)

                    # Call LLM API for sentiment analysis
                    sentiment = analyze_sentiment_with_llm(title + " " + content)

                    news_data = {
                        "title": title,
                        "content": content[:500],  # Only save partial content
                        "source": "sina",
                        "date": news_date,
                        "sentiment": sentiment,
                        "keywords": keywords_str,
                    }

                    all_news.append(news_data)

                    # Save to database
                    conn = create_connection()
                    cursor = conn.cursor()

                    # Check if already exists
                    cursor.execute(
                        "SELECT id FROM news_data WHERE title = %s", (title,)
                    )
                    if cursor.fetchone() is None:
                        sql = """
                        INSERT INTO news_data (title, content, source, date, sentiment, keywords)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """
                        try:
                            cursor.execute(
                                sql,
                                (
                                    title,
                                    content[:1000],
                                    "sina",
                                    news_date,
                                    sentiment,
                                    keywords_str,
                                ),
                            )
                            conn.commit()
                        except Exception as e:
                            st.warning(f"Failed to save news data to database: {e}")

                    cursor.close()
                    conn.close()

                except Exception as e:
                    st.warning(f"Error processing single news item: {e}")
                    continue

        return pd.DataFrame(all_news)
    except Exception as e:
        st.error(f"Failed to get news data: {e}")
        return pd.DataFrame()


# Data processing functions
def preprocess_stock_data(df):
    """
    Preprocess stock data, calculate technical indicators
    """
    # Check if data exists
    if df is None or df.empty:
        return None

    # Copy data to avoid modifying original
    df_processed = df.copy()

    # Calculate moving averages
    df_processed["MA5"] = df_processed["close"].rolling(window=5).mean()
    df_processed["MA10"] = df_processed["close"].rolling(window=10).mean()
    df_processed["MA20"] = df_processed["close"].rolling(window=20).mean()

    # Calculate returns
    df_processed["Daily_Return"] = df_processed["close"].pct_change()

    # Calculate volatility (standard deviation)
    df_processed["Volatility_5d"] = df_processed["Daily_Return"].rolling(window=5).std()

    # Calculate MACD
    close = df_processed["close"].values
    if len(close) > 26:
        df_processed["EMA12"] = talib.EMA(close, timeperiod=12)
        df_processed["EMA26"] = talib.EMA(close, timeperiod=26)
        df_processed["MACD"] = df_processed["EMA12"] - df_processed["EMA26"]
        df_processed["Signal"] = talib.EMA(df_processed["MACD"].values, timeperiod=9)
        df_processed["Histogram"] = df_processed["MACD"] - df_processed["Signal"]

    # Calculate RSI
    if len(close) > 14:
        df_processed["RSI"] = talib.RSI(close, timeperiod=14)

    # Handle missing values
    df_processed.fillna(method="bfill", inplace=True)

    # Add trend labels (for classification problems)
    df_processed["Target_Classification"] = 0
    df_processed.loc[
        df_processed["close"].shift(-1) > df_processed["close"], "Target_Classification"
    ] = 1

    # Add future n-day price changes (for regression problems)
    for days in [1, 3, 5]:
        df_processed[f"Target_Regression_{days}d"] = (
            df_processed["close"].shift(-days) / df_processed["close"] - 1
        )

    # Handle any remaining missing values
    df_processed.dropna(inplace=True)

    return df_processed


# Feature engineering functions
def create_features(df, target_col, prediction_days=5):
    """
    Create features and target variables
    """
    # Feature columns
    feature_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "MA5",
        "MA10",
        "MA20",
        "Daily_Return",
        "Volatility_5d",
    ]

    if "RSI" in df.columns:
        feature_columns.extend(["RSI"])

    if "MACD" in df.columns:
        feature_columns.extend(["MACD", "Signal", "Histogram"])

    # Extract features and targets
    X = df[feature_columns].values
    y = df[target_col].values

    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler, feature_columns


# Machine learning model training functions
def train_model(X_train, y_train, model_name):
    """
    Train machine learning model
    """
    if model_name == "linear_regression":
        model = LinearRegression()
    elif model_name == "decision_tree":
        model = DecisionTreeRegressor(max_depth=5, random_state=42)
    elif model_name == "random_forest":
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    elif model_name == "svm":
        model = SVR(kernel="rbf")
    elif model_name == "decision_tree_classifier":
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
    elif model_name == "random_forest_classifier":
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    elif model_name == "svm_classifier":
        model = SVC(kernel="rbf", probability=True)
    else:
        raise ValueError("Unsupported model type")

    model.fit(X_train, y_train)
    return model


# K-Means clustering functions
def cluster_stocks(df, n_clusters=3):
    """
    Use K-Means clustering algorithm to cluster stock data
    """
    # Select features for clustering
    features = ["Daily_Return", "Volatility_5d"]
    if "RSI" in df.columns:
        features.append("RSI")

    # Extract features
    X = df[features].dropna().values

    # Normalize
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    # Add cluster labels to dataframe
    cluster_df = df.copy()
    cluster_df = cluster_df.dropna(subset=features)
    cluster_df["Cluster"] = clusters

    return cluster_df, kmeans.cluster_centers_


# Visualization functions
def plot_stock_price(df):
    """Plot stock price trend chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index, df["close"], label="Close Price")
    ax.plot(df.index, df["MA5"], label="5-day MA")
    ax.plot(df.index, df["MA20"], label="20-day MA")
    ax.set_title("Stock Price Trend Chart")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    return fig


def plot_candlestick(df):
    """Plot candlestick chart"""
    df_plot = df.copy()
    df_plot.index.name = "Date"

    # Rename columns to match mplfinance requirements
    df_plot = df_plot.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )

    # Set Chinese font
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun"]
    plt.rcParams["axes.unicode_minus"] = False

    # Create custom style
    mc = mpf.make_marketcolors(up="red", down="green", inherit=True)
    s = mpf.make_mpf_style(
        base_mpf_style="charles",
        marketcolors=mc,
        gridcolor="lightgray",
        figcolor="white",
        y_on_right=False,
    )

    # Use mplfinance to plot candlestick chart
    fig, axlist = mpf.plot(
        df_plot,
        type="candle",
        volume=True,
        # Remove title parameter
        ylabel="Price",
        ylabel_lower="Volume",
        style=s,
        returnfig=True,
        figsize=(10, 8),
    )

    # Get chart axes, only set label font, not title
    if len(axlist) > 0:
        ax_main = axlist[0]
        ax_main.set_ylabel("Price", fontproperties="SimHei")

        if len(axlist) > 2:  # Has volume subplot
            ax_volume = axlist[2]
            ax_volume.set_ylabel("Volume", fontproperties="SimHei")

    return fig


def plot_technical_indicators(df):
    """Plot technical indicators"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # MACD chart
    if "MACD" in df.columns:
        ax1.plot(df.index, df["MACD"], label="MACD")
        ax1.plot(df.index, df["Signal"], label="Signal")
        ax1.bar(df.index, df["Histogram"], label="Histogram", alpha=0.5)
        ax1.set_title("MACD Indicator")
        ax1.legend()
        ax1.grid(True)

    # RSI chart
    if "RSI" in df.columns:
        ax2.plot(df.index, df["RSI"], color="purple", label="RSI")
        ax2.axhline(y=70, color="r", linestyle="-", alpha=0.3)
        ax2.axhline(y=30, color="g", linestyle="-", alpha=0.3)
        ax2.set_title("RSI Indicator")
        ax2.set_ylabel("RSI")
        ax2.set_xlabel("Date")
        ax2.legend()
        ax2.grid(True)

    plt.tight_layout()
    return fig


def plot_prediction_results(y_test, y_pred, model_name):
    """Plot prediction results comparison chart"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(y_test, label="Actual")
    ax.plot(y_pred, label="Predicted")
    ax.set_title(f"{model_name} Prediction Results")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)

    return fig


def plot_correlation_heatmap(df):
    """Plot correlation heatmap"""
    # Select numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    # Calculate correlation
    corr = numeric_df.corr()

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
    ax.set_title("Feature Correlation Heatmap")

    return fig


def plot_clusters(df, feature_x, feature_y):
    """Plot clustering results"""
    if "Cluster" not in df.columns:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot scatter plot for each cluster
    clusters = df["Cluster"].unique()
    for cluster in clusters:
        cluster_data = df[df["Cluster"] == cluster]
        ax.scatter(
            cluster_data[feature_x],
            cluster_data[feature_y],
            label=f"Cluster {cluster}",
            alpha=0.7,
        )

    ax.set_title("Stock Clustering Results")
    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.legend()
    ax.grid(True)

    return fig


def plot_news_sentiment(news_df):
    """Plot news sentiment analysis"""
    if news_df.empty or "sentiment" not in news_df.columns:
        return None

    # Set Chinese font
    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "SimSun",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    # Group by date and calculate average sentiment score
    if "date" in news_df.columns:
        sentiment_by_date = news_df.groupby("date")["sentiment"].mean().reset_index()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(
            sentiment_by_date["date"], sentiment_by_date["sentiment"], color="skyblue"
        )
        ax.set_title("Daily News Sentiment Score", fontproperties="SimHei", fontsize=14)
        ax.set_xlabel("Date", fontproperties="SimHei", fontsize=12)
        ax.set_ylabel(
            "Sentiment Score (Negative→Positive)", fontproperties="SimHei", fontsize=12
        )
        ax.grid(True, axis="y")
        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig
    return None


def generate_news_wordcloud(news_df):
    """Generate news keyword word cloud"""
    if news_df.empty:
        return None

    # Extract all keywords
    all_keywords = []
    if "keywords" in news_df.columns:
        for keywords in news_df["keywords"]:
            if isinstance(keywords, str):
                all_keywords.extend(keywords.split(","))

    # If no keywords, use titles
    if not all_keywords and "title" in news_df.columns:
        text = " ".join(news_df["title"].dropna().tolist())
        words = jieba.cut(text)
        all_keywords = [w for w in words if len(w) > 1]

    if not all_keywords:
        return None

    text = " ".join(all_keywords)

    # Set Chinese font
    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "SimSun",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    # Try multiple possible Chinese font paths
    font_paths = [
        "simhei.ttf",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",  # Microsoft YaHei
        "C:/Windows/Fonts/simsun.ttc",  # SimSun
        "/System/Library/Fonts/PingFang.ttc",  # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
    ]

    font_path = None
    for path in font_paths:
        if os.path.exists(path):
            font_path = path
            break

    # Generate word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        font_path=font_path,
        max_words=100,
        collocations=False,
        relative_scaling=0.5,
    ).generate(text)

    # Display word cloud
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("News Keyword Word Cloud", fontproperties="SimHei", fontsize=14)
    plt.tight_layout()

    return fig


def plot_news_sentiment_distribution(news_df):
    """Plot news sentiment distribution histogram"""
    if news_df.empty or "sentiment" not in news_df.columns:
        return None

    # Set Chinese font
    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "SimSun",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        news_df["sentiment"], bins=20, color="skyblue", alpha=0.7, edgecolor="black"
    )
    ax.set_xlabel("Sentiment Score", fontproperties="SimHei", fontsize=12)
    ax.set_ylabel("Frequency", fontproperties="SimHei", fontsize=12)
    ax.set_title(
        "News Sentiment Distribution Histogram", fontproperties="SimHei", fontsize=14
    )
    ax.grid(True, alpha=0.3)

    # Add statistical information
    mean_sentiment = news_df["sentiment"].mean()
    ax.axvline(
        mean_sentiment,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_sentiment:.3f}",
    )
    ax.legend(prop={"family": "SimHei"})

    plt.tight_layout()
    return fig


# Save results functions
def save_prediction_to_db(
    symbol,
    model_name,
    prediction_date,
    predicted_value,
    actual_value,
    accuracy,
    model_params,
):
    """Save prediction results to database"""
    try:
        conn = create_connection()
        cursor = conn.cursor()

        sql = """
        INSERT INTO prediction_results 
        (symbol, model_name, prediction_date, predicted_value, actual_value, accuracy, model_params)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """

        cursor.execute(
            sql,
            (
                symbol,
                model_name,
                prediction_date,
                float(predicted_value),
                float(actual_value) if actual_value is not None else None,
                float(accuracy),
                json.dumps(model_params),
            ),
        )

        conn.commit()
        cursor.close()
        conn.close()

        return True
    except Exception as e:
        st.error(f"Failed to save prediction results to database: {e}")
        return False


def get_table_download_link(df, filename, text):
    """Generate data download link"""
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href


def export_to_excel(df, filename):
    """Export data to Excel"""
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine="xlsxwriter")
    df.to_excel(writer, sheet_name="Sheet1")
    writer.close()
    output.seek(0)

    b64 = base64.b64encode(output.read()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">{filename}</a>'
    return href


# Data acquisition tab
def data_acquisition_tab():
    st.header("Data Acquisition")

    # Create two columns
    col1, col2 = st.columns(2)

    # Stock data acquisition
    with col1:
        st.subheader("Stock Data Acquisition")

        # Add stock code options
        stock_input_type = st.radio(
            "Select input method", ["Direct input", "Select from list"], horizontal=True
        )

        if stock_input_type == "Direct input":
            symbol = st.text_input("Stock code", "600000")
            st.caption("A-share code format: 600000(Shanghai) or 000001(Shenzhen)")
        else:
            try:
                # Get A-share code list
                with st.spinner("Getting stock list..."):
                    stock_list = get_stock_list()

                if stock_list:
                    # Convert to selectable format
                    stock_options = [
                        f"{stock['code'].split('.')[-1]} - {stock['name']}"
                        for stock in stock_list[:1000]
                    ]  # Limit quantity to avoid too many
                    selected_stock = st.selectbox("Select stock", stock_options)
                    symbol = selected_stock.split(" - ")[0]  # Extract stock code
                else:
                    st.error("Failed to get stock list")
                    symbol = st.text_input("Stock code", "600000")
            except Exception as e:
                st.error(f"Failed to get stock list: {e}")
                symbol = st.text_input("Stock code", "600000")

        start_date = st.date_input("Start date", datetime.now() - timedelta(days=365))
        end_date = st.date_input("End date", datetime.now())

        if st.button("Get stock data"):
            with st.spinner("Getting stock data..."):
                # Save stock code to session state
                st.session_state.stock_symbol = symbol

                stock_df = fetch_stock_data(
                    symbol,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                )
                if stock_df is not None and not stock_df.empty:
                    st.session_state.stock_df = stock_df
                    st.success(
                        f"Successfully obtained {symbol} stock data, {len(stock_df)} records"
                    )
                    st.dataframe(stock_df.head())
                else:
                    st.error("Failed to get stock data")

        # Add get stock basic info button
        if "stock_symbol" in st.session_state:
            if st.button("Get stock basic info", key="get_stock_info_btn"):
                with st.spinner("Getting stock basic info..."):
                    try:
                        symbol = st.session_state.stock_symbol
                        stock_info = get_stock_basic_info(symbol)

                        if stock_info:
                            st.write("Stock basic info:")
                            st.write(f"Stock code: {stock_info[0]}")
                            st.write(f"Stock name: {stock_info[1]}")
                            st.write(f"Listing date: {stock_info[2]}")
                            st.write(
                                f"Delisting date: {stock_info[3] if stock_info[3] else 'Present'}"
                            )
                            st.write(
                                f"Stock type: {'A-share' if stock_info[4]=='1' else 'Other'}"
                            )
                            st.write(
                                f"Status: {'Listed' if stock_info[5]=='1' else 'Delisted'}"
                            )
                        else:
                            st.warning(f"Stock {symbol} basic info not found")
                    except Exception as e:
                        st.error(f"Failed to get stock basic info: {e}")

    # Index data acquisition and news data
    with col2:
        st.subheader("Index Data Acquisition")

        # Major indices list
        major_indices = {
            "sh.000001": "Shanghai Composite Index",
            "sz.399001": "Shenzhen Component Index",
            "sz.399006": "ChiNext Index",
            "sh.000016": "SSE 50",
            "sh.000300": "CSI 300",
            "sz.399905": "CSI 500",
        }

        selected_index = st.selectbox(
            "Select index", list(major_indices.items()), format_func=lambda x: x[1]
        )

        index_start_date = st.date_input(
            "Index start date", datetime.now() - timedelta(days=365), key="index_start"
        )
        index_end_date = st.date_input(
            "Index end date", datetime.now(), key="index_end"
        )

        if st.button("Get index data"):
            with st.spinner("Getting index data..."):
                index_code = selected_index[0]
                index_df = fetch_index_data(
                    index_code,
                    index_start_date.strftime("%Y-%m-%d"),
                    index_end_date.strftime("%Y-%m-%d"),
                )

                if index_df is not None and not index_df.empty:
                    st.session_state.index_df = index_df
                    st.session_state.index_name = selected_index[1]
                    st.success(
                        f"Successfully obtained {selected_index[1]} index data, {len(index_df)} records"
                    )
                    st.dataframe(index_df.head())
                else:
                    st.error("Failed to get index data")

        # News data crawling
        st.subheader("News Data Crawling")

        news_keyword = st.text_input("Search keyword", "Alibaba")
        news_pages = st.slider("Crawl pages", 1, 5, 2)

        if st.button("Get news data"):
            with st.spinner("Getting news data..."):
                news_df = fetch_news_data(news_keyword, news_pages)
                if not news_df.empty:
                    st.session_state.news_df = news_df
                    st.success(f"Successfully obtained {len(news_df)} news records")
                    st.dataframe(news_df[["title", "date", "sentiment"]].head())
                else:
                    st.error("Failed to get news data")

    # Industry classification data
    st.subheader("Industry Classification Data")

    if st.button("Get industry classification data"):
        with st.spinner("Getting industry classification data..."):
            industry_df = get_industry_data()

            if industry_df is not None and not industry_df.empty:
                st.session_state.industry_df = industry_df
                st.success(
                    f"Successfully obtained industry classification data, {len(industry_df)} records"
                )

                # Display industry statistics
                industry_counts = industry_df["industry"].value_counts().reset_index()
                industry_counts.columns = ["Industry", "Number of listed companies"]

                # Display industry distribution chart
                st.write("Industry distribution:")
                # Set Chinese font
                plt.rcParams["font.sans-serif"] = [
                    "SimHei",
                    "Microsoft YaHei",
                    "SimSun",
                    "Arial Unicode MS",
                    "DejaVu Sans",
                ]
                plt.rcParams["axes.unicode_minus"] = False

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(
                    industry_counts["Industry"][:15],
                    industry_counts["Number of listed companies"][:15],
                    color="steelblue",
                    alpha=0.7,
                )
                ax.set_xlabel("Industry", fontproperties="SimHei", fontsize=12)
                ax.set_ylabel(
                    "Number of listed companies", fontproperties="SimHei", fontsize=12
                )
                ax.set_title(
                    "Number of listed companies by industry (Top 15)",
                    fontproperties="SimHei",
                    fontsize=14,
                )
                plt.xticks(rotation=45, ha="right")
                ax.grid(True, alpha=0.3, axis="y")
                plt.tight_layout()
                st.pyplot(fig)

                # Display industry classification data table
                st.write("Industry classification data:")
                st.dataframe(industry_df.head(20))
            else:
                st.error("Failed to get industry classification data")

    # Trading calendar data
    st.subheader("Trading Calendar Data")

    calendar_col1, calendar_col2 = st.columns(2)
    with calendar_col1:
        calendar_start = st.date_input(
            "Calendar start date",
            datetime.now() - timedelta(days=30),
            key="calendar_start",
        )
    with calendar_col2:
        calendar_end = st.date_input(
            "Calendar end date", datetime.now(), key="calendar_end"
        )

    if st.button("Get trading calendar"):
        with st.spinner("Getting trading calendar..."):
            calendar_df = get_trade_calendar(
                calendar_start.strftime("%Y-%m-%d"), calendar_end.strftime("%Y-%m-%d")
            )

            if calendar_df is not None and not calendar_df.empty:
                st.session_state.calendar_df = calendar_df
                st.success(
                    f"Successfully obtained trading calendar, {len(calendar_df)} records"
                )

                # Display trading and non-trading days count
                trading_days = calendar_df[calendar_df["is_trading_day"]].shape[0]
                non_trading_days = calendar_df[~calendar_df["is_trading_day"]].shape[0]

                st.write(f"Trading days: {trading_days}")
                st.write(f"Non-trading days: {non_trading_days}")

                # Display trading calendar table
                st.write("Trading calendar:")
                st.dataframe(calendar_df)
            else:
                st.error("Failed to get trading calendar")

    # File upload
    st.subheader("Upload CSV/Excel File")
    uploaded_file = st.file_uploader("Select file", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success(f"Successfully uploaded file, {len(df)} records")
            st.dataframe(df.head())
            st.session_state.uploaded_df = df

            # Check if it's stock data
            if all(col in df.columns for col in ["open", "high", "low", "close"]):
                st.info(
                    "Detected uploaded stock data, can be processed in data analysis tab"
                )
        except Exception as e:
            st.error(f"Failed to read file: {e}")


# Data analysis tab
def data_analysis_tab():
    st.header("Data Analysis")

    # Check if stock data exists
    if "stock_df" in st.session_state:
        st.subheader("Stock Data Preprocessing")

        if st.button("Data Preprocessing"):
            with st.spinner("Processing data..."):
                processed_df = preprocess_stock_data(st.session_state.stock_df)
                if processed_df is not None:
                    st.session_state.processed_df = processed_df
                    st.success("Data preprocessing completed")
                    st.write("Processed data sample:")
                    st.dataframe(processed_df.head())

                    # Display basic statistics
                    st.subheader("Basic Statistics")
                    st.write(processed_df.describe())

                    # Display technical indicators
                    st.subheader("Technical Indicators")
                    tech_indicators = ["close", "MA5", "MA20"]
                    if "RSI" in processed_df.columns:
                        tech_indicators.append("RSI")
                    st.line_chart(processed_df[tech_indicators])

        if "processed_df" in st.session_state:
            # Correlation analysis
            st.subheader("Correlation Analysis")
            if st.button("Generate Correlation Matrix"):
                # Select numeric columns
                numeric_df = st.session_state.processed_df.select_dtypes(
                    include=[np.number]
                )
                corr = numeric_df.corr()

                # Display correlation matrix
                st.write("Feature Correlation Matrix:")
                st.dataframe(corr.style.background_gradient(cmap="coolwarm"))

                # Correlation heatmap
                fig = plot_correlation_heatmap(st.session_state.processed_df)
                st.pyplot(fig)

            # Add data visualization options
            st.subheader("Data Visualization")

            viz_type = st.selectbox(
                "Select Visualization Type",
                ["Candlestick Chart", "Technical Indicators", "Return Distribution"],
            )

            if viz_type == "Candlestick Chart":
                if st.button("Generate Candlestick Chart"):
                    fig = plot_candlestick(st.session_state.processed_df)
                    st.pyplot(fig)

            elif viz_type == "Technical Indicators":
                if st.button("Generate Technical Indicators Chart"):
                    fig = plot_technical_indicators(st.session_state.processed_df)
                    st.pyplot(fig)

            elif viz_type == "Return Distribution":
                if st.button("Generate Return Distribution Chart"):
                    # Set Chinese font
                    plt.rcParams["font.sans-serif"] = [
                        "SimHei",
                        "Microsoft YaHei",
                        "SimSun",
                        "Arial Unicode MS",
                        "DejaVu Sans",
                    ]
                    plt.rcParams["axes.unicode_minus"] = False

                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(
                        st.session_state.processed_df["Daily_Return"].dropna(),
                        bins=50,
                        color="lightgreen",
                        alpha=0.7,
                        edgecolor="black",
                    )
                    ax.set_title(
                        "Daily Return Distribution",
                        fontproperties="SimHei",
                        fontsize=14,
                    )
                    ax.set_xlabel("Daily Return", fontproperties="SimHei", fontsize=12)
                    ax.set_ylabel("Frequency", fontproperties="SimHei", fontsize=12)
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
    elif "uploaded_df" in st.session_state:
        if all(
            col in st.session_state.uploaded_df.columns
            for col in ["open", "high", "low", "close"]
        ):
            st.info("Using uploaded stock data")
            if st.button("Process Uploaded Data"):
                with st.spinner("Processing data..."):
                    processed_df = preprocess_stock_data(st.session_state.uploaded_df)
                    if processed_df is not None:
                        st.session_state.processed_df = processed_df
                        st.success("Data preprocessing completed")
                        st.write("Processed data sample:")
                        st.dataframe(processed_df.head())
        else:
            st.info(
                "Uploaded file is not in standard stock data format, please ensure it contains open, high, low, close columns"
            )
    else:
        st.info("Please first get stock data in the data acquisition tab")

    # Index data analysis
    if "index_df" in st.session_state:
        st.subheader("Index Data Analysis")

        if st.button("Analyze Index Data"):
            with st.spinner("Analyzing index data..."):
                index_df = st.session_state.index_df
                index_name = st.session_state.index_name

                # Calculate index moving averages
                index_df["MA5"] = index_df["close"].rolling(window=5).mean()
                index_df["MA20"] = index_df["close"].rolling(window=20).mean()
                index_df["MA60"] = index_df["close"].rolling(window=60).mean()

                # Calculate index returns
                index_df["Daily_Return"] = index_df["close"].pct_change()

                # Display index basic statistics
                st.write(f"{index_name} Basic Statistics:")
                st.dataframe(index_df.describe())

                # Plot index trend chart
                # Set Chinese font
                plt.rcParams["font.sans-serif"] = [
                    "SimHei",
                    "Microsoft YaHei",
                    "SimSun",
                    "Arial Unicode MS",
                    "DejaVu Sans",
                ]
                plt.rcParams["axes.unicode_minus"] = False

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(index_df.index, index_df["close"], label="Close Price")
                ax.plot(index_df.index, index_df["MA5"], label="5-day MA")
                ax.plot(index_df.index, index_df["MA20"], label="20-day MA")
                ax.plot(index_df.index, index_df["MA60"], label="60-day MA")
                ax.set_title(
                    f"{index_name} Trend Chart", fontproperties="SimHei", fontsize=14
                )
                ax.set_xlabel("Date", fontproperties="SimHei", fontsize=12)
                ax.set_ylabel("Index Points", fontproperties="SimHei", fontsize=12)
                ax.legend(prop={"family": "SimHei"})
                ax.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

                # Plot index return distribution chart
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(
                    index_df["Daily_Return"].dropna(),
                    bins=50,
                    color="lightblue",
                    alpha=0.7,
                    edgecolor="black",
                )
                ax.set_title(
                    f"{index_name} Daily Return Distribution",
                    fontproperties="SimHei",
                    fontsize=14,
                )
                ax.set_xlabel("Daily Return", fontproperties="SimHei", fontsize=12)
                ax.set_ylabel("Frequency", fontproperties="SimHei", fontsize=12)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)

                # If stock data exists, calculate correlation with index
                if "processed_df" in st.session_state:
                    st.subheader("Stock and Index Correlation Analysis")

                    stock_df = st.session_state.processed_df

                    # Ensure index is datetime type
                    if not isinstance(stock_df.index, pd.DatetimeIndex):
                        stock_df.index = pd.to_datetime(stock_df.index)

                    # Calculate common date range
                    common_dates = sorted(set(stock_df.index) & set(index_df.index))

                    if common_dates:
                        # Extract data for common date range
                        stock_returns = stock_df.loc[common_dates, "Daily_Return"]
                        index_returns = index_df.loc[common_dates, "Daily_Return"]

                        # Calculate correlation coefficient
                        correlation = stock_returns.corr(index_returns)

                        st.write(
                            f"Stock and {index_name} return correlation coefficient: {correlation:.4f}"
                        )

                        # Plot scatter chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.scatter(index_returns, stock_returns, alpha=0.5)
                        ax.set_title(
                            f"Stock Return vs {index_name} Return (Correlation: {correlation:.4f})",
                            fontproperties="SimHei",
                            fontsize=14,
                        )
                        ax.set_xlabel(
                            f"{index_name} Daily Return",
                            fontproperties="SimHei",
                            fontsize=12,
                        )
                        ax.set_ylabel(
                            "Stock Daily Return", fontproperties="SimHei", fontsize=12
                        )
                        ax.grid(True, alpha=0.3)

                        # Add regression line
                        z = np.polyfit(index_returns, stock_returns, 1)
                        p = np.poly1d(z)
                        ax.plot(index_returns, p(index_returns), "r--", linewidth=2)

                        plt.tight_layout()
                        st.pyplot(fig)

                        # Calculate Beta coefficient
                        beta = correlation * (stock_returns.std() / index_returns.std())
                        st.write(f"Beta Coefficient (Market Sensitivity): {beta:.4f}")
                        st.write(
                            f"Beta Explanation: {'Higher than market volatility' if beta > 1 else 'Lower than market volatility'}"
                        )

    # News data analysis
    if "news_df" in st.session_state:
        st.subheader("News Data Analysis")

        # Aggregate news by date
        if "date" in st.session_state.news_df.columns:
            news_count_by_date = (
                st.session_state.news_df.groupby("date")
                .size()
                .reset_index(name="count")
            )
            st.write("Daily news count:")
            st.bar_chart(news_count_by_date.set_index("date"))

        # Sentiment analysis
        if "sentiment" in st.session_state.news_df.columns:
            st.write("News sentiment distribution:")
            fig = plot_news_sentiment(st.session_state.news_df)
            if fig:
                st.pyplot(fig)
            else:
                st.info(
                    "Unable to generate sentiment analysis chart, please ensure news data contains sentiment scores and dates"
                )

            # Average sentiment score
            avg_sentiment = st.session_state.news_df["sentiment"].mean()
            st.metric(
                "Average sentiment score",
                f"{avg_sentiment:.3f}",
                delta="Positive" if avg_sentiment > 0 else "Negative",
            )

        # Generate word cloud
        st.write("News keyword word cloud:")
        fig = generate_news_wordcloud(st.session_state.news_df)
        if fig:
            st.pyplot(fig)

        # Use LLM to analyze news insights
        st.subheader("News Insights Analysis")
        if st.button("Analyze news trends", key="analyze_news_trend_btn"):
            with st.spinner("Using AI to analyze news trends..."):
                insights = extract_news_insights_with_llm(st.session_state.news_df)
                st.info(insights)

                # Save news insights to session state
                st.session_state.news_insights = insights

        # Plot news sentiment distribution histogram
        st.subheader("News Sentiment Distribution Histogram")
        fig = plot_news_sentiment_distribution(st.session_state.news_df)
        if fig:
            st.pyplot(fig)
    else:
        st.info("Please first get news data in the data acquisition tab")

    # Industry data analysis
    if "industry_df" in st.session_state:
        st.subheader("Industry Data Analysis")

        industry_df = st.session_state.industry_df

        # Display industry statistics
        industry_counts = industry_df["industry"].value_counts()

        # Select specific industry for analysis
        selected_industry = st.selectbox(
            "Select industry to analyze",
            options=sorted(industry_df["industry"].unique()),
        )

        if st.button("Analyze industry stocks"):
            with st.spinner("Analyzing industry stocks..."):
                # Filter stocks in selected industry
                industry_stocks = industry_df[
                    industry_df["industry"] == selected_industry
                ]

                st.write(
                    f"{selected_industry} industry listed companies: {len(industry_stocks)} companies"
                )
                st.dataframe(industry_stocks)

                # Stock code list
                stock_codes = industry_stocks["code"].tolist()

                # If stock and index data exist, add industry vs market comparison analysis option
                if (
                    "processed_df" in st.session_state
                    and "index_df" in st.session_state
                ):
                    st.subheader("Industry vs Individual Stock and Market Comparison")

                    stock_symbol = st.session_state.stock_symbol
                    stock_data = st.session_state.processed_df
                    index_data = st.session_state.index_df

                    # Prompt user
                    st.write(
                        f"Does the stock you are currently analyzing ({stock_symbol}) belong to the {selected_industry} industry:"
                    )

                    # Check if current stock belongs to selected industry
                    is_in_industry = False
                    for code in stock_codes:
                        if code.endswith(stock_symbol):
                            is_in_industry = True
                            break

                    st.write("Yes" if is_in_industry else "No")

                    # Industry index trend (example only, actual implementation requires more industry stock data)
                    st.write(
                        "For complete industry index data, please use professional data sources"
                    )


# Machine learning prediction tab
def ml_prediction_tab():
    st.header("Machine Learning Prediction")

    if "processed_df" in st.session_state:
        # Create two columns
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Regression Prediction")

            # Select regression target
            regression_target = st.selectbox(
                "Select regression prediction target",
                [
                    "Target_Regression_1d",
                    "Target_Regression_3d",
                    "Target_Regression_5d",
                ],
            )

            # Select regression model
            regression_model = st.selectbox(
                "Select regression model",
                ["linear_regression", "decision_tree", "random_forest", "svm"],
            )

            # Train regression model button
            if st.button("Train regression model"):
                with st.spinner("Training model..."):
                    # Create features and targets
                    X_train, X_test, y_train, y_test, scaler, feature_cols = (
                        create_features(
                            st.session_state.processed_df, regression_target
                        )
                    )

                    # Train model
                    model = train_model(X_train, y_train, regression_model)

                    # Predict
                    y_pred = model.predict(X_test)

                    # Calculate MSE
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)

                    # Save results
                    st.session_state.regression_results = {
                        "y_test": y_test,
                        "y_pred": y_pred,
                        "mse": mse,
                        "rmse": rmse,
                        "model": model,
                        "model_name": regression_model,
                        "target": regression_target,
                    }

                    # Display results
                    st.success(
                        f"Model training completed, Mean Squared Error (MSE): {mse:.6f}, Root Mean Squared Error (RMSE): {rmse:.6f}"
                    )

                    # Visualize prediction results
                    fig = plot_prediction_results(y_test, y_pred, regression_model)
                    st.pyplot(fig)

                    # Use LLM to interpret regression prediction results
                    st.subheader("AI Prediction Result Interpretation")
                    with st.spinner("Generating AI analysis report..."):
                        symbol = st.session_state.get("stock_symbol", "UNKNOWN")
                        interpretation = interpret_prediction_with_llm(
                            symbol, y_pred, regression_model, f"RMSE: {rmse:.6f}"
                        )
                        st.info(interpretation)

                    # Save prediction results to database
                    symbol = st.session_state.get("stock_symbol", "UNKNOWN")
                    save_prediction_to_db(
                        symbol=symbol,
                        model_name=regression_model,
                        prediction_date=datetime.now().strftime("%Y-%m-%d"),
                        predicted_value=float(y_pred[-1]),
                        actual_value=float(y_test[-1]) if len(y_test) > 0 else None,
                        accuracy=float(rmse),
                        model_params={"features": feature_cols},
                    )

        with col2:
            st.subheader("Classification Prediction")

            # Select classification model
            classification_model = st.selectbox(
                "Select classification model",
                [
                    "decision_tree_classifier",
                    "random_forest_classifier",
                    "svm_classifier",
                ],
            )

            # Train classification model button
            if st.button("Train classification model"):
                with st.spinner("Training model..."):
                    # Create features and targets
                    X_train, X_test, y_train, y_test, scaler, feature_cols = (
                        create_features(
                            st.session_state.processed_df, "Target_Classification"
                        )
                    )

                    # Train model
                    model = train_model(X_train, y_train, classification_model)

                    # Predict
                    y_pred = model.predict(X_test)

                    # Calculate accuracy
                    accuracy = accuracy_score(y_test, y_pred)

                    # Save results
                    st.session_state.classification_results = {
                        "y_test": y_test,
                        "y_pred": y_pred,
                        "accuracy": accuracy,
                        "model": model,
                        "model_name": classification_model,
                    }

                    # Display results
                    st.success(f"Model training completed, accuracy: {accuracy:.4f}")

                    # Display classification report
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.write("Classification report:")
                    st.dataframe(report_df)

                    # Visualize classification results
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(y_test, label="Actual values", marker="o", linestyle="--")
                    ax.plot(y_pred, label="Predicted values", marker="x")
                    ax.set_title(f"{classification_model} Classification Results")
                    ax.set_xlabel("Samples")
                    ax.set_ylabel("Class (0: Down, 1: Up)")
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)

                    # Use LLM to interpret prediction results
                    st.subheader("AI Prediction Result Interpretation")
                    with st.spinner("Generating AI analysis report..."):
                        symbol = st.session_state.get("stock_symbol", "UNKNOWN")
                        interpretation = interpret_prediction_with_llm(
                            symbol,
                            y_pred,
                            classification_model,
                            f"Accuracy: {accuracy:.4f}",
                        )
                        st.info(interpretation)

        # K-Means clustering
        st.subheader("K-Means Clustering Analysis")

        n_clusters = st.slider("Select number of clusters", 2, 5, 3)

        if st.button("Execute clustering analysis"):
            with st.spinner("Executing clustering analysis..."):
                # Execute clustering
                cluster_df, cluster_centers = cluster_stocks(
                    st.session_state.processed_df, n_clusters
                )

                st.session_state.cluster_df = cluster_df
                st.session_state.cluster_centers = cluster_centers

                # Display clustering results
                st.success(
                    f"Clustering analysis completed, {n_clusters} clusters in total"
                )

                # Display sample count for each cluster
                cluster_counts = cluster_df["Cluster"].value_counts().sort_index()
                st.write("Sample count for each cluster:")
                st.bar_chart(cluster_counts)

                # Visualize clustering results
                fig = plot_clusters(cluster_df, "Daily_Return", "Volatility_5d")
                st.pyplot(fig)

                # Analyze features of each cluster
                st.write("Mean features for each cluster:")
                # Only select numeric columns for mean calculation, avoid mean operation on non-numeric columns
                numeric_cols = cluster_df.select_dtypes(
                    include=["number"]
                ).columns.tolist()
                numeric_cols = [col for col in numeric_cols if col != "Cluster"]
                if numeric_cols:
                    cluster_means = cluster_df.groupby("Cluster")[numeric_cols].mean()
                    st.dataframe(cluster_means)
    else:
        st.info("Please first process stock data in the data analysis tab")


# Visualization display tab
def visualization_tab():
    st.header("Visualization Display")

    # Stock data visualization
    if "processed_df" in st.session_state:
        st.subheader("Stock Data Visualization")

        viz_option = st.selectbox(
            "Select visualization type",
            [
                "Candlestick Chart",
                "Price Trend Chart",
                "Technical Indicators",
                "Correlation Heatmap",
            ],
        )

        if viz_option == "Candlestick Chart":
            fig = plot_candlestick(st.session_state.processed_df)
            st.pyplot(fig)

        elif viz_option == "Price Trend Chart":
            fig = plot_stock_price(st.session_state.processed_df)
            st.pyplot(fig)

        elif viz_option == "Technical Indicators":
            fig = plot_technical_indicators(st.session_state.processed_df)
            st.pyplot(fig)

        elif viz_option == "Correlation Heatmap":
            fig = plot_correlation_heatmap(st.session_state.processed_df)
            st.pyplot(fig)

    # Prediction results visualization
    if (
        "regression_results" in st.session_state
        or "classification_results" in st.session_state
    ):
        st.subheader("Prediction Results Visualization")

        pred_viz_option = st.radio(
            "Select prediction result type",
            ["Regression Prediction", "Classification Prediction"],
            horizontal=True,
        )

        if (
            pred_viz_option == "Regression Prediction"
            and "regression_results" in st.session_state
        ):
            results = st.session_state.regression_results
            fig = plot_prediction_results(
                results["y_test"], results["y_pred"], results["model_name"]
            )
            st.pyplot(fig)

            st.metric("Root Mean Squared Error (RMSE)", f"{results['rmse']:.6f}")

        elif (
            pred_viz_option == "Classification Prediction"
            and "classification_results" in st.session_state
        ):
            results = st.session_state.classification_results

            # Display confusion matrix
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(results["y_test"], results["y_pred"])

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues")
            ax.set_xlabel("Predicted Class")
            ax.set_ylabel("Actual Class")
            ax.set_title("Confusion Matrix")
            ax.set_xticklabels(["Down", "Up"])
            ax.set_yticklabels(["Down", "Up"])
            st.pyplot(fig)

            st.metric("Accuracy", f"{results['accuracy']:.4f}")

    # Clustering results visualization
    if "cluster_df" in st.session_state:
        st.subheader("Clustering Results Visualization")

        # Select features to display in scatter plot
        col1, col2 = st.columns(2)
        with col1:
            x_feature = st.selectbox(
                "X-axis feature", ["Daily_Return", "Volatility_5d", "RSI", "close"]
            )
        with col2:
            y_feature = st.selectbox(
                "Y-axis feature", ["Volatility_5d", "Daily_Return", "RSI", "close"]
            )

        fig = plot_clusters(st.session_state.cluster_df, x_feature, y_feature)
        st.pyplot(fig)

        # Feature distribution for each cluster
        # Only select numeric columns
        numeric_cols = st.session_state.cluster_df.select_dtypes(
            include=["number"]
        ).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != "Cluster"]

        if numeric_cols:
            # Create cluster statistics
            cluster_stats = (
                st.session_state.cluster_df.groupby("Cluster")[numeric_cols]
                .agg(["mean", "std"])
                .dropna(axis=1)
            )

            st.write("Cluster statistics:")
            st.dataframe(cluster_stats)

    # News data visualization
    if "news_df" in st.session_state:
        st.subheader("News Data Visualization")

        news_viz_option = st.radio(
            "Select news visualization type",
            [
                "Sentiment Analysis",
                "Sentiment Distribution Histogram",
                "Keyword Word Cloud",
            ],
            horizontal=True,
        )

        if news_viz_option == "Sentiment Analysis":
            fig = plot_news_sentiment(st.session_state.news_df)
            if fig:
                st.pyplot(fig)
            else:
                st.info(
                    "Unable to generate sentiment analysis chart, please ensure news data contains sentiment scores and dates"
                )

        elif news_viz_option == "Sentiment Distribution Histogram":
            fig = plot_news_sentiment_distribution(st.session_state.news_df)
            if fig:
                st.pyplot(fig)
            else:
                st.info(
                    "Unable to generate sentiment distribution chart, please ensure news data contains sentiment scores"
                )

        elif news_viz_option == "Keyword Word Cloud":
            fig = generate_news_wordcloud(st.session_state.news_df)
            if fig:
                st.pyplot(fig)
            else:
                st.info(
                    "Unable to generate word cloud, please ensure news data contains keywords or titles"
                )

    # Index visualization
    if "index_df" in st.session_state:
        st.subheader("Index Data Visualization")

        index_df = st.session_state.index_df
        index_name = st.session_state.index_name

        # Calculate moving averages
        if "MA5" not in index_df.columns:
            index_df["MA5"] = index_df["close"].rolling(window=5).mean()
            index_df["MA20"] = index_df["close"].rolling(window=20).mean()
            index_df["MA60"] = index_df["close"].rolling(window=60).mean()

        # Plot index trend chart
        # Set Chinese font
        plt.rcParams["font.sans-serif"] = [
            "SimHei",
            "Microsoft YaHei",
            "SimSun",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]
        plt.rcParams["axes.unicode_minus"] = False

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(index_df.index, index_df["close"], label="Close Price")
        ax.plot(index_df.index, index_df["MA5"], label="5-day MA")
        ax.plot(index_df.index, index_df["MA20"], label="20-day MA")
        ax.plot(index_df.index, index_df["MA60"], label="60-day MA")
        ax.set_title(f"{index_name} Trend Chart", fontproperties="SimHei", fontsize=14)
        ax.set_xlabel("Date", fontproperties="SimHei", fontsize=12)
        ax.set_ylabel("Index Points", fontproperties="SimHei", fontsize=12)
        ax.legend(prop={"family": "SimHei"})
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # Plot index candlestick chart
        # Rename columns to match mplfinance requirements
        index_plot_df = index_df.copy()
        index_plot_df = index_plot_df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
        index_plot_df.index.name = "Date"

        # Set Chinese font
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun"]
        plt.rcParams["axes.unicode_minus"] = False

        # Create custom style
        mc = mpf.make_marketcolors(up="red", down="green", inherit=True)
        s = mpf.make_mpf_style(
            base_mpf_style="charles",
            marketcolors=mc,
            gridcolor="lightgray",
            figcolor="white",
            y_on_right=False,
        )

        fig, axlist = mpf.plot(
            index_plot_df,
            type="candle",
            volume=True,
            # Remove title parameter
            ylabel="Index Points",
            ylabel_lower="Volume",
            style=s,
            returnfig=True,
            figsize=(10, 8),
        )

        # Get chart axes, only set label font, not title
        if len(axlist) > 0:
            ax_main = axlist[0]
            ax_main.set_ylabel("Index Points", fontproperties="SimHei")

            if len(axlist) > 2:  # Has volume subplot
                ax_volume = axlist[2]
                ax_volume.set_ylabel("Volume", fontproperties="SimHei")

        st.pyplot(fig)


# Export results tab
def export_tab():
    st.header("Export Results")

    # Select data to export
    export_option = st.selectbox(
        "Select data to export",
        [
            "Raw Stock Data",
            "Processed Stock Data",
            "Prediction Results",
            "News Data",
            "Index Data",
        ],
    )

    # Select export format
    export_format = st.radio("Select export format", ["CSV", "Excel"], horizontal=True)

    if st.button("Export data"):
        if export_option == "Raw Stock Data" and "stock_df" in st.session_state:
            if export_format == "CSV":
                st.markdown(
                    get_table_download_link(
                        st.session_state.stock_df, "stock_data.csv", "Download CSV file"
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    export_to_excel(st.session_state.stock_df, "stock_data.xlsx"),
                    unsafe_allow_html=True,
                )

        elif (
            export_option == "Processed Stock Data"
            and "processed_df" in st.session_state
        ):
            if export_format == "CSV":
                st.markdown(
                    get_table_download_link(
                        st.session_state.processed_df,
                        "processed_stock_data.csv",
                        "Download CSV file",
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    export_to_excel(
                        st.session_state.processed_df, "processed_stock_data.xlsx"
                    ),
                    unsafe_allow_html=True,
                )

        elif export_option == "Prediction Results":
            if "regression_results" in st.session_state:
                results = st.session_state.regression_results
                results_df = pd.DataFrame(
                    {"actual": results["y_test"], "predicted": results["y_pred"]}
                )

                if export_format == "CSV":
                    st.markdown(
                        get_table_download_link(
                            results_df, "prediction_results.csv", "Download CSV file"
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        export_to_excel(results_df, "prediction_results.xlsx"),
                        unsafe_allow_html=True,
                    )
            else:
                st.warning("No prediction results available")

        elif export_option == "News Data" and "news_df" in st.session_state:
            if export_format == "CSV":
                st.markdown(
                    get_table_download_link(
                        st.session_state.news_df, "news_data.csv", "Download CSV file"
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    export_to_excel(st.session_state.news_df, "news_data.xlsx"),
                    unsafe_allow_html=True,
                )

        elif export_option == "Index Data" and "index_df" in st.session_state:
            if export_format == "CSV":
                st.markdown(
                    get_table_download_link(
                        st.session_state.index_df, "index_data.csv", "Download CSV file"
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    export_to_excel(st.session_state.index_df, "index_data.xlsx"),
                    unsafe_allow_html=True,
                )

        else:
            st.warning("No data available")

    # Extract prediction results from database
    st.subheader("Get Historical Predictions from Database")

    if st.button("Query historical predictions"):
        try:
            conn = create_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
            SELECT symbol, model_name, prediction_date, predicted_value, actual_value, accuracy
            FROM prediction_results
            ORDER BY prediction_date DESC
            LIMIT 50
            """
            )

            result = cursor.fetchall()

            if result:
                predictions_df = pd.DataFrame(
                    result,
                    columns=[
                        "Stock Code",
                        "Model Name",
                        "Prediction Date",
                        "Predicted Value",
                        "Actual Value",
                        "Accuracy",
                    ],
                )

                st.write("Historical prediction results:")
                st.dataframe(predictions_df)

                if export_format == "CSV":
                    st.markdown(
                        get_table_download_link(
                            predictions_df,
                            "historical_predictions.csv",
                            "Download historical prediction data",
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        export_to_excel(predictions_df, "historical_predictions.xlsx"),
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No historical prediction data in database")

            cursor.close()
            conn.close()

        except Exception as e:
            st.error(f"Database query failed: {e}")

    # Database table query
    st.subheader("Database Table Query")

    table_name = st.selectbox(
        "Select table to query",
        [
            "stock_data",
            "news_data",
            "prediction_results",
            "index_data",
            "financial_data",
        ],
    )

    limit = st.slider("Limit returned records", 10, 1000, 100)

    if st.button("Query data"):
        try:
            conn = create_connection()
            cursor = conn.cursor()

            cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")

            result = cursor.fetchall()

            if result:
                # Get column names
                cursor.execute(f"SHOW COLUMNS FROM {table_name}")
                columns = [column[0] for column in cursor.fetchall()]

                # Create DataFrame
                df = pd.DataFrame(result, columns=columns)

                st.write(f"{table_name} table data:")
                st.dataframe(df)

                if export_format == "CSV":
                    st.markdown(
                        get_table_download_link(
                            df, f"{table_name}_query.csv", "Download query results"
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        export_to_excel(df, f"{table_name}_query.xlsx"),
                        unsafe_allow_html=True,
                    )
            else:
                st.info(f"No data in {table_name} table")

            cursor.close()
            conn.close()

        except Exception as e:
            st.error(f"Database query failed: {e}")


# Main function
def main():
    st.title("Financial Data Analysis and Prediction System")

    # Sidebar - System configuration
    st.sidebar.header("System Configuration")

    # Initialize database button
    if st.sidebar.button("Initialize Database"):
        init_database()

    # Display system information
    st.sidebar.subheader("System Information")
    st.sidebar.info(
        """
    Version: 1.0.0
    Data Source: BaoStock
    Support: A-Share Market
    Features: Stock data analysis, technical indicators, machine learning prediction
    """
    )

    # Display current date
    st.sidebar.subheader("Current Date")
    st.sidebar.write(datetime.now().strftime("%Y-%m-%d"))

    # Add data cleanup options
    st.sidebar.subheader("Data Management")
    if st.sidebar.button("Clear All Session Data"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.sidebar.success("All session data cleared")

    # Main interface tabs
    tabs = st.tabs(
        [
            "Data Acquisition",
            "Data Analysis",
            "Machine Learning Prediction",
            "Visualization",
            "Export Results",
        ]
    )

    # Data acquisition tab
    with tabs[0]:
        data_acquisition_tab()

    # Data analysis tab
    with tabs[1]:
        data_analysis_tab()

    # Machine learning prediction tab
    with tabs[2]:
        ml_prediction_tab()

    # Visualization tab
    with tabs[3]:
        visualization_tab()

    # Export results tab
    with tabs[4]:
        export_tab()


if __name__ == "__main__":
    main()
