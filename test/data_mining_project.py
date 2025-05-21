import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import baostock as bs
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import jieba
import jieba.analyse
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import pymysql
import mplfinance as mpf
import talib
import io
import base64
import os
import json
import re
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# 设置页面配置
st.set_page_config(
    page_title="金融数据分析与预测系统",
    page_icon="💹",
    layout="wide"
)

# 画图设置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False 

# AI模型API配置
LLM_API_CONFIG = {
    "api_key": st.secrets.get("LLM_API_KEY", "sk-4DKrMT5Du1BR6e1eYvru5kjb7u7FBzPR59cVyChZrC7SJZcg"),
    "base_url": st.secrets.get("LLM_API_BASE_URL", "https://yunwu.ai/v1"),
    "model_name": st.secrets.get("LLM_MODEL_NAME", "gpt-4o"),
    "timeout": 30
}

# 配置MySQL数据库连接
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'db': 'financial_analysis',
    'charset': 'utf8mb4'
}

# 创建数据库连接
def create_connection():
    try:
        conn = pymysql.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            db=DB_CONFIG['db'],
            charset='utf8mb4',
            use_unicode=True
        )
        return conn
    except Exception as e:
        st.error(f"数据库连接失败: {e}")
        return None
        
# 初始化数据库
def init_database():
    try:
        conn = pymysql.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            charset=DB_CONFIG['charset']
        )
        cursor = conn.cursor()
        
        # 创建数据库
        cursor.execute("CREATE DATABASE IF NOT EXISTS financial_analysis")
        cursor.execute("USE financial_analysis")
        
        # 创建股票数据表
        cursor.execute("""
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
        """)
        
        # 创建新闻数据表
        cursor.execute("""
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
        """)
        
        # 创建预测结果表
        cursor.execute("""
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
        """)
        
        # 创建指数数据表
        cursor.execute("""
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
        """)
        
        # 创建财务数据表
        cursor.execute("""
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
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        st.success("数据库初始化成功！")
    except Exception as e:
        st.error(f"数据库初始化失败: {e}")

# 使用通用LLM API进行情感分析
def analyze_sentiment_with_llm(text):
    """
    使用通用LLM API进行文本情感分析
    返回值范围从-1（极负面）到1（极正面）
    """
    try:
        # 截取文本，避免过长
        if len(text) > 1000:
            text = text[:1000]
            
        # API请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLM_API_CONFIG['api_key']}"
        }
        
        # API请求体
        payload = {
            "model": LLM_API_CONFIG["model_name"],
            "messages": [
                {"role": "system", "content": "你是一个专业的金融情感分析AI。请分析以下财经新闻的情感倾向，对于股票市场/企业的影响评分。评分范围从-1（极度负面）到1（极度正面），只需返回一个数字，不要解释。"},
                {"role": "user", "content": f"对以下财经新闻进行情感分析评分(-1到1)，仅返回数字：\n\n{text}"}
            ],
            "temperature": 0.3
        }
        
        # 发送请求
        response = requests.post(
            f"{LLM_API_CONFIG['base_url']}/chat/completions",
            headers=headers,
            json=payload,
            timeout=LLM_API_CONFIG["timeout"]
        )
        
        # 检查响应状态
        if response.status_code == 200:
            response_data = response.json()
            sentiment_score = response_data["choices"][0]["message"]["content"].strip()
            
            # 确保结果是数字
            try:
                sentiment_score = float(sentiment_score)
                # 确保在-1到1范围内
                sentiment_score = max(-1, min(1, sentiment_score))
            except ValueError:
                # 如果解析失败，从文本中提取数字
                import re
                match = re.search(r'(-?\d+(\.\d+)?)', sentiment_score)
                if match:
                    sentiment_score = float(match.group(1))
                    sentiment_score = max(-1, min(1, sentiment_score))
                else:
                    # 默认为中性
                    sentiment_score = 0.0
        else:
            st.warning(f"API请求失败，状态码: {response.status_code}，使用中性评分")
            sentiment_score = 0.0
                
        return sentiment_score
        
    except Exception as e:
        st.warning(f"情感分析API调用失败: {e}，使用中性评分")
        return 0.0  # 出错时返回中性评分

# 通过LLM API进行新闻摘要和关键信息提取
def extract_news_insights_with_llm(news_df):
    """
    使用LLM API对一组新闻进行摘要分析
    返回关键见解和市场趋势分析
    """
    try:
        # 选择最近的10条新闻标题
        recent_news = news_df.sort_values('date', ascending=False).head(10)
        titles = recent_news['title'].tolist()
        
        # 构建新闻摘要请求
        titles_text = "\n".join([f"{i+1}. {title}" for i, title in enumerate(titles)])
        
        # API请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLM_API_CONFIG['api_key']}"
        }
        
        # API请求体
        payload = {
            "model": LLM_API_CONFIG["model_name"],
            "messages": [
                {"role": "system", "content": "你是一个专业的金融分析师。请分析以下财经新闻标题，提取关键市场趋势和见解。"},
                {"role": "user", "content": f"请分析以下财经新闻标题，识别主要市场趋势、潜在投资机会和风险。提供简洁的分析（200字以内）：\n\n{titles_text}"}
            ],
            "temperature": 0.7
        }
        
        # 发送请求
        response = requests.post(
            f"{LLM_API_CONFIG['base_url']}/chat/completions",
            headers=headers,
            json=payload,
            timeout=LLM_API_CONFIG["timeout"]
        )
        
        # 检查响应状态
        if response.status_code == 200:
            response_data = response.json()
            return response_data["choices"][0]["message"]["content"].strip()
        else:
            st.warning(f"API请求失败，状态码: {response.status_code}")
            return "无法获取新闻分析结果，请检查API连接。"
        
    except Exception as e:
        st.warning(f"新闻分析API调用失败: {e}")
        return "无法获取新闻分析结果，请检查API连接。"

# 使用LLM API生成预测结果解读
def interpret_prediction_with_llm(stock_name, prediction_data, model_name, metrics):
    """
    使用LLM API解读预测结果
    生成专业的分析报告
    """
    try:
        # 格式化预测数据
        prediction_text = f"模型: {model_name}\n"
        prediction_text += f"预测准确率/误差: {metrics}\n"
        prediction_text += f"最近预测值: {prediction_data[-1]:.4f}\n"
        
        # API请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLM_API_CONFIG['api_key']}"
        }
        
        # API请求体
        payload = {
            "model": LLM_API_CONFIG["model_name"],
            "messages": [
                {"role": "system", "content": "你是一个专业的金融分析师，擅长解读股票预测模型结果。"},
                {"role": "user", "content": f"请解读以下{stock_name}股票的预测结果，提供专业的分析见解和可能的投资建议（200字左右）：\n\n{prediction_text}"}
            ],
            "temperature": 0.7
        }
        
        # 发送请求
        response = requests.post(
            f"{LLM_API_CONFIG['base_url']}/chat/completions",
            headers=headers,
            json=payload,
            timeout=LLM_API_CONFIG["timeout"]
        )
        
        # 检查响应状态
        if response.status_code == 200:
            response_data = response.json()
            return response_data["choices"][0]["message"]["content"].strip()
        else:
            st.warning(f"API请求失败，状态码: {response.status_code}")
            return "无法获取预测分析结果，请检查API连接。"
        
    except Exception as e:
        st.warning(f"预测解读API调用失败: {e}")
        return "无法获取预测分析结果，请检查API连接。"

# 从BaoStock获取股票数据
def fetch_stock_data(symbol, start_date, end_date):
    """从BaoStock获取股票数据"""
    try:
        # 检查数据库中是否已有数据
        conn = create_connection()
        cursor = conn.cursor()
        
        cursor.execute(f"""
        SELECT * FROM stock_data 
        WHERE symbol = '{symbol}' 
        AND date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY date ASC
        """)
        
        result = cursor.fetchall()
        
        # 如果数据库中已有完整数据，直接返回
        if len(result) > 0:
            df = pd.DataFrame(result, columns=['id', 'symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'created_at'])
            df = df.drop(['id', 'created_at'], axis=1)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df
        
        # 使用BaoStock获取数据
        # 登录BaoStock
        lg = bs.login()
        if lg.error_code != '0':
            st.error(f"BaoStock登录失败: {lg.error_msg}")
            return None
            
        # 构建完整的股票代码
        if symbol.startswith('6'):
            bs_symbol = f"sh.{symbol}"
        elif symbol.startswith(('0', '3')):
            bs_symbol = f"sz.{symbol}"
        else:
            bs_symbol = symbol
            
        # 获取股票数据
        rs = bs.query_history_k_data_plus(
            bs_symbol,
            "date,open,high,low,close,volume,amount,adjustflag",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3"  # 复权类型：3-前复权 2-后复权 1-不复权
        )
        
        if rs.error_code != '0':
            st.error(f"获取股票数据失败: {rs.error_msg}")
            bs.logout()
            return None
            
        # 处理数据
        data_list = []
        while (rs.next()):
            data_list.append(rs.get_row_data())
            
        # 登出BaoStock
        bs.logout()
        
        if not data_list:
            st.warning(f"未找到 {symbol} 从 {start_date} 到 {end_date} 的数据")
            return None
            
        # 转换为DataFrame
        df = pd.DataFrame(data_list, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'adjustflag'])
        
        # 转换数据类型
        df['date'] = pd.to_datetime(df['date'])
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        # 设置日期为索引
        df.set_index('date', inplace=True)
        
        # 保存到数据库
        for idx, row in df.iterrows():
            sql = """
            INSERT INTO stock_data (symbol, date, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (
                symbol,
                idx.strftime('%Y-%m-%d'),
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['volume'])
            ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return df
    except Exception as e:
        st.error(f"获取股票数据失败: {e}")
        try:
            bs.logout()  # 确保登出BaoStock
        except:
            pass
        return None

# 获取股票基本信息函数
def get_stock_basic_info(symbol):
    """获取股票基本信息"""
    try:
        # 格式化股票代码
        if symbol.startswith('6'):
            bs_symbol = f"sh.{symbol}"
        elif symbol.startswith(('0', '3')):
            bs_symbol = f"sz.{symbol}"
        else:
            bs_symbol = symbol
            
        # 登录BaoStock
        lg = bs.login()
        if lg.error_code != '0':
            st.error(f"BaoStock登录失败: {lg.error_msg}")
            return None
            
        # 获取股票基本信息
        rs = bs.query_stock_basic(code=bs_symbol)
        
        if rs.error_code != '0':
            st.error(f"获取股票信息失败: {rs.error_msg}")
            bs.logout()
            return None
            
        data_list = []
        while (rs.next()):
            data_list.append(rs.get_row_data())
            
        # 登出BaoStock
        bs.logout()
        
        if data_list:
            return data_list[0]
        else:
            return None
    except Exception as e:
        st.error(f"获取股票基本信息失败: {e}")
        try:
            bs.logout()
        except:
            pass
        return None

# 获取全部股票列表
def get_stock_list():
    """获取所有A股股票列表"""
    try:
        # 登录BaoStock
        lg = bs.login()
        if lg.error_code != '0':
            st.error(f"BaoStock登录失败: {lg.error_msg}")
            return []
            
        # 获取A股代码列表
        rs = bs.query_stock_basic()
        
        if rs.error_code != '0':
            st.error(f"获取股票列表失败: {rs.error_msg}")
            bs.logout()
            return []
            
        data_list = []
        while (rs.next()):
            row = rs.get_row_data()
            # 只选取A股且状态为上市的股票
            if row[4] == '1' and row[5] == '1':  # type=1表示A股，status=1表示上市
                data_list.append({'code': row[0], 'name': row[1]})
                
        # 登出BaoStock
        bs.logout()
        
        return data_list
    except Exception as e:
        st.error(f"获取股票列表失败: {e}")
        try:
            bs.logout()
        except:
            pass
        return []

# 获取行业分类数据
def get_industry_data():
    """获取行业分类数据"""
    try:
        # 登录BaoStock
        lg = bs.login()
        if lg.error_code != '0':
            st.error(f"BaoStock登录失败: {lg.error_msg}")
            return None
            
        # 获取行业分类数据
        rs = bs.query_stock_industry()
        
        if rs.error_code != '0':
            st.error(f"获取行业分类失败: {rs.error_msg}")
            bs.logout()
            return None
            
        industry_list = []
        while (rs.next()):
            industry_list.append(rs.get_row_data())
            
        # 登出BaoStock
        bs.logout()
        
        if industry_list:
            df = pd.DataFrame(industry_list, columns=[
                'code', 'code_name', 'industry', 'industry_classification'
            ])
            return df
        else:
            return None
    except Exception as e:
        st.error(f"获取行业分类数据失败: {e}")
        try:
            bs.logout()
        except:
            pass
        return None

# 获取指数数据
def fetch_index_data(index_code, start_date, end_date):
    """获取指数数据"""
    try:
        # 检查数据库中是否已有数据
        conn = create_connection()
        cursor = conn.cursor()
        
        index_name = {
            'sh.000001': '上证指数',
            'sz.399001': '深证成指',
            'sz.399006': '创业板指',
            'sh.000016': '上证50',
            'sh.000300': '沪深300',
            'sz.399905': '中证500'
        }.get(index_code, index_code)
        
        cursor.execute(f"""
        SELECT * FROM index_data 
        WHERE index_code = '{index_code}' 
        AND date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY date ASC
        """)
        
        result = cursor.fetchall()
        
        # 如果数据库中已有完整数据，直接返回
        if len(result) > 0:
            df = pd.DataFrame(result, columns=['id', 'index_code', 'index_name', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'created_at'])
            df = df.drop(['id', 'index_code', 'index_name', 'created_at'], axis=1)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df
        
        # 登录BaoStock
        lg = bs.login()
        if lg.error_code != '0':
            st.error(f"BaoStock登录失败: {lg.error_msg}")
            return None
            
        # 获取指数数据
        rs = bs.query_history_k_data_plus(
            index_code,
            "date,open,high,low,close,volume,amount",
            start_date=start_date,
            end_date=end_date,
            frequency="d"
        )
        
        if rs.error_code != '0':
            st.error(f"获取指数数据失败: {rs.error_msg}")
            bs.logout()
            return None
            
        data_list = []
        while (rs.next()):
            data_list.append(rs.get_row_data())
            
        # 登出BaoStock
        bs.logout()
        
        if not data_list:
            st.warning(f"未找到指数 {index_code} 从 {start_date} 到 {end_date} 的数据")
            return None
            
        # 转换为DataFrame
        df = pd.DataFrame(data_list, columns=[
            'date', 'open', 'high', 'low', 'close', 'volume', 'amount'
        ])
        
        # 转换数据类型
        df['date'] = pd.to_datetime(df['date'])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # 设置日期为索引
        df.set_index('date', inplace=True)
        
        # 保存到数据库
        for idx, row in df.iterrows():
            sql = """
            INSERT INTO index_data (index_code, index_name, date, open, high, low, close, volume, amount)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (
                index_code,
                index_name,
                idx.strftime('%Y-%m-%d'),
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['volume']),
                float(row['amount'])
            ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return df
    except Exception as e:
        st.error(f"获取指数数据失败: {e}")
        try:
            bs.logout()
        except:
            pass
        return None

# 获取交易日历
def get_trade_calendar(start_date, end_date):
    """获取交易日历"""
    try:
        # 登录BaoStock
        lg = bs.login()
        if lg.error_code != '0':
            st.error(f"BaoStock登录失败: {lg.error_msg}")
            return None
            
        # 获取交易日历
        rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
        
        if rs.error_code != '0':
            st.error(f"获取交易日历失败: {rs.error_msg}")
            bs.logout()
            return None
            
        data_list = []
        while (rs.next()):
            data_list.append(rs.get_row_data())
            
        # 登出BaoStock
        bs.logout()
        
        if data_list:
            df = pd.DataFrame(data_list, columns=['calendar_date', 'is_trading_day'])
            df['calendar_date'] = pd.to_datetime(df['calendar_date'])
            df['is_trading_day'] = df['is_trading_day'].apply(lambda x: True if x == '1' else False)
            return df
        else:
            return None
    except Exception as e:
        st.error(f"获取交易日历失败: {e}")
        try:
            bs.logout()
        except:
            pass
        return None

# 获取公司财务数据
def get_financial_data(symbol, year, quarter):
    """获取公司财务数据"""
    try:
        # 格式化股票代码
        if symbol.startswith('6'):
            bs_symbol = f"sh.{symbol}"
        elif symbol.startswith(('0', '3')):
            bs_symbol = f"sz.{symbol}"
        else:
            bs_symbol = symbol
            
        # 登录BaoStock
        lg = bs.login()
        if lg.error_code != '0':
            st.error(f"BaoStock登录失败: {lg.error_msg}")
            return None
            
        # 获取季度业绩报表
        rs = bs.query_performance_express_report(bs_symbol, year, quarter)
        
        if rs.error_code != '0':
            st.error(f"获取季度业绩报表失败: {rs.error_msg}")
            bs.logout()
            return None
            
        data_list = []
        while (rs.next()):
            data_list.append(rs.get_row_data())
            
        # 获取利润表
        rs_profit = bs.query_profit_data(bs_symbol, year=year, quarter=quarter)
        
        profit_data = []
        if rs_profit.error_code == '0':
            while (rs_profit.next()):
                profit_data.append(rs_profit.get_row_data())
                
        # 获取资产负债表
        rs_balance = bs.query_balance_data(bs_symbol, year=year, quarter=quarter)
        
        balance_data = []
        if rs_balance.error_code == '0':
            while (rs_balance.next()):
                balance_data.append(rs_balance.get_row_data())
                
        # 获取现金流量表
        rs_cash = bs.query_cash_flow_data(bs_symbol, year=year, quarter=quarter)
        
        cash_flow_data = []
        if rs_cash.error_code == '0':
            while (rs_cash.next()):
                cash_flow_data.append(rs_cash.get_row_data())
        
        # 登出BaoStock
        bs.logout()
        
        result = {
            'performance': pd.DataFrame(data_list) if data_list else None,
            'profit': pd.DataFrame(profit_data) if profit_data else None,
            'balance': pd.DataFrame(balance_data) if balance_data else None,
            'cash_flow': pd.DataFrame(cash_flow_data) if cash_flow_data else None
        }
        
        return result
    except Exception as e:
        st.error(f"获取财务数据失败: {e}")
        try:
            bs.logout()
        except:
            pass
        return None

# 爬取新闻数据
def fetch_news_data(keyword, num_pages=2):
    """爬取新浪财经新闻数据"""
    try:
        all_news = []
        
        for page in range(1, num_pages + 1):
            url = f"https://search.sina.com.cn/?q={keyword}&c=news&from=&col=&range=&source=&country=&size=&time=&a=&page={page}&pf=0&ps=0&dpc=1"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            news_items = soup.select('.box-result')
            
            for item in news_items:
                try:
                    title_elem = item.select_one('h2 a')
                    if not title_elem:
                        continue
                        
                    title = title_elem.text.strip()
                    link = title_elem['href']
                    
                    # 提取日期
                    time_elem = item.select_one('.fgray_time')
                    news_date = datetime.now().strftime('%Y-%m-%d')
                    if time_elem:
                        date_str = time_elem.text.strip()
                        if '年' in date_str and '月' in date_str and '日' in date_str:
                            date_match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', date_str)
                            if date_match:
                                year, month, day = date_match.groups()
                                news_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                    
                    # 获取新闻内容
                    content = ""
                    try:
                        news_response = requests.get(link, headers=headers, timeout=5)
                        news_soup = BeautifulSoup(news_response.text, 'html.parser')
                        content_elem = news_soup.select_one('.article-content') or news_soup.select_one('#artibody')
                        if content_elem:
                            paras = content_elem.select('p')
                            content = '\n'.join([p.text.strip() for p in paras])
                    except Exception as e:
                        content = "获取内容失败"
                    
                    # 提取关键词
                    keywords = jieba.analyse.extract_tags(title + ' ' + content, topK=5, withWeight=False)
                    keywords_str = ','.join(keywords)
                    
                    # 调用大模型API进行情感分析
                    sentiment = analyze_sentiment_with_llm(title + ' ' + content)
                    
                    news_data = {
                        'title': title,
                        'content': content[:500],  # 只保存部分内容
                        'source': 'sina',
                        'date': news_date,
                        'sentiment': sentiment,
                        'keywords': keywords_str
                    }
                    
                    all_news.append(news_data)
                    
                    # 保存到数据库
                    conn = create_connection()
                    cursor = conn.cursor()
                    
                    # 检查是否已存在
                    cursor.execute("SELECT id FROM news_data WHERE title = %s", (title,))
                    if cursor.fetchone() is None:
                        sql = """
                        INSERT INTO news_data (title, content, source, date, sentiment, keywords)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """
                        try:
                            cursor.execute(sql, (
                                title,
                                content[:1000],
                                'sina',
                                news_date,
                                sentiment,
                                keywords_str
                            ))
                            conn.commit()
                        except Exception as e:
                            st.warning(f"保存新闻数据到数据库失败: {e}")
                    
                    cursor.close()
                    conn.close()
                    
                except Exception as e:
                    st.warning(f"处理单条新闻时出错: {e}")
                    continue
        
        return pd.DataFrame(all_news)
    except Exception as e:
        st.error(f"获取新闻数据失败: {e}")
        return pd.DataFrame()

# 数据处理函数
def preprocess_stock_data(df):
    """
    预处理股票数据，计算技术指标
    """
    # 检查是否有数据
    if df is None or df.empty:
        return None
    
    # 复制一份数据，避免修改原数据
    df_processed = df.copy()
    
    # 计算移动平均线
    df_processed['MA5'] = df_processed['close'].rolling(window=5).mean()
    df_processed['MA10'] = df_processed['close'].rolling(window=10).mean()
    df_processed['MA20'] = df_processed['close'].rolling(window=20).mean()
    
    # 计算收益率
    df_processed['Daily_Return'] = df_processed['close'].pct_change()
    
    # 计算波动率（标准差）
    df_processed['Volatility_5d'] = df_processed['Daily_Return'].rolling(window=5).std()
    
    # 计算MACD
    close = df_processed['close'].values
    if len(close) > 26:
        df_processed['EMA12'] = talib.EMA(close, timeperiod=12)
        df_processed['EMA26'] = talib.EMA(close, timeperiod=26)
        df_processed['MACD'] = df_processed['EMA12'] - df_processed['EMA26']
        df_processed['Signal'] = talib.EMA(df_processed['MACD'].values, timeperiod=9)
        df_processed['Histogram'] = df_processed['MACD'] - df_processed['Signal']
    
    # 计算RSI
    if len(close) > 14:
        df_processed['RSI'] = talib.RSI(close, timeperiod=14)
    
    # 处理缺失值
    df_processed.fillna(method='bfill', inplace=True)
    
    # 添加趋势标签（分类问题使用）
    df_processed['Target_Classification'] = 0
    df_processed.loc[df_processed['close'].shift(-1) > df_processed['close'], 'Target_Classification'] = 1
    
    # 添加未来n天价格变化（回归问题使用）
    for days in [1, 3, 5]:
        df_processed[f'Target_Regression_{days}d'] = df_processed['close'].shift(-days) / df_processed['close'] - 1
    
    # 再次处理可能出现的缺失值
    df_processed.dropna(inplace=True)
    
    return df_processed

# 特征工程函数
def create_features(df, target_col, prediction_days=5):
    """
    创建特征和目标变量
    """
    # 特征列
    feature_columns = ['open', 'high', 'low', 'close', 'volume', 
                        'MA5', 'MA10', 'MA20', 'Daily_Return', 
                        'Volatility_5d']
    
    if 'RSI' in df.columns:
        feature_columns.extend(['RSI'])
        
    if 'MACD' in df.columns:
        feature_columns.extend(['MACD', 'Signal', 'Histogram'])
    
    # 提取特征和目标
    X = df[feature_columns].values
    y = df[target_col].values
    
    # 归一化特征
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, scaler, feature_columns

# 机器学习模型训练函数
def train_model(X_train, y_train, model_name):
    """
    训练机器学习模型
    """
    if model_name == "linear_regression":
        model = LinearRegression()
    elif model_name == "decision_tree":
        model = DecisionTreeRegressor(max_depth=5, random_state=42)
    elif model_name == "random_forest":
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    elif model_name == "svm":
        model = SVR(kernel='rbf')
    elif model_name == "decision_tree_classifier":
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
    elif model_name == "random_forest_classifier":
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    elif model_name == "svm_classifier":
        model = SVC(kernel='rbf', probability=True)
    else:
        raise ValueError("不支持的模型类型")
    
    model.fit(X_train, y_train)
    return model

# K-Means聚类函数
def cluster_stocks(df, n_clusters=3):
    """
    使用K-Means聚类算法对股票数据进行聚类
    """
    # 选择用于聚类的特征
    features = ['Daily_Return', 'Volatility_5d']
    if 'RSI' in df.columns:
        features.append('RSI')
    
    # 提取特征
    X = df[features].dropna().values
    
    # 归一化
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 应用K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # 添加聚类标签到数据框
    cluster_df = df.copy()
    cluster_df = cluster_df.dropna(subset=features)
    cluster_df['Cluster'] = clusters
    
    return cluster_df, kmeans.cluster_centers_

# 可视化函数
def plot_stock_price(df):
    """绘制股票价格走势图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index, df['close'], label='收盘价')
    ax.plot(df.index, df['MA5'], label='5日均线')
    ax.plot(df.index, df['MA20'], label='20日均线')
    ax.set_title('股票价格走势图')
    ax.set_xlabel('日期')
    ax.set_ylabel('价格')
    ax.legend()
    ax.grid(True)
    return fig

def plot_candlestick(df):
    """绘制K线图"""
    df_plot = df.copy()
    df_plot.index.name = 'Date'
    
    # 重命名列以匹配mplfinance要求
    df_plot = df_plot.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    
    # 使用mplfinance绘制K线图
    fig, axlist = mpf.plot(df_plot, type='candle', volume=True, 
                          title='股票K线图',
                          ylabel='价格',
                          ylabel_lower='成交量',
                          style='charles',
                          returnfig=True,
                          figsize=(10, 8))
    
    return fig

def plot_technical_indicators(df):
    """绘制技术指标"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # MACD图
    if 'MACD' in df.columns:
        ax1.plot(df.index, df['MACD'], label='MACD')
        ax1.plot(df.index, df['Signal'], label='Signal')
        ax1.bar(df.index, df['Histogram'], label='Histogram', alpha=0.5)
        ax1.set_title('MACD指标')
        ax1.legend()
        ax1.grid(True)
    
    # RSI图
    if 'RSI' in df.columns:
        ax2.plot(df.index, df['RSI'], color='purple', label='RSI')
        ax2.axhline(y=70, color='r', linestyle='-', alpha=0.3)
        ax2.axhline(y=30, color='g', linestyle='-', alpha=0.3)
        ax2.set_title('RSI指标')
        ax2.set_ylabel('RSI')
        ax2.set_xlabel('日期')
        ax2.legend()
        ax2.grid(True)
        
    plt.tight_layout()
    return fig

def plot_prediction_results(y_test, y_pred, model_name):
    """绘制预测结果对比图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(y_test, label='实际值')
    ax.plot(y_pred, label='预测值')
    ax.set_title(f'{model_name} 预测结果')
    ax.set_xlabel('样本')
    ax.set_ylabel('值')
    ax.legend()
    ax.grid(True)
    
    return fig

def plot_correlation_heatmap(df):
    """绘制相关性热力图"""
    # 选择数值列
    numeric_df = df.select_dtypes(include=[np.number])
    
    # 计算相关性
    corr = numeric_df.corr()
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, fmt=".2f")
    ax.set_title('特征相关性热力图')
    
    return fig

def plot_clusters(df, feature_x, feature_y):
    """绘制聚类结果"""
    if 'Cluster' not in df.columns:
        return None
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 为每个聚类绘制散点图
    clusters = df['Cluster'].unique()
    for cluster in clusters:
        cluster_data = df[df['Cluster'] == cluster]
        ax.scatter(cluster_data[feature_x], cluster_data[feature_y], 
                   label=f'聚类 {cluster}', alpha=0.7)
    
    ax.set_title('股票聚类结果')
    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.legend()
    ax.grid(True)
    
    return fig

def plot_news_sentiment(news_df):
    """绘制新闻情感分析"""
    if news_df.empty or 'sentiment' not in news_df.columns:
        return None
        
    # 按日期分组计算平均情感得分
    if 'date' in news_df.columns:
        sentiment_by_date = news_df.groupby('date')['sentiment'].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(sentiment_by_date['date'], sentiment_by_date['sentiment'], color='skyblue')
        ax.set_title('每日新闻情感得分')
        ax.set_xlabel('日期')
        ax.set_ylabel('情感得分（负面→正面）')
        ax.grid(True, axis='y')
        plt.xticks(rotation=45)
        
        return fig
    return None

def generate_news_wordcloud(news_df):
    """生成新闻关键词词云"""
    if news_df.empty:
        return None
        
    # 提取所有关键词
    all_keywords = []
    if 'keywords' in news_df.columns:
        for keywords in news_df['keywords']:
            if isinstance(keywords, str):
                all_keywords.extend(keywords.split(','))
    
    # 如果没有关键词，使用标题
    if not all_keywords and 'title' in news_df.columns:
        text = ' '.join(news_df['title'].dropna().tolist())
        words = jieba.cut(text)
        all_keywords = [w for w in words if len(w) > 1]
    
    if not all_keywords:
        return None
        
    text = ' '.join(all_keywords)
    
    # 生成词云
    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                          font_path='simhei.ttf' if os.path.exists('simhei.ttf') else None,
                          max_words=100).generate(text)
    
    # 显示词云
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('新闻关键词词云')
    
    return fig

# 保存结果函数
def save_prediction_to_db(symbol, model_name, prediction_date, predicted_value, actual_value, accuracy, model_params):
    """将预测结果保存到数据库"""
    try:
        conn = create_connection()
        cursor = conn.cursor()
        
        sql = """
        INSERT INTO prediction_results 
        (symbol, model_name, prediction_date, predicted_value, actual_value, accuracy, model_params)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        
        cursor.execute(sql, (
            symbol,
            model_name,
            prediction_date,
            float(predicted_value),
            float(actual_value) if actual_value is not None else None,
            float(accuracy),
            json.dumps(model_params)
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True
    except Exception as e:
        st.error(f"保存预测结果到数据库失败: {e}")
        return False

def get_table_download_link(df, filename, text):
    """生成数据下载链接"""
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def export_to_excel(df, filename):
    """导出数据到Excel"""
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.close()
    output.seek(0)
    
    b64 = base64.b64encode(output.read()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">{filename}</a>'
    return href

# 数据获取标签页
def data_acquisition_tab():
    st.header("数据获取")
    
    # 创建两列
    col1, col2 = st.columns(2)
    
    # 股票数据获取
    with col1:
        st.subheader("股票数据获取")
        
        # 增加股票代码选项
        stock_input_type = st.radio(
            "选择输入方式",
            ["直接输入", "从列表选择"],
            horizontal=True
        )
        
        if stock_input_type == "直接输入":
            symbol = st.text_input("股票代码", "600000")
            st.caption("A股代码格式: 600000(上证) 或 000001(深证)")
        else:
            try:
                # 获取A股代码列表
                with st.spinner("正在获取股票列表..."):
                    stock_list = get_stock_list()
                
                if stock_list:
                    # 转换为可选择的格式
                    stock_options = [f"{stock['code'].split('.')[-1]} - {stock['name']}" for stock in stock_list[:1000]]  # 限制数量避免过多
                    selected_stock = st.selectbox("选择股票", stock_options)
                    symbol = selected_stock.split(' - ')[0]  # 提取股票代码
                else:
                    st.error("获取股票列表失败")
                    symbol = st.text_input("股票代码", "600000")
            except Exception as e:
                st.error(f"获取股票列表失败: {e}")
                symbol = st.text_input("股票代码", "600000")
        
        start_date = st.date_input("开始日期", datetime.now() - timedelta(days=365))
        end_date = st.date_input("结束日期", datetime.now())
        
        if st.button("获取股票数据"):
            with st.spinner("正在获取股票数据..."):
                # 保存股票代码到会话状态
                st.session_state.stock_symbol = symbol
                
                stock_df = fetch_stock_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                if stock_df is not None and not stock_df.empty:
                    st.session_state.stock_df = stock_df
                    st.success(f"成功获取 {symbol} 的股票数据，共 {len(stock_df)} 条记录")
                    st.dataframe(stock_df.head())
                else:
                    st.error("获取股票数据失败")
                    
        # 添加获取股票基本信息按钮
        if 'stock_symbol' in st.session_state:
            if st.button("获取股票基本信息"):
                with st.spinner("正在获取股票基本信息..."):
                    try:
                        symbol = st.session_state.stock_symbol
                        stock_info = get_stock_basic_info(symbol)
                        
                        if stock_info:
                            st.write("股票基本信息:")
                            st.write(f"股票代码: {stock_info[0]}")
                            st.write(f"股票名称: {stock_info[1]}")
                            st.write(f"上市日期: {stock_info[2]}")
                            st.write(f"退市日期: {stock_info[3] if stock_info[3] else '至今'}")
                            st.write(f"股票类型: {'A股' if stock_info[4]=='1' else '其他'}")
                            st.write(f"状态: {'上市' if stock_info[5]=='1' else '退市'}")
                        else:
                            st.warning(f"未找到股票 {symbol} 的基本信息")
                    except Exception as e:
                        st.error(f"获取股票基本信息失败: {e}")
        
        # 添加获取财务数据按钮
        if 'stock_symbol' in st.session_state:
            st.subheader("财务数据获取")
            
            col_year, col_quarter = st.columns(2)
            with col_year:
                year = st.selectbox("选择年份", list(range(datetime.now().year, 2005, -1)))
            with col_quarter:
                quarter = st.selectbox("选择季度", [1, 2, 3, 4])
                
            if st.button("获取财务数据"):
                with st.spinner("正在获取财务数据..."):
                    try:
                        symbol = st.session_state.stock_symbol
                        financial_data = get_financial_data(symbol, year, quarter)
                        
                        if financial_data:
                            # 显示业绩报表
                            if financial_data['performance'] is not None and not financial_data['performance'].empty:
                                st.write("季度业绩报表:")
                                st.dataframe(financial_data['performance'])
                                
                            # 显示利润表摘要
                            if financial_data['profit'] is not None and not financial_data['profit'].empty:
                                st.write("利润表摘要:")
                                profit_summary = financial_data['profit']
                                # 选择关键指标显示
                                key_profit_metrics = ['code', 'pubDate', 'statDate', 'roeAvg', 'npMargin', 'gpMargin', 'netProfit', 'epsTTM', 'MBRevenue', 'totalShare']
                                metrics_to_show = [col for col in key_profit_metrics if col in financial_data['profit'].columns]
                                st.dataframe(financial_data['profit'][metrics_to_show])
                                
                                # 保存到会话状态
                                st.session_state.financial_data = financial_data
                            else:
                                st.warning(f"未找到 {symbol} {year}年第{quarter}季度的财务数据")
                        else:
                            st.warning(f"未找到 {symbol} {year}年第{quarter}季度的财务数据")
                    except Exception as e:
                        st.error(f"获取财务数据失败: {e}")
    
    # 指数数据获取和新闻数据
    with col2:
        st.subheader("指数数据获取")
        
        # 主要指数列表
        major_indices = {
            'sh.000001': '上证指数',
            'sz.399001': '深证成指',
            'sz.399006': '创业板指',
            'sh.000016': '上证50',
            'sh.000300': '沪深300',
            'sz.399905': '中证500'
        }
        
        selected_index = st.selectbox("选择指数", list(major_indices.items()), format_func=lambda x: x[1])
        
        index_start_date = st.date_input("指数开始日期", datetime.now() - timedelta(days=365), key="index_start")
        index_end_date = st.date_input("指数结束日期", datetime.now(), key="index_end")
        
        if st.button("获取指数数据"):
            with st.spinner("正在获取指数数据..."):
                index_code = selected_index[0]
                index_df = fetch_index_data(index_code, index_start_date.strftime('%Y-%m-%d'), index_end_date.strftime('%Y-%m-%d'))
                
                if index_df is not None and not index_df.empty:
                    st.session_state.index_df = index_df
                    st.session_state.index_name = selected_index[1]
                    st.success(f"成功获取 {selected_index[1]} 的指数数据，共 {len(index_df)} 条记录")
                    st.dataframe(index_df.head())
                else:
                    st.error("获取指数数据失败")
        
        # 新闻数据爬取
        st.subheader("新闻数据爬取")
        
        news_keyword = st.text_input("搜索关键词", "阿里巴巴")
        news_pages = st.slider("爬取页数", 1, 5, 2)
        
        if st.button("获取新闻数据"):
            with st.spinner("正在获取新闻数据..."):
                news_df = fetch_news_data(news_keyword, news_pages)
                if not news_df.empty:
                    st.session_state.news_df = news_df
                    st.success(f"成功获取 {len(news_df)} 条新闻数据")
                    st.dataframe(news_df[['title', 'date', 'sentiment']].head())
                else:
                    st.error("获取新闻数据失败")
    
    # 行业分类数据
    st.subheader("行业分类数据")
    
    if st.button("获取行业分类数据"):
        with st.spinner("正在获取行业分类数据..."):
            industry_df = get_industry_data()
            
            if industry_df is not None and not industry_df.empty:
                st.session_state.industry_df = industry_df
                st.success(f"成功获取行业分类数据，共 {len(industry_df)} 条记录")
                
                # 显示行业统计
                industry_counts = industry_df['industry'].value_counts().reset_index()
                industry_counts.columns = ['行业', '上市公司数量']
                
                # 显示行业分布图表
                st.write("行业分布:")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(industry_counts['行业'][:15], industry_counts['上市公司数量'][:15])
                ax.set_xlabel('行业')
                ax.set_ylabel('上市公司数量')
                ax.set_title('各行业上市公司数量(前15名)')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                
                # 显示行业分类数据表格
                st.write("行业分类数据:")
                st.dataframe(industry_df.head(20))
            else:
                st.error("获取行业分类数据失败")
                
    # 交易日历数据
    st.subheader("交易日历数据")
    
    calendar_col1, calendar_col2 = st.columns(2)
    with calendar_col1:
        calendar_start = st.date_input("日历开始日期", datetime.now() - timedelta(days=30), key="calendar_start")
    with calendar_col2:
        calendar_end = st.date_input("日历结束日期", datetime.now(), key="calendar_end")
        
    if st.button("获取交易日历"):
        with st.spinner("正在获取交易日历..."):
            calendar_df = get_trade_calendar(calendar_start.strftime('%Y-%m-%d'), calendar_end.strftime('%Y-%m-%d'))
            
            if calendar_df is not None and not calendar_df.empty:
                st.session_state.calendar_df = calendar_df
                st.success(f"成功获取交易日历，共 {len(calendar_df)} 条记录")
                
                # 显示交易日和非交易日数量
                trading_days = calendar_df[calendar_df['is_trading_day']].shape[0]
                non_trading_days = calendar_df[~calendar_df['is_trading_day']].shape[0]
                
                st.write(f"交易日数量: {trading_days}")
                st.write(f"非交易日数量: {non_trading_days}")
                
                # 显示交易日历表格
                st.write("交易日历:")
                st.dataframe(calendar_df)
            else:
                st.error("获取交易日历失败")
                
    # 文件上传
    st.subheader("上传CSV/Excel文件")
    uploaded_file = st.file_uploader("选择文件", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"成功上传文件，共 {len(df)} 条记录")
            st.dataframe(df.head())
            st.session_state.uploaded_df = df
            
            # 检查是否为股票数据
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                st.info("检测到上传的是股票数据，可在数据分析标签页进行处理")
        except Exception as e:
            st.error(f"读取文件失败: {e}")

# 数据分析标签页
def data_analysis_tab():
    st.header("数据分析")
    
    # 检查是否有股票数据
    if 'stock_df' in st.session_state:
        st.subheader("股票数据预处理")
        
        if st.button("数据预处理"):
            with st.spinner("正在处理数据..."):
                processed_df = preprocess_stock_data(st.session_state.stock_df)
                if processed_df is not None:
                    st.session_state.processed_df = processed_df
                    st.success("数据预处理完成")
                    st.write("处理后的数据样本：")
                    st.dataframe(processed_df.head())
                    
                    # 显示基本统计信息
                    st.subheader("基本统计信息")
                    st.write(processed_df.describe())
                    
                    # 显示技术指标
                    st.subheader("技术指标")
                    tech_indicators = ['close', 'MA5', 'MA20']
                    if 'RSI' in processed_df.columns:
                        tech_indicators.append('RSI')
                    st.line_chart(processed_df[tech_indicators])
        
        if 'processed_df' in st.session_state:
            # 相关性分析
            st.subheader("相关性分析")
            if st.button("生成相关性矩阵"):
                # 选择数值列
                numeric_df = st.session_state.processed_df.select_dtypes(include=[np.number])
                corr = numeric_df.corr()
                
                # 显示相关性矩阵
                st.write("特征相关性矩阵：")
                st.dataframe(corr.style.background_gradient(cmap='coolwarm'))
                
                # 相关性热力图
                fig = plot_correlation_heatmap(st.session_state.processed_df)
                st.pyplot(fig)
                
            # 添加数据可视化选项
            st.subheader("数据可视化")
            
            viz_type = st.selectbox(
                "选择可视化类型",
                ["K线图", "技术指标", "收益率分布"]
            )
            
            if viz_type == "K线图":
                if st.button("生成K线图"):
                    fig = plot_candlestick(st.session_state.processed_df)
                    st.pyplot(fig)
            
            elif viz_type == "技术指标":
                if st.button("生成技术指标图"):
                    fig = plot_technical_indicators(st.session_state.processed_df)
                    st.pyplot(fig)
            
            elif viz_type == "收益率分布":
                if st.button("生成收益率分布图"):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(st.session_state.processed_df['Daily_Return'].dropna(), bins=50)
                    ax.set_title("日收益率分布")
                    ax.set_xlabel("日收益率")
                    ax.set_ylabel("频率")
                    ax.grid(True)
                    st.pyplot(fig)
    elif 'uploaded_df' in st.session_state:
        if all(col in st.session_state.uploaded_df.columns for col in ['open', 'high', 'low', 'close']):
            st.info("使用上传的股票数据")
            if st.button("处理上传的数据"):
                with st.spinner("正在处理数据..."):
                    processed_df = preprocess_stock_data(st.session_state.uploaded_df)
                    if processed_df is not None:
                        st.session_state.processed_df = processed_df
                        st.success("数据预处理完成")
                        st.write("处理后的数据样本：")
                        st.dataframe(processed_df.head())
        else:
            st.info("上传的文件不是标准股票数据格式，请确保包含open, high, low, close列")
    else:
        st.info("请先在数据获取标签页中获取股票数据")
    
    # 指数数据分析
    if 'index_df' in st.session_state:
        st.subheader("指数数据分析")
        
        if st.button("分析指数数据"):
            with st.spinner("正在分析指数数据..."):
                index_df = st.session_state.index_df
                index_name = st.session_state.index_name
                
                # 计算指数的移动平均线
                index_df['MA5'] = index_df['close'].rolling(window=5).mean()
                index_df['MA20'] = index_df['close'].rolling(window=20).mean()
                index_df['MA60'] = index_df['close'].rolling(window=60).mean()
                
                # 计算指数的收益率
                index_df['Daily_Return'] = index_df['close'].pct_change()
                
                # 显示指数基本统计信息
                st.write(f"{index_name}基本统计信息：")
                st.dataframe(index_df.describe())
                
                # 绘制指数走势图
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(index_df.index, index_df['close'], label='收盘价')
                ax.plot(index_df.index, index_df['MA5'], label='5日均线')
                ax.plot(index_df.index, index_df['MA20'], label='20日均线')
                ax.plot(index_df.index, index_df['MA60'], label='60日均线')
                ax.set_title(f'{index_name}走势图')
                ax.set_xlabel('日期')
                ax.set_ylabel('价格')
                ax.legend()
                ax.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # 绘制指数收益率分布图
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(index_df['Daily_Return'].dropna(), bins=50)
                ax.set_title(f'{index_name}日收益率分布')
                ax.set_xlabel('日收益率')
                ax.set_ylabel('频率')
                ax.grid(True)
                st.pyplot(fig)
                
                # 如果有股票数据，计算与指数的相关性
                if 'processed_df' in st.session_state:
                    st.subheader("股票与指数相关性分析")
                    
                    stock_df = st.session_state.processed_df
                    
                    # 确保索引是日期类型
                    if not isinstance(stock_df.index, pd.DatetimeIndex):
                        stock_df.index = pd.to_datetime(stock_df.index)
                    
                    # 计算共同的日期范围
                    common_dates = sorted(set(stock_df.index) & set(index_df.index))
                    
                    if common_dates:
                        # 提取共同日期范围的数据
                        stock_returns = stock_df.loc[common_dates, 'Daily_Return']
                        index_returns = index_df.loc[common_dates, 'Daily_Return']
                        
                        # 计算相关系数
                        correlation = stock_returns.corr(index_returns)
                        
                        st.write(f"股票与{index_name}收益率相关系数: {correlation:.4f}")
                        
                        # 绘制散点图
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.scatter(index_returns, stock_returns, alpha=0.5)
                        ax.set_title(f'股票收益率 vs {index_name}收益率 (相关系数: {correlation:.4f})')
                        ax.set_xlabel(f'{index_name}日收益率')
                        ax.set_ylabel('股票日收益率')
                        ax.grid(True)
                        
                        # 添加回归线
                        z = np.polyfit(index_returns, stock_returns, 1)
                        p = np.poly1d(z)
                        ax.plot(index_returns, p(index_returns), "r--")
                        
                        st.pyplot(fig)
                        
                        # 计算Beta系数
                        beta = correlation * (stock_returns.std() / index_returns.std())
                        st.write(f"Beta系数（市场敏感度）: {beta:.4f}")
                        st.write(f"Beta解释: {'高于市场波动性' if beta > 1 else '低于市场波动性'}")
    
    # 新闻数据分析
    if 'news_df' in st.session_state:
        st.subheader("新闻数据分析")
        
        # 按日期聚合新闻
        if 'date' in st.session_state.news_df.columns:
            news_count_by_date = st.session_state.news_df.groupby('date').size().reset_index(name='count')
            st.write("每日新闻数量：")
            st.bar_chart(news_count_by_date.set_index('date'))
        
        # 情感分析
        if 'sentiment' in st.session_state.news_df.columns:
            st.write("新闻情感分布：")
            fig, ax = plt.subplots()
            ax.hist(st.session_state.news_df['sentiment'], bins=20)
            ax.set_xlabel('情感得分')
            ax.set_ylabel('频率')
            ax.set_title('新闻情感分布直方图')
            st.pyplot(fig)
            
            # 平均情感得分
            avg_sentiment = st.session_state.news_df['sentiment'].mean()
            st.metric("平均情感得分", f"{avg_sentiment:.3f}", 
                      delta="积极" if avg_sentiment > 0 else "消极")
            
            # 生成词云
            st.write("新闻关键词词云：")
            fig = generate_news_wordcloud(st.session_state.news_df)
            if fig:
                st.pyplot(fig)
            
            # 使用大模型分析新闻洞察
            st.subheader("新闻洞察分析")
            if st.button("分析新闻趋势"):
                with st.spinner("正在使用AI分析新闻趋势..."):
                    insights = extract_news_insights_with_llm(st.session_state.news_df)
                    st.info(insights)
                    
                    # 保存新闻洞察到会话状态
                    st.session_state.news_insights = insights
    else:
        st.info("请先在数据获取标签页中获取新闻数据")
    
    # 行业数据分析
    if 'industry_df' in st.session_state:
        st.subheader("行业数据分析")
        
        industry_df = st.session_state.industry_df
        
        # 显示行业统计
        industry_counts = industry_df['industry'].value_counts()
        
        # 选择特定行业分析
        selected_industry = st.selectbox(
            "选择要分析的行业",
            options=sorted(industry_df['industry'].unique())
        )
        
        if st.button("分析行业股票"):
            with st.spinner("正在分析行业股票..."):
                # 筛选所选行业的股票
                industry_stocks = industry_df[industry_df['industry'] == selected_industry]
                
                st.write(f"{selected_industry}行业上市公司: {len(industry_stocks)}家")
                st.dataframe(industry_stocks)
                
                # 股票代码列表
                stock_codes = industry_stocks['code'].tolist()
                
                # 如果有股票和指数数据，添加行业与市场对比分析选项
                if 'processed_df' in st.session_state and 'index_df' in st.session_state:
                    st.subheader("行业与个股、大盘对比")
                    
                    stock_symbol = st.session_state.stock_symbol
                    stock_data = st.session_state.processed_df
                    index_data = st.session_state.index_df
                    
                    # 提示用户
                    st.write(f"您当前分析的股票({stock_symbol})是否属于{selected_industry}行业:")
                    
                    # 检查当前股票是否属于所选行业
                    is_in_industry = False
                    for code in stock_codes:
                        if code.endswith(stock_symbol):
                            is_in_industry = True
                            break
                    
                    st.write("是" if is_in_industry else "否")
                    
                    # 行业指数走势 (这里仅做示例，实际需要获取更多行业股票数据)
                    st.write("如需获取完整行业指数数据，建议使用专业数据源")

# 机器学习预测标签页
def ml_prediction_tab():
    st.header("机器学习预测")
    
    if 'processed_df' in st.session_state:
        # 创建两列
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("回归预测")
            
            # 选择回归目标
            regression_target = st.selectbox(
                "选择回归预测目标",
                ['Target_Regression_1d', 'Target_Regression_3d', 'Target_Regression_5d']
            )
            
            # 选择回归模型
            regression_model = st.selectbox(
                "选择回归模型",
                ['linear_regression', 'decision_tree', 'random_forest', 'svm']
            )
            
            # 训练回归模型按钮
            if st.button("训练回归模型"):
                with st.spinner("正在训练模型..."):
                    # 创建特征和目标
                    X_train, X_test, y_train, y_test, scaler, feature_cols = create_features(
                        st.session_state.processed_df, regression_target
                    )
                    
                    # 训练模型
                    model = train_model(X_train, y_train, regression_model)
                    
                    # 预测
                    y_pred = model.predict(X_test)
                    
                    # 计算MSE
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    
                    # 保存结果
                    st.session_state.regression_results = {
                        'y_test': y_test,
                        'y_pred': y_pred,
                        'mse': mse,
                        'rmse': rmse,
                        'model': model,
                        'model_name': regression_model,
                        'target': regression_target
                    }
                    
                    # 显示结果
                    st.success(f"模型训练完成，均方误差(MSE): {mse:.6f}, 均方根误差(RMSE): {rmse:.6f}")
                    
                    # 可视化预测结果
                    fig = plot_prediction_results(y_test, y_pred, regression_model)
                    st.pyplot(fig)
                    
                    # 使用大模型解读回归预测结果
                    st.subheader("AI解读预测结果")
                    with st.spinner("正在生成AI分析报告..."):
                        symbol = st.session_state.get('stock_symbol', 'UNKNOWN')
                        interpretation = interpret_prediction_with_llm(
                            symbol, 
                            y_pred, 
                            regression_model, 
                            f"RMSE: {rmse:.6f}"
                        )
                        st.info(interpretation)
                    
                    # 保存预测结果到数据库
                    symbol = st.session_state.get('stock_symbol', 'UNKNOWN')
                    save_prediction_to_db(
                        symbol=symbol,
                        model_name=regression_model,
                        prediction_date=datetime.now().strftime('%Y-%m-%d'),
                        predicted_value=float(y_pred[-1]),
                        actual_value=float(y_test[-1]) if len(y_test) > 0 else None,
                        accuracy=float(rmse),
                        model_params={"features": feature_cols}
                    )
        
        with col2:
            st.subheader("分类预测")
            
            # 选择分类模型
            classification_model = st.selectbox(
                "选择分类模型",
                ['decision_tree_classifier', 'random_forest_classifier', 'svm_classifier']
            )
            
            # 训练分类模型按钮
            if st.button("训练分类模型"):
                with st.spinner("正在训练模型..."):
                    # 创建特征和目标
                    X_train, X_test, y_train, y_test, scaler, feature_cols = create_features(
                        st.session_state.processed_df, 'Target_Classification'
                    )
                    
                    # 训练模型
                    model = train_model(X_train, y_train, classification_model)
                    
                    # 预测
                    y_pred = model.predict(X_test)
                    
                    # 计算准确率
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # 保存结果
                    st.session_state.classification_results = {
                        'y_test': y_test,
                        'y_pred': y_pred,
                        'accuracy': accuracy,
                        'model': model,
                        'model_name': classification_model
                    }
                    
                    # 显示结果
                    st.success(f"模型训练完成，准确率: {accuracy:.4f}")
                    
                    # 显示分类报告
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.write("分类报告：")
                    st.dataframe(report_df)
                    
                    # 可视化分类结果
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(y_test, label='实际值', marker='o', linestyle='--')
                    ax.plot(y_pred, label='预测值', marker='x')
                    ax.set_title(f'{classification_model} 分类结果')
                    ax.set_xlabel('样本')
                    ax.set_ylabel('类别 (0:下跌, 1:上涨)')
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)
                    
                    # 使用大模型解读预测结果
                    st.subheader("AI解读预测结果")
                    with st.spinner("正在生成AI分析报告..."):
                        symbol = st.session_state.get('stock_symbol', 'UNKNOWN')
                        interpretation = interpret_prediction_with_llm(
                            symbol, 
                            y_pred, 
                            classification_model, 
                            f"准确率: {accuracy:.4f}"
                        )
                        st.info(interpretation)
        
        # K-Means聚类
        st.subheader("K-Means聚类分析")
        
        n_clusters = st.slider("选择聚类数量", 2, 5, 3)
        
        if st.button("执行聚类分析"):
            with st.spinner("正在执行聚类分析..."):
                # 执行聚类
                cluster_df, cluster_centers = cluster_stocks(
                    st.session_state.processed_df, n_clusters
                )
                
                st.session_state.cluster_df = cluster_df
                st.session_state.cluster_centers = cluster_centers
                
                # 显示聚类结果
                st.success(f"聚类分析完成，共 {n_clusters} 个聚类")
                
                # 显示每个聚类的样本数
                cluster_counts = cluster_df['Cluster'].value_counts().sort_index()
                st.write("各聚类样本数：")
                st.bar_chart(cluster_counts)
                
                # 可视化聚类结果
                fig = plot_clusters(cluster_df, 'Daily_Return', 'Volatility_5d')
                st.pyplot(fig)
                
                # 分析每个聚类的特征
                st.write("各聚类特征均值：")
                cluster_means = cluster_df.groupby('Cluster').mean()
                st.dataframe(cluster_means[['Daily_Return', 'Volatility_5d', 'RSI']])
    else:
        st.info("请先在数据分析标签页中处理股票数据")

# 可视化展示标签页
def visualization_tab():
    st.header("可视化展示")
    
    # 股票数据可视化
    if 'processed_df' in st.session_state:
        st.subheader("股票数据可视化")
        
        viz_option = st.selectbox(
            "选择可视化类型",
            ['K线图', '价格走势图', '技术指标', '相关性热力图']
        )
        
        if viz_option == 'K线图':
            fig = plot_candlestick(st.session_state.processed_df)
            st.pyplot(fig)
        
        elif viz_option == '价格走势图':
            fig = plot_stock_price(st.session_state.processed_df)
            st.pyplot(fig)
        
        elif viz_option == '技术指标':
            fig = plot_technical_indicators(st.session_state.processed_df)
            st.pyplot(fig)
        
        elif viz_option == '相关性热力图':
            fig = plot_correlation_heatmap(st.session_state.processed_df)
            st.pyplot(fig)
    
    # 预测结果可视化
    if 'regression_results' in st.session_state or 'classification_results' in st.session_state:
        st.subheader("预测结果可视化")
        
        pred_viz_option = st.radio(
            "选择预测结果类型",
            ['回归预测', '分类预测'],
            horizontal=True
        )
        
        if pred_viz_option == '回归预测' and 'regression_results' in st.session_state:
            results = st.session_state.regression_results
            fig = plot_prediction_results(results['y_test'], results['y_pred'], results['model_name'])
            st.pyplot(fig)
            
            st.metric("均方根误差 (RMSE)", f"{results['rmse']:.6f}")
        
        elif pred_viz_option == '分类预测' and 'classification_results' in st.session_state:
            results = st.session_state.classification_results
            
            # 显示混淆矩阵
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(results['y_test'], results['y_pred'])
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
            ax.set_xlabel('预测类别')
            ax.set_ylabel('实际类别')
            ax.set_title('混淆矩阵')
            ax.set_xticklabels(['下跌', '上涨'])
            ax.set_yticklabels(['下跌', '上涨'])
            st.pyplot(fig)
            
            st.metric("准确率", f"{results['accuracy']:.4f}")
    
    # 聚类结果可视化
    if 'cluster_df' in st.session_state:
        st.subheader("聚类结果可视化")
        
        # 选择要在散点图中显示的特征
        col1, col2 = st.columns(2)
        with col1:
            x_feature = st.selectbox("X轴特征", ['Daily_Return', 'Volatility_5d', 'RSI', 'close'])
        with col2:
            y_feature = st.selectbox("Y轴特征", ['Volatility_5d', 'Daily_Return', 'RSI', 'close'])
        
        fig = plot_clusters(st.session_state.cluster_df, x_feature, y_feature)
        st.pyplot(fig)
        
        # 每个聚类的特征分布
        cluster_stats = st.session_state.cluster_df.groupby('Cluster').agg({
            'Daily_Return': ['mean', 'std'],
            'Volatility_5d': ['mean', 'std'],
            'RSI': ['mean', 'std'] if 'RSI' in st.session_state.cluster_df.columns else None
        }).dropna(axis=1)
        
        st.write("聚类统计信息：")
        st.dataframe(cluster_stats)
    
    # 新闻数据可视化
    if 'news_df' in st.session_state:
        st.subheader("新闻数据可视化")
        
        news_viz_option = st.radio(
            "选择新闻可视化类型",
            ['情感分析', '关键词词云'],
            horizontal=True
        )
        
        if news_viz_option == '情感分析':
            fig = plot_news_sentiment(st.session_state.news_df)
            if fig:
                st.pyplot(fig)
            else:
                st.info("无法生成情感分析图，请确保新闻数据包含情感分数和日期")
        
        elif news_viz_option == '关键词词云':
            fig = generate_news_wordcloud(st.session_state.news_df)
            if fig:
                st.pyplot(fig)
            else:
                st.info("无法生成词云，请确保新闻数据包含关键词或标题")
    
    # 指数可视化
    if 'index_df' in st.session_state:
        st.subheader("指数数据可视化")
        
        index_df = st.session_state.index_df
        index_name = st.session_state.index_name
        
        # 计算移动平均线
        if 'MA5' not in index_df.columns:
            index_df['MA5'] = index_df['close'].rolling(window=5).mean()
            index_df['MA20'] = index_df['close'].rolling(window=20).mean()
            index_df['MA60'] = index_df['close'].rolling(window=60).mean()
        
        # 绘制指数走势图
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(index_df.index, index_df['close'], label='收盘价')
        ax.plot(index_df.index, index_df['MA5'], label='5日均线')
        ax.plot(index_df.index, index_df['MA20'], label='20日均线')
        ax.plot(index_df.index, index_df['MA60'], label='60日均线')
        ax.set_title(f'{index_name}走势图')
        ax.set_xlabel('日期')
        ax.set_ylabel('指数点位')
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # 绘制指数K线图
        # 重命名列以匹配mplfinance要求
        index_plot_df = index_df.copy()
        index_plot_df = index_plot_df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        index_plot_df.index.name = 'Date'
        
        fig, axlist = mpf.plot(index_plot_df, type='candle', volume=True, 
                              title=f'{index_name} K线图',
                              ylabel='指数点位',
                              ylabel_lower='成交量',
                              style='charles',
                              returnfig=True,
                              figsize=(10, 8))
        st.pyplot(fig)

# 导出结果标签页
def export_tab():
    st.header("导出结果")
    
    # 选择要导出的数据
    export_option = st.selectbox(
        "选择要导出的数据",
        ['股票原始数据', '处理后的股票数据', '预测结果', '新闻数据', '指数数据', '财务数据']
    )
    
    # 选择导出格式
    export_format = st.radio(
        "选择导出格式",
        ['CSV', 'Excel'],
        horizontal=True
    )
    
    if st.button("导出数据"):
        if export_option == '股票原始数据' and 'stock_df' in st.session_state:
            if export_format == 'CSV':
                st.markdown(get_table_download_link(
                    st.session_state.stock_df, 
                    "stock_data.csv", 
                    "下载CSV文件"
                ), unsafe_allow_html=True)
            else:
                st.markdown(export_to_excel(
                    st.session_state.stock_df, 
                    "stock_data.xlsx"
                ), unsafe_allow_html=True)
                
        elif export_option == '处理后的股票数据' and 'processed_df' in st.session_state:
            if export_format == 'CSV':
                st.markdown(get_table_download_link(
                    st.session_state.processed_df, 
                    "processed_stock_data.csv", 
                    "下载CSV文件"
                ), unsafe_allow_html=True)
            else:
                st.markdown(export_to_excel(
                    st.session_state.processed_df, 
                    "processed_stock_data.xlsx"
                ), unsafe_allow_html=True)
                
        elif export_option == '预测结果':
            if 'regression_results' in st.session_state:
                results = st.session_state.regression_results
                results_df = pd.DataFrame({
                    'actual': results['y_test'],
                    'predicted': results['y_pred']
                })
                
                if export_format == 'CSV':
                    st.markdown(get_table_download_link(
                        results_df, 
                        "prediction_results.csv", 
                        "下载CSV文件"
                    ), unsafe_allow_html=True)
                else:
                    st.markdown(export_to_excel(
                        results_df, 
                        "prediction_results.xlsx"
                    ), unsafe_allow_html=True)
            else:
                st.warning("没有可用的预测结果")
                
        elif export_option == '新闻数据' and 'news_df' in st.session_state:
            if export_format == 'CSV':
                st.markdown(get_table_download_link(
                    st.session_state.news_df, 
                    "news_data.csv", 
                    "下载CSV文件"
                ), unsafe_allow_html=True)
            else:
                st.markdown(export_to_excel(
                    st.session_state.news_df, 
                    "news_data.xlsx"
                ), unsafe_allow_html=True)
        
        elif export_option == '指数数据' and 'index_df' in st.session_state:
            if export_format == 'CSV':
                st.markdown(get_table_download_link(
                    st.session_state.index_df, 
                    "index_data.csv", 
                    "下载CSV文件"
                ), unsafe_allow_html=True)
            else:
                st.markdown(export_to_excel(
                    st.session_state.index_df, 
                    "index_data.xlsx"
                ), unsafe_allow_html=True)
        
        elif export_option == '财务数据' and 'financial_data' in st.session_state:
            financial_data = st.session_state.financial_data
            
            # 选择要导出的财务报表类型
            financial_type = st.selectbox(
                "选择财务报表类型",
                ['performance', 'profit', 'balance', 'cash_flow']
            )
            
            if financial_data[financial_type] is not None and not financial_data[financial_type].empty:
                if export_format == 'CSV':
                    st.markdown(get_table_download_link(
                        financial_data[financial_type], 
                        f"{financial_type}_data.csv", 
                        "下载CSV文件"
                    ), unsafe_allow_html=True)
                else:
                    st.markdown(export_to_excel(
                        financial_data[financial_type], 
                        f"{financial_type}_data.xlsx"
                    ), unsafe_allow_html=True)
            else:
                st.warning(f"没有可用的{financial_type}财务数据")
                
        else:
            st.warning("没有可用的数据")
    
    # 提取数据库中的预测结果
    st.subheader("从数据库获取历史预测")
    
    if st.button("查询历史预测"):
        try:
            conn = create_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
            SELECT symbol, model_name, prediction_date, predicted_value, actual_value, accuracy
            FROM prediction_results
            ORDER BY prediction_date DESC
            LIMIT 50
            """)
            
            result = cursor.fetchall()
            
            if result:
                predictions_df = pd.DataFrame(result, columns=[
                    '股票代码', '模型名称', '预测日期', '预测值', '实际值', '准确率'
                ])
                
                st.write("历史预测结果：")
                st.dataframe(predictions_df)
                
                if export_format == 'CSV':
                    st.markdown(get_table_download_link(
                        predictions_df, 
                        "historical_predictions.csv", 
                        "下载历史预测数据"
                    ), unsafe_allow_html=True)
                else:
                    st.markdown(export_to_excel(
                        predictions_df, 
                        "historical_predictions.xlsx"
                    ), unsafe_allow_html=True)
            else:
                st.info("数据库中没有历史预测数据")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            st.error(f"查询数据库失败: {e}")
    
    # 数据库表查询
    st.subheader("数据库表查询")
    
    table_name = st.selectbox(
        "选择要查询的表",
        ['stock_data', 'news_data', 'prediction_results', 'index_data', 'financial_data']
    )
    
    limit = st.slider("限制返回记录数", 10, 1000, 100)
    
    if st.button("查询数据"):
        try:
            conn = create_connection()
            cursor = conn.cursor()
            
            cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
            
            result = cursor.fetchall()
            
            if result:
                # 获取列名
                cursor.execute(f"SHOW COLUMNS FROM {table_name}")
                columns = [column[0] for column in cursor.fetchall()]
                
                # 创建DataFrame
                df = pd.DataFrame(result, columns=columns)
                
                st.write(f"{table_name} 表数据:")
                st.dataframe(df)
                
                if export_format == 'CSV':
                    st.markdown(get_table_download_link(
                        df, 
                        f"{table_name}_query.csv", 
                        "下载查询结果"
                    ), unsafe_allow_html=True)
                else:
                    st.markdown(export_to_excel(
                        df, 
                        f"{table_name}_query.xlsx"
                    ), unsafe_allow_html=True)
            else:
                st.info(f"{table_name} 表中没有数据")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            st.error(f"查询数据库失败: {e}")

# 主函数
def main():
    st.title("金融数据分析与预测系统")
    
    # 侧边栏 - 系统配置
    st.sidebar.header("系统配置")
    
    # 初始化数据库按钮
    if st.sidebar.button("初始化数据库"):
        init_database()
    
    # 显示系统信息
    st.sidebar.subheader("系统信息")
    st.sidebar.info("""
    版本: 1.0.0
    数据源: BaoStock
    支持: A股市场
    功能: 股票数据分析、技术指标、机器学习预测
    """)
    
    # 显示当前日期
    st.sidebar.subheader("当前日期")
    st.sidebar.write(datetime.now().strftime('%Y-%m-%d'))
    
    # 添加数据清理选项
    st.sidebar.subheader("数据管理")
    if st.sidebar.button("清除所有会话数据"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.sidebar.success("已清除所有会话数据")
    
    # 主界面标签页
    tabs = st.tabs(["数据获取", "数据分析", "机器学习预测", "可视化展示", "导出结果"])
    
    # 数据获取标签页
    with tabs[0]:
        data_acquisition_tab()
    
    # 数据分析标签页
    with tabs[1]:
        data_analysis_tab()
    
    # 机器学习预测标签页
    with tabs[2]:
        ml_prediction_tab()
    
    # 可视化展示标签页
    with tabs[3]:
        visualization_tab()
    
    # 导出结果标签页
    with tabs[4]:
        export_tab()

if __name__ == "__main__":
    main()