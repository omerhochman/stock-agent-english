import re, difflib, akshare as ak
from datetime import datetime, timedelta
from .cache import get_cached_data
from src.utils.logging_config import setup_logger

logger = setup_logger('industry_adapter')

def get_industry_by_code(stock_code: str) -> str:
    """
    在所有行业成份里逐一搜索，返回该股票所属行业名称
    优化版本：使用缓存提高查询效率
    """
    cache_key = f"industry_by_code_{stock_code}"
    
    def _fetch_industry():
        board_df = ak.stock_board_industry_name_em()          # 行业列表
        for ind in board_df["板块名称"]:
            cons = ak.stock_board_industry_cons_em(ind)       # 成份股
            if stock_code in cons["代码"].values:
                return ind
        raise ValueError(f"在 100 个行业里都没找到 {stock_code}")
    
    # 使用较长缓存期，因为行业分类不常变化
    industry = get_cached_data(cache_key, _fetch_industry, ttl_days=30)
    logger.info(f"获取股票 {stock_code} 所属行业: {industry}")
    return industry

def _strip_suffix(name: str) -> str:
    """去掉常见后缀，留核心关键字"""
    return re.sub(r"(行业|Ⅱ|Ⅰ|板块)$", "", name).strip()

def get_industry_valuation(industry: str) -> tuple[float, float]:
    """
    先在申万 1/2/3 级表精确匹配；若失败，做一次模糊匹配
    返回 (PE_TTM, PB)
    带缓存版本
    """
    cache_key = f"industry_valuation_{industry.replace(' ', '_')}"
    
    def _fetch_valuation():
        fetchers = (
            ak.sw_index_first_info,    # 一级
            ak.sw_index_second_info,   # 二级
            ak.sw_index_third_info,    # 三级
        )
        stripped = _strip_suffix(industry)

        # ---------- 1) 精确匹配 ----------
        for fetch in fetchers:
            df = fetch()
            hit = df[df["行业名称"] == industry]
            if not hit.empty:
                row = hit.iloc[0]
                return float(row["TTM(滚动)市盈率"]), float(row["市净率"])

        # ---------- 2) 模糊匹配 ----------
        for fetch in fetchers:
            df = fetch()
            names = df["行业名称"].tolist()
            best = difflib.get_close_matches(stripped, names, n=1, cutoff=0.6)
            if best:
                row = df[df["行业名称"] == best[0]].iloc[0]
                return float(row["TTM(滚动)市盈率"]), float(row["市净率"])

        # ---------- 3) 仍未命中 ----------
        raise ValueError(f"在申万 1/2/3 级行业里都没找到: {industry}")
    
    # 行业估值指标每天更新一次即可
    result = get_cached_data(cache_key, _fetch_valuation, ttl_days=1)
    logger.info(f"获取行业 {industry} 估值: PE={result[0]}, PB={result[1]}")
    return result

def get_industry_growth(industry: str, window_days: int = 252) -> float:
    """
    计算"近 window_days 个交易日"的收盘涨幅
    window_days=252 ≈ 一年
    带缓存版本
    """
    cache_key = f"industry_growth_{industry.replace(' ', '_')}_{window_days}days"
    
    def _fetch_growth():
        end = datetime.today().strftime("%Y%m%d")
        beg = (datetime.today() - timedelta(days=window_days*1.4)).strftime("%Y%m%d")
        hist = ak.stock_board_industry_hist_em(
            symbol=industry, start_date=beg, end_date=end,
            period="日k", adjust=""
        )
        # 取足够长区间后，再选择最后 window_days 根做精确窗口
        hist = hist.tail(window_days)
        return round(hist["收盘"].iloc[-1] / hist["收盘"].iloc[0] - 1, 4)
    
    # 历史涨幅每小时更新一次即可
    growth = get_cached_data(cache_key, _fetch_growth, ttl_days=0.04)  # 约1小时
    logger.info(f"获取行业 {industry} 近 {window_days} 日涨幅: {growth*100:.2f}%")
    return growth

def query_industry_metrics(stock_code: str) -> dict:
    """
    综合查询股票的行业指标
    """
    try:
        logger.info(f"开始查询股票 {stock_code} 的行业指标")
        ind = get_industry_by_code(stock_code)
        pe, pb = get_industry_valuation(ind)
        growth = get_industry_growth(ind)        # 近一年涨幅
        
        result = {
            "stock": stock_code,
            "industry": ind,
            "industry_avg_pe": pe,
            "industry_avg_pb": pb,
            "industry_growth": growth
        }
        logger.info(f"成功获取行业指标: {result}")
        return result
    except Exception as e:
        logger.error(f"查询行业指标失败: {e}")
        # 返回默认值
        return {
            "stock": stock_code,
            "industry": "未知行业",
            "industry_avg_pe": 15,
            "industry_avg_pb": 1.5,
            "industry_growth": 0.05
        }

# 测试函数
if __name__ == "__main__":
    res = query_industry_metrics("600519")
    print(res)