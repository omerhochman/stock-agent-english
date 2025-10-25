import re, difflib, akshare as ak
from datetime import datetime, timedelta
from .cache import get_cached_data
from src.utils.logging_config import setup_logger

logger = setup_logger('industry_adapter')

def get_industry_by_code(stock_code: str) -> str:
    """
    Search through all industry components one by one, return the industry name the stock belongs to
    Optimized version: uses cache to improve query efficiency
    """
    cache_key = f"industry_by_code_{stock_code}"
    
    def _fetch_industry():
        board_df = ak.stock_board_industry_name_em()          # Industry list
        for ind in board_df["板块名称"]:  # Sector name
            cons = ak.stock_board_industry_cons_em(ind)       # Component stocks
            if stock_code in cons["代码"].values:  # Code
                return ind
        raise ValueError(f"Could not find {stock_code} in 100 industries")
    
    # Use longer cache period since industry classification doesn't change often
    industry = get_cached_data(cache_key, _fetch_industry, ttl_days=30)
    logger.info(f"Get industry for stock {stock_code}: {industry}")
    return industry

def _strip_suffix(name: str) -> str:
    """Remove common suffixes, keep core keywords"""
    return re.sub(r"(行业|Ⅱ|Ⅰ|板块)$", "", name).strip()  # Remove 行业(industry), Ⅱ, Ⅰ, 板块(sector) suffixes

def get_industry_valuation(industry: str) -> tuple[float, float]:
    """
    First try exact match in Shenwan 1/2/3 level tables; if fails, do fuzzy matching
    Returns (PE_TTM, PB)
    Cached version
    """
    cache_key = f"industry_valuation_{industry.replace(' ', '_')}"
    
    def _fetch_valuation():
        fetchers = (
            ak.sw_index_first_info,    # Level 1
            ak.sw_index_second_info,   # Level 2
            ak.sw_index_third_info,    # Level 3
        )
        stripped = _strip_suffix(industry)

        # ---------- 1) Exact match ----------
        for fetch in fetchers:
            df = fetch()
            hit = df[df["行业名称"] == industry]  # Industry name
            if not hit.empty:
                row = hit.iloc[0]
                return float(row["TTM(滚动)市盈率"]), float(row["市净率"])  # TTM P/E ratio, P/B ratio

        # ---------- 2) Fuzzy match ----------
        for fetch in fetchers:
            df = fetch()
            names = df["行业名称"].tolist()  # Industry name
            best = difflib.get_close_matches(stripped, names, n=1, cutoff=0.6)
            if best:
                row = df[df["行业名称"] == best[0]].iloc[0]  # Industry name
                return float(row["TTM(滚动)市盈率"]), float(row["市净率"])  # TTM P/E ratio, P/B ratio

        # ---------- 3) Still no match ----------
        raise ValueError(f"Could not find in Shenwan 1/2/3 level industries: {industry}")
    
    # Industry valuation metrics only need to be updated once per day
    result = get_cached_data(cache_key, _fetch_valuation, ttl_days=1)
    logger.info(f"Get industry {industry} valuation: PE={result[0]}, PB={result[1]}")
    return result

def get_industry_growth(industry: str, window_days: int = 252) -> float:
    """
    Calculate closing price increase for "recent window_days trading days"
    window_days=252 ≈ one year
    Cached version
    """
    cache_key = f"industry_growth_{industry.replace(' ', '_')}_{window_days}days"
    
    def _fetch_growth():
        end = datetime.today().strftime("%Y%m%d")
        beg = (datetime.today() - timedelta(days=window_days*1.4)).strftime("%Y%m%d")
        hist = ak.stock_board_industry_hist_em(
            symbol=industry, start_date=beg, end_date=end,
            period="日k", adjust=""  # Daily K-line
        )
        # Take a long enough interval, then select the last window_days bars for precise window
        hist = hist.tail(window_days)
        return round(hist["收盘"].iloc[-1] / hist["收盘"].iloc[0] - 1, 4)  # Close price
    
    # Historical gains only need to be updated once per hour
    growth = get_cached_data(cache_key, _fetch_growth, ttl_days=0.04)  # About 1 hour
    logger.info(f"Get industry {industry} recent {window_days} day growth: {growth*100:.2f}%")
    return growth

def query_industry_metrics(stock_code: str) -> dict:
    """
    Comprehensive query of stock industry metrics
    """
    try:
        logger.info(f"Start querying industry metrics for stock {stock_code}")
        ind = get_industry_by_code(stock_code)
        pe, pb = get_industry_valuation(ind)
        growth = get_industry_growth(ind)        # Recent one year growth
        
        result = {
            "stock": stock_code,
            "industry": ind,
            "industry_avg_pe": pe,
            "industry_avg_pb": pb,
            "industry_growth": growth
        }
        logger.info(f"Successfully obtained industry metrics: {result}")
        return result
    except Exception as e:
        logger.error(f"Failed to query industry metrics: {e}")
        # Return default values
        return {
            "stock": stock_code,
            "industry": "Unknown Industry",
            "industry_avg_pe": 15,
            "industry_avg_pb": 1.5,
            "industry_growth": 0.05
        }

# Test function
if __name__ == "__main__":
    res = query_industry_metrics("600519")
    print(res)