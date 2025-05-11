import re, difflib, akshare as ak
from datetime import datetime, timedelta

def get_industry_by_code(stock_code: str) -> str:
    """
    在所有行业成份里逐一搜索，返回该股票所属行业名称
    （利用你源码里的 stock_board_industry_cons_em）
    """
    board_df = ak.stock_board_industry_name_em()          # 行业列表
    for ind in board_df["板块名称"]:
        cons = ak.stock_board_industry_cons_em(ind)       # 成份股
        if stock_code in cons["代码"].values:
            return ind
    raise ValueError(f"在 100 个行业里都没找到 {stock_code}")

def _strip_suffix(name: str) -> str:
    """去掉常见后缀，留核心关键字"""
    return re.sub(r"(行业|Ⅱ|Ⅰ|板块)$", "", name).strip()

def get_industry_valuation(industry: str) -> tuple[float, float]:
    """
    先在申万 1/2/3 级表精确匹配；若失败，做一次模糊匹配
    返回 (PE_TTM, PB)
    """
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

def get_industry_growth(industry: str,
                        window_days: int = 252) -> float:
    """
    计算“近 window_days 个交易日”的收盘涨幅
    window_days=252 ≈ 一年
    """
    end = datetime.today().strftime("%Y%m%d")
    beg = (datetime.today() - timedelta(days=window_days*1.4)).strftime("%Y%m%d")
    hist = ak.stock_board_industry_hist_em(
        symbol=industry, start_date=beg, end_date=end,
        period="日k", adjust=""
    )
    # 取足够长区间后，再选择最后 window_days 根做精确窗口
    hist = hist.tail(window_days)
    return round(hist["收盘"].iloc[-1] / hist["收盘"].iloc[0] - 1, 4)

def query_industry_metrics(stock_code: str = "600519") -> dict:
    ind = get_industry_by_code(stock_code)
    pe, pb = get_industry_valuation(ind)
    growth = get_industry_growth(ind)        # 近一年涨幅
    return {"stock": stock_code,
            "industry": ind,
            "pe_ttm": pe,
            "pb": pb,
            "growth_1y": growth}

if __name__ == "__main__":
    res = query_industry_metrics("600054")
    print(res)
