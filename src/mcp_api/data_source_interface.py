from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, List

class DataSourceError(Exception):
    """数据源错误的基础异常类。"""
    pass


class LoginError(DataSourceError):
    """当登录数据源失败时引发的异常。"""
    pass


class NoDataFoundError(DataSourceError):
    """当给定查询没有找到数据时引发的异常。"""
    pass


class FinancialDataSource(ABC):
    """
    定义金融数据源接口的抽象基类。
    此类的实现提供对特定金融数据API的访问
    （例如Baostock、Akshare）。
    """

    @abstractmethod
    def get_historical_k_data(
        self,
        code: str,
        start_date: str,
        end_date: str,
        frequency: str = "d",
        adjust_flag: str = "3",
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        获取给定股票代码的历史K线（OHLCV）数据。

        参数:
            code: 股票代码（例如'sh.600000'，'sz.000001'）。
            start_date: 开始日期，格式为'YYYY-MM-DD'。
            end_date: 结束日期，格式为'YYYY-MM-DD'。
            frequency: 数据频率。常见值取决于底层数据源
                      （例如'd'表示日线，'w'表示周线，'m'表示月线，
                      '5'，'15'，'30'，'60'表示分钟线）。默认为'd'。
            adjust_flag: 历史数据的调整标志。常见值取决于数据源
                        （例如'1'表示前复权，'2'表示后复权，'3'表示不复权）。
                        默认为'3'。
            fields: 要检索的特定字段的可选列表。如果为None，
                   则检索实现定义的默认字段。

        返回:
            包含历史K线数据的pandas DataFrame，其列对应于请求的字段。

        异常:
            LoginError: 如果登录数据源失败。
            NoDataFoundError: 如果查询没有找到数据。
            DataSourceError: 其他与数据源相关的错误。
            ValueError: 如果输入参数无效。
        """
        pass

    @abstractmethod
    def get_stock_basic_info(self, code: str) -> pd.DataFrame:
        """
        获取给定股票代码的基本信息。

        参数:
            code: 股票代码（例如'sh.600000'，'sz.000001'）。

        返回:
            包含基本股票信息的pandas DataFrame。
            结构和列取决于底层数据源。
            通常包含名称、行业、上市日期等信息。

        异常:
            LoginError: 如果登录数据源失败。
            NoDataFoundError: 如果查询没有找到数据。
            DataSourceError: 其他与数据源相关的错误。
            ValueError: 如果输入代码无效。
        """
        pass

    @abstractmethod
    def get_trade_dates(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """获取一定范围内的交易日期信息。"""
        pass

    @abstractmethod
    def get_all_stock(self, date: Optional[str] = None) -> pd.DataFrame:
        """获取特定日期所有股票及其交易状态的列表。"""
        pass

    @abstractmethod
    def get_deposit_rate_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """获取基准存款利率。"""
        pass

    @abstractmethod
    def get_loan_rate_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """获取基准贷款利率。"""
        pass

    @abstractmethod
    def get_required_reserve_ratio_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None, year_type: str = '0') -> pd.DataFrame:
        """获取存款准备金率数据。"""
        pass

    @abstractmethod
    def get_money_supply_data_month(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """获取月度货币供应量数据（M0，M1，M2）。"""
        pass

    @abstractmethod
    def get_money_supply_data_year(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """获取年度货币供应量数据（M0，M1，M2 - 年末余额）。"""
        pass

    @abstractmethod
    def get_shibor_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """获取SHIBOR（上海银行间同业拆放利率）数据。"""
        pass