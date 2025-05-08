"""
因子数据API基础类和通用函数
"""
from src.tools.data_source_adapter import DataAPI
from src.utils.logging_config import setup_logger

# 创建数据API实例
data_api = DataAPI()

# 设置日志记录
logger = setup_logger('factor_data_api')