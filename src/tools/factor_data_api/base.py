"""
因子数据API基础类和通用函数
"""
import logging

from src.tools.data_source_adapter import DataAPI

# 创建数据API实例
data_api = DataAPI()

def setup_logger(name):
    """设置日志记录器"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

# 设置日志记录
logger = setup_logger('factor_data_api')