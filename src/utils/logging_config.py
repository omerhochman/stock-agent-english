import os
import sys
import logging
from typing import Optional
from datetime import datetime


# 预定义的图标
SUCCESS_ICON = "✓"
ERROR_ICON = "✗"
WAIT_ICON = "🔄"


def setup_logger(name: str, log_dir: Optional[str] = None) -> logging.Logger:
    """设置统一的日志配置

    Args:
        name: logger的名称
        log_dir: 日志文件目录，如果为None则使用默认的logs目录

    Returns:
        配置好的logger实例
    """
    # 获取或创建 logger
    logger = logging.getLogger(name)
    
    # 如果已经有处理器，先清除所有处理器，确保不会重复添加
    if logger.handlers:
        logger.handlers.clear()
    
    # 设置日志级别
    logger.setLevel(logging.DEBUG)  # logger本身记录DEBUG级别及以上
    logger.propagate = False  # 防止日志消息传播到父级logger

    # 创建控制台处理器
    # 这样可以确保日志输出到原始的标准输出，而不是被重定向后的输出
    console_handler = logging.StreamHandler(sys.__stdout__)
    console_handler.setLevel(logging.INFO)  # 控制台只显示INFO及以上级别

    # 自定义过滤器，过滤掉HTTP请求和其他噪音日志
    class NoiseFilter(logging.Filter):
        def filter(self, record):
            # 过滤掉HTTP请求和响应的详细日志
            message = record.getMessage()
            if any(x in message for x in [
                'HTTP Request', 
                'HTTP Response', 
                'Sending HTTP Request',
                'Request options',
                'connect_tcp',
                'send_request',
                'receive_response'
            ]):
                return False
            return True
    
    # 添加过滤器到控制台处理器
    console_handler.addFilter(NoiseFilter())

    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    # 创建文件处理器
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 使用时间戳来创建唯一的日志文件名，避免文件冲突
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    
    # 尝试创建文件处理器，如果失败则记录错误但不中断程序
    try:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # 文件记录DEBUG级别及以上的日志
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except (IOError, PermissionError) as e:
        # 直接打印到原始stdout，确保错误信息可见
        sys.__stdout__.write(f"警告: 无法创建日志文件 {log_file}: {str(e)}\n")
        sys.__stdout__.flush()

    # 添加控制台处理器
    logger.addHandler(console_handler)
    
    # 记录一条消息，表明日志器已初始化
    logger.info(f"日志器 '{name}' 已初始化，输出到 {log_file}")

    return logger

def setup_global_logging(log_dir: Optional[str] = None):
    """设置全局日志配置，降低所有已知噪音源的日志级别
    
    Args:
        log_dir: 日志文件目录，如果为None则使用默认的logs目录
    """
    # 确保日志目录存在
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 配置根日志处理器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # 添加文件处理器到根日志记录器
    log_file = os.path.join(log_dir, "app.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # 检查根日志记录器是否已有此处理器
    has_file_handler = False
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == file_handler.baseFilename:
            has_file_handler = True
            break
    
    if not has_file_handler:
        root_logger.addHandler(file_handler)
    
    # 设置第三方库的日志级别
    third_party_loggers = [
        'urllib3', 'openai', 'httpx', 'httpcore', 
        'asyncio', 'uvicorn', 'requests', 'matplotlib'
    ]
    
    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)