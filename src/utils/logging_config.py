import os
import sys
import logging
import logging.handlers
from typing import Optional
from datetime import datetime
from pathlib import Path


# 预定义的图标
SUCCESS_ICON = "✓"
ERROR_ICON = "✗"
WAIT_ICON = "🔄"

# 全局日志配置
_log_dir = None
_console_level = logging.WARNING  # 默认控制台只显示WARNING及以上
_file_level = logging.DEBUG       # 文件记录所有级别
_initialized = False


class ColoredFormatter(logging.Formatter):
    """带颜色的控制台日志格式化器"""
    
    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 紫色
        'RESET': '\033[0m'      # 重置
    }
    
    def format(self, record):
        # 简化的控制台格式
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        
        # 简化格式：只显示时间、级别和消息
        formatted_time = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        formatted_message = f"{log_color}[{formatted_time}] {record.levelname}: {record.getMessage()}{reset_color}"
        
        return formatted_message


class NoiseFilter(logging.Filter):
    """过滤噪音日志的过滤器"""
    
    NOISE_PATTERNS = [
        'HTTP Request', 'HTTP Response', 'Sending HTTP Request',
        'Request options', 'connect_tcp', 'send_request', 'receive_response',
        'Starting new HTTPS connection', 'Resetting dropped connection',
        'urllib3.connectionpool', 'requests.packages.urllib3',
        'httpx', 'httpcore'
    ]
    
    def filter(self, record):
        message = record.getMessage()
        # 过滤掉包含噪音模式的日志
        return not any(pattern in message for pattern in self.NOISE_PATTERNS)


def setup_global_logging(log_dir: Optional[str] = None, 
                        console_level: int = logging.WARNING,
                        file_level: int = logging.DEBUG,
                        max_bytes: int = 10*1024*1024,  # 10MB
                        backup_count: int = 5):
    """设置全局日志配置
    
    Args:
        log_dir: 日志文件目录
        console_level: 控制台日志级别
        file_level: 文件日志级别
        max_bytes: 单个日志文件最大大小
        backup_count: 保留的备份文件数量
    """
    global _log_dir, _console_level, _file_level, _initialized
    
    if _initialized:
        return
    
    _console_level = console_level
    _file_level = file_level
    
    # 设置日志目录
    if log_dir is None:
        _log_dir = Path(__file__).parent.parent.parent / 'logs'
    else:
        _log_dir = Path(log_dir)
    
    _log_dir.mkdir(exist_ok=True)
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # 清除现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(ColoredFormatter())
    console_handler.addFilter(NoiseFilter())
    root_logger.addHandler(console_handler)
    
    # 创建轮转文件处理器
    log_file = _log_dir / 'application.log'
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, 
        maxBytes=max_bytes, 
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(file_level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # 设置第三方库的日志级别
    third_party_loggers = [
        'urllib3', 'openai', 'httpx', 'httpcore', 
        'asyncio', 'uvicorn', 'requests', 'matplotlib',
        'baostock', 'akshare', 'pandas', 'numpy'
    ]
    
    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    
    _initialized = True
    
    # 记录初始化信息
    logger = logging.getLogger(__name__)
    logger.info(f"日志系统已初始化 - 控制台级别: {logging.getLevelName(console_level)}, "
               f"文件级别: {logging.getLevelName(file_level)}")
    logger.info(f"日志文件: {log_file}")


def setup_logger(name: str, log_dir: Optional[str] = None) -> logging.Logger:
    """设置统一的日志配置（兼容旧接口）

    Args:
        name: logger的名称
        log_dir: 日志文件目录（已废弃，使用全局配置）

    Returns:
        配置好的logger实例
    """
    # 确保全局日志系统已初始化
    if not _initialized:
        setup_global_logging(log_dir)
    
    # 获取或创建 logger
    logger = logging.getLogger(name)
    
    # 不需要额外配置，使用全局配置
    return logger


def set_console_level(level: int):
    """动态设置控制台日志级别"""
    global _console_level
    _console_level = level
    
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
            handler.setLevel(level)
            break
    
    logger = logging.getLogger(__name__)
    logger.info(f"控制台日志级别已设置为: {logging.getLevelName(level)}")


def get_log_stats():
    """获取日志统计信息"""
    if not _log_dir:
        return {}
    
    stats = {}
    log_files = list(_log_dir.glob('*.log*'))
    
    total_size = 0
    for log_file in log_files:
        if log_file.is_file():
            size = log_file.stat().st_size
            total_size += size
            stats[log_file.name] = {
                'size_mb': round(size / (1024 * 1024), 2),
                'modified': datetime.fromtimestamp(log_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            }
    
    stats['total_size_mb'] = round(total_size / (1024 * 1024), 2)
    stats['file_count'] = len(log_files)
    
    return stats


def cleanup_old_logs(days: int = 7):
    """清理指定天数之前的日志文件"""
    if not _log_dir:
        return 0
    
    from datetime import timedelta
    cutoff_time = datetime.now() - timedelta(days=days)
    
    cleaned_count = 0
    for log_file in _log_dir.glob('*.log*'):
        if log_file.is_file():
            file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
            if file_time < cutoff_time:
                try:
                    log_file.unlink()
                    cleaned_count += 1
                except OSError:
                    pass
    
    if cleaned_count > 0:
        logger = logging.getLogger(__name__)
        logger.info(f"已清理 {cleaned_count} 个旧日志文件")
    
    return cleaned_count