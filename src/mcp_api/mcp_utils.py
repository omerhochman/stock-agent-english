import baostock as bs
import os
import sys
import logging
from contextlib import contextmanager
from .data_source_interface import LoginError

# --- 日志设置 ---
def setup_logging(level=logging.INFO):
    """配置应用程序的基本日志记录。"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # 如果依赖项的日志过于冗长，可以选择将其静音
    # logging.getLogger("mcp").setLevel(logging.WARNING)

# 为此模块获取一个日志记录器实例（可选，但是良好实践）
logger = logging.getLogger(__name__)

# --- Baostock上下文管理器 ---
@contextmanager
def baostock_login_context():
    """处理Baostock登录和登出的上下文管理器，抑制标准输出消息。"""
    # 重定向标准输出以抑制登录/登出消息
    original_stdout_fd = sys.stdout.fileno()
    saved_stdout_fd = os.dup(original_stdout_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)

    os.dup2(devnull_fd, original_stdout_fd)
    os.close(devnull_fd)

    logger.debug("尝试Baostock登录...")
    lg = bs.login()
    logger.debug(f"登录结果: code={lg.error_code}, msg={lg.error_msg}")

    # 恢复标准输出
    os.dup2(saved_stdout_fd, original_stdout_fd)
    os.close(saved_stdout_fd)

    if lg.error_code != '0':
        # 在抛出异常前记录错误
        logger.error(f"Baostock登录失败: {lg.error_msg}")
        raise LoginError(f"Baostock登录失败: {lg.error_msg}")

    logger.info("Baostock登录成功。")
    try:
        yield  # API调用发生在这里
    finally:
        # 再次重定向标准输出以进行登出
        original_stdout_fd = sys.stdout.fileno()
        saved_stdout_fd = os.dup(original_stdout_fd)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)

        os.dup2(devnull_fd, original_stdout_fd)
        os.close(devnull_fd)

        logger.debug("尝试Baostock登出...")
        bs.logout()
        logger.debug("登出完成。")

        # 恢复标准输出
        os.dup2(saved_stdout_fd, original_stdout_fd)
        os.close(saved_stdout_fd)
        logger.info("Baostock登出成功。")