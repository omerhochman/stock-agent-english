import functools
import time
import logging

logger = logging.getLogger(__name__)

def retry(max_tries=3, delay_seconds=2, backoff=2, exceptions=(Exception,)):
    """
    重试装饰器：当函数失败时自动重试
    
    Args:
        max_tries: 最大重试次数
        delay_seconds: 初始延迟秒数
        backoff: 延迟增长倍数
        exceptions: 需要重试的异常类型
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mtries, mdelay = max_tries, delay_seconds
            while mtries > 0:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    logger.warning(f"函数 {func.__name__} 调用失败: {str(e)}, "
                                 f"剩余重试次数: {mtries-1}")
                    
                    mtries -= 1
                    if mtries == 0:
                        logger.error(f"函数 {func.__name__} 达到最大重试次数，抛出异常")
                        raise
                    
                    time.sleep(mdelay)
                    mdelay *= backoff
            return func(*args, **kwargs)
        return wrapper
    return decorator