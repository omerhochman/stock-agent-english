import functools
import time

from src.utils.logging_config import setup_logger

logger = setup_logger("retry")


def retry(max_tries=3, delay_seconds=2, backoff=2, exceptions=(Exception,)):
    """
    Retry decorator: automatically retry when function fails

    Args:
        max_tries: Maximum number of retries
        delay_seconds: Initial delay in seconds
        backoff: Delay growth multiplier
        exceptions: Exception types that need retry
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mtries, mdelay = max_tries, delay_seconds
            while mtries > 0:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    logger.warning(
                        f"Function {func.__name__} call failed: {str(e)}, "
                        f"remaining retries: {mtries-1}"
                    )

                    mtries -= 1
                    if mtries == 0:
                        logger.error(
                            f"Function {func.__name__} reached maximum retry count, throwing exception"
                        )
                        raise

                    time.sleep(mdelay)
                    mdelay *= backoff
            return func(*args, **kwargs)

        return wrapper

    return decorator
