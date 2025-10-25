import os
import sys
import logging
import logging.handlers
from typing import Optional
from datetime import datetime
from pathlib import Path


# Predefined icons
SUCCESS_ICON = "✓"
ERROR_ICON = "✗"
WAIT_ICON = "🔄"

# Global logging configuration
_log_dir = None
_console_level = logging.WARNING  # Default console only shows WARNING and above
_file_level = logging.DEBUG       # File records all levels
_initialized = False


class ColoredFormatter(logging.Formatter):
    """Colored console log formatter"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Purple
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Simplified console format
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        
        # Simplified format: only show time, level and message
        formatted_time = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        formatted_message = f"{log_color}[{formatted_time}] {record.levelname}: {record.getMessage()}{reset_color}"
        
        return formatted_message


class NoiseFilter(logging.Filter):
    """Filter for filtering noise logs"""
    
    NOISE_PATTERNS = [
        'HTTP Request', 'HTTP Response', 'Sending HTTP Request',
        'Request options', 'connect_tcp', 'send_request', 'receive_response',
        'Starting new HTTPS connection', 'Resetting dropped connection',
        'urllib3.connectionpool', 'requests.packages.urllib3',
        'httpx', 'httpcore'
    ]
    
    def filter(self, record):
        message = record.getMessage()
        # Filter out logs containing noise patterns
        return not any(pattern in message for pattern in self.NOISE_PATTERNS)


def setup_global_logging(log_dir: Optional[str] = None, 
                        console_level: int = logging.WARNING,
                        file_level: int = logging.DEBUG,
                        max_bytes: int = 10*1024*1024,  # 10MB
                        backup_count: int = 5):
    """Setup global logging configuration
    
    Args:
        log_dir: Log file directory
        console_level: Console log level
        file_level: File log level
        max_bytes: Maximum size of single log file
        backup_count: Number of backup files to keep
    """
    global _log_dir, _console_level, _file_level, _initialized
    
    if _initialized:
        return
    
    _console_level = console_level
    _file_level = file_level
    
    # Setup log directory
    if log_dir is None:
        _log_dir = Path(__file__).parent.parent.parent / 'logs'
    else:
        _log_dir = Path(log_dir)
    
    _log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(ColoredFormatter())
    console_handler.addFilter(NoiseFilter())
    root_logger.addHandler(console_handler)
    
    # Create rotating file handler
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
    
    # Set log levels for third-party libraries
    third_party_loggers = [
        'urllib3', 'openai', 'httpx', 'httpcore', 
        'asyncio', 'uvicorn', 'requests', 'matplotlib',
        'baostock', 'akshare', 'pandas', 'numpy'
    ]
    
    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    
    _initialized = True
    
    # Record initialization information
    logger = logging.getLogger(__name__)
    logger.info(f"Logging system initialized - Console level: {logging.getLevelName(console_level)}, "
               f"File level: {logging.getLevelName(file_level)}")
    logger.info(f"Log file: {log_file}")


def setup_logger(name: str, log_dir: Optional[str] = None) -> logging.Logger:
    """Setup unified logging configuration (compatible with old interface)

    Args:
        name: Logger name
        log_dir: Log file directory (deprecated, uses global configuration)

    Returns:
        Configured logger instance
    """
    # Ensure global logging system is initialized
    if not _initialized:
        setup_global_logging(log_dir)
    
    # Get or create logger
    logger = logging.getLogger(name)
    
    # No additional configuration needed, use global configuration
    return logger


def set_console_level(level: int):
    """Dynamically set console log level"""
    global _console_level
    _console_level = level
    
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
            handler.setLevel(level)
            break
    
    logger = logging.getLogger(__name__)
    logger.info(f"Console log level set to: {logging.getLevelName(level)}")


def get_log_stats():
    """Get log statistics"""
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
    """Clean up log files older than specified days"""
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
        logger.info(f"Cleaned up {cleaned_count} old log files")
    
    return cleaned_count