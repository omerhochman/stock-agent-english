import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import TextIO, Optional


class OutputLogger:
    """
    简化的输出日志器，主要用于控制台输出，减少文件输出。
    """

    def __init__(self, filename: Optional[str] = None, enable_file_logging: bool = False):
        """初始化输出日志器
        
        Args:
            filename: 日志文件名（可选）
            enable_file_logging: 是否启用文件日志记录
        """
        self.terminal = sys.stdout
        self.log_file: Optional[TextIO] = None
        self.filename = filename
        self.closed = False
        self.enable_file_logging = enable_file_logging
        
        # 只有在明确启用文件日志时才创建文件
        if enable_file_logging:
            if filename is None:
                # 创建日志目录
                Path("logs").mkdir(exist_ok=True)
                # 使用简化的文件名，避免每次运行都创建新文件
                filename = "logs/console_output.log"
            
            try:
                # 使用追加模式，避免覆盖之前的日志
                self.log_file = open(filename, "a", encoding='utf-8')
                self.filename = filename
                
                # 在文件中记录会话开始
                session_start = f"\n{'='*50}\n会话开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*50}\n"
                self.log_file.write(session_start)
                self.log_file.flush()
                
                # 仅在终端打印确认消息
                self._write_to_terminal(f"文件日志已启用: {filename}\n")
            except (IOError, PermissionError) as e:
                self._write_to_terminal(f"警告: 无法创建日志文件 {filename}: {e}\n")
                self.enable_file_logging = False

    def write(self, message: str) -> None:
        """写入终端，可选择性写入文件"""
        self._write_to_terminal(message)
        
        # 只有在启用文件日志且消息重要时才写入文件
        if self.enable_file_logging and self._should_log_to_file(message):
            self._write_to_file(message)

    def _should_log_to_file(self, message: str) -> bool:
        """判断是否应该将消息写入文件"""
        # 过滤掉一些不重要的输出
        skip_patterns = [
            '--- Starting Workflow',
            '--- Finished Workflow', 
            '--- API State updated',
            'OutputLogger 已初始化',
            '日志器',
            'HTTP Request',
            'HTTP Response'
        ]
        
        return not any(pattern in message for pattern in skip_patterns)

    def _write_to_terminal(self, message: str) -> None:
        """仅写入终端"""
        self.terminal.write(message)
        self.terminal.flush()
        
    def _write_to_file(self, message: str) -> None:
        """仅写入文件"""
        if not self.closed and self.log_file:
            try:
                # 添加时间戳到文件日志
                timestamp = datetime.now().strftime('%H:%M:%S')
                timestamped_message = f"[{timestamp}] {message}"
                self.log_file.write(timestamped_message)
                self.log_file.flush()
            except (ValueError, IOError) as e:
                self._write_to_terminal(f"警告: 写入日志文件失败: {e}\n")
                
    def _direct_print(self, message):
        """直接打印到终端，绕过任何日志重定向"""
        sys.__stdout__.write(f"{message}\n")
        sys.__stdout__.flush()

    def flush(self) -> None:
        """刷新两个输出"""
        self.terminal.flush()
        if not self.closed and self.log_file:
            try:
                self.log_file.flush()
            except (ValueError, IOError):
                pass

    def close(self) -> None:
        """显式关闭日志文件"""
        if not self.closed and self.log_file:
            try:
                # 在文件中记录会话结束
                session_end = f"\n会话结束: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*50}\n"
                self.log_file.write(session_end)
                self.log_file.close()
                self.closed = True
                if self.enable_file_logging:
                    self._write_to_terminal(f"日志文件 {self.filename} 已关闭。\n")
            except (ValueError, IOError) as e:
                self._write_to_terminal(f"警告: 关闭日志文件失败: {e}\n")

    def __del__(self) -> None:
        """清理工作，关闭日志文件"""
        self.close()


class SimpleConsoleLogger:
    """
    更简单的控制台日志器，不创建任何文件
    """
    
    def __init__(self):
        self.terminal = sys.stdout
        
    def write(self, message: str) -> None:
        """只写入终端"""
        self.terminal.write(message)
        self.terminal.flush()
        
    def flush(self) -> None:
        """刷新输出"""
        self.terminal.flush()
        
    def close(self) -> None:
        """无需关闭操作"""
        pass
        
    def __del__(self) -> None:
        """无需清理操作"""
        pass