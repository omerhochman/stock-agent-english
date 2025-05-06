import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO


class OutputLogger:
    """
    重定向 stdout 到控制台和文件的类。
    确保正确清理并避免重复定义。
    """

    def __init__(self, filename: str | None = None):
        """初始化输出日志器"""
        self.terminal = sys.stdout
        if filename is None:
            # 创建日志目录
            Path("logs").mkdir(exist_ok=True)
            # 使用时间戳生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"logs/output_{timestamp}.txt"

        self.log_file: TextIO = open(filename, "w", encoding='utf-8')
        self.filename = filename
        self.closed = False
        
        # 仅在终端打印确认消息（不写入日志文件）
        self._write_to_terminal(f"OutputLogger 已初始化。日志输出到 {filename}\n")

    def write(self, message: str) -> None:
        """写入终端和文件"""
        self._write_to_terminal(message)
        self._write_to_file(message)

    def _write_to_terminal(self, message: str) -> None:
        """仅写入终端"""
        self.terminal.write(message)
        self.terminal.flush()
        
    def _write_to_file(self, message: str) -> None:
        """仅写入文件"""
        if not self.closed and hasattr(self, 'log_file') and self.log_file:
            try:
                self.log_file.write(message)
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
        if not self.closed and hasattr(self, 'log_file') and self.log_file:
            try:
                self.log_file.flush()
            except (ValueError, IOError):
                pass

    def close(self) -> None:
        """显式关闭日志文件"""
        if not self.closed and hasattr(self, 'log_file') and self.log_file:
            try:
                self.log_file.close()
                self.closed = True
                self._write_to_terminal(f"日志文件 {self.filename} 已关闭。\n")
            except (ValueError, IOError) as e:
                self._write_to_terminal(f"警告: 关闭日志文件失败: {e}\n")

    def __del__(self) -> None:
        """清理工作，关闭日志文件"""
        self.close()