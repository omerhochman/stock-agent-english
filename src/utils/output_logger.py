import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, TextIO


class OutputLogger:
    """
    Simplified output logger, mainly for console output, reducing file output.
    """

    def __init__(
        self, filename: Optional[str] = None, enable_file_logging: bool = False
    ):
        """Initialize output logger

        Args:
            filename: Log file name (optional)
            enable_file_logging: Whether to enable file logging
        """
        self.terminal = sys.stdout
        self.log_file: Optional[TextIO] = None
        self.filename = filename
        self.closed = False
        self.enable_file_logging = enable_file_logging

        # Only create file when file logging is explicitly enabled
        if enable_file_logging:
            if filename is None:
                # Create log directory
                Path("logs").mkdir(exist_ok=True)
                # Use simplified filename to avoid creating new files on each run
                filename = "logs/console_output.log"

            try:
                # Use append mode to avoid overwriting previous logs
                self.log_file = open(filename, "a", encoding="utf-8")
                self.filename = filename

                # Record session start in file
                session_start = f"\n{'='*50}\nSession started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*50}\n"
                self.log_file.write(session_start)
                self.log_file.flush()

                # Only print confirmation message to terminal
                self._write_to_terminal(f"File logging enabled: {filename}\n")
            except (IOError, PermissionError) as e:
                self._write_to_terminal(
                    f"Warning: Unable to create log file {filename}: {e}\n"
                )
                self.enable_file_logging = False

    def write(self, message: str) -> None:
        """Write to terminal, optionally write to file"""
        self._write_to_terminal(message)

        # Only write to file when file logging is enabled and message is important
        if self.enable_file_logging and self._should_log_to_file(message):
            self._write_to_file(message)

    def _should_log_to_file(self, message: str) -> bool:
        """Determine whether message should be written to file"""
        # Filter out some unimportant output
        skip_patterns = [
            "--- Starting Workflow",
            "--- Finished Workflow",
            "--- API State updated",
            "OutputLogger initialized",
            "Logger",
            "HTTP Request",
            "HTTP Response",
        ]

        return not any(pattern in message for pattern in skip_patterns)

    def _write_to_terminal(self, message: str) -> None:
        """Write only to terminal"""
        self.terminal.write(message)
        self.terminal.flush()

    def _write_to_file(self, message: str) -> None:
        """Write only to file"""
        if not self.closed and self.log_file:
            try:
                # Add timestamp to file log
                timestamp = datetime.now().strftime("%H:%M:%S")
                timestamped_message = f"[{timestamp}] {message}"
                self.log_file.write(timestamped_message)
                self.log_file.flush()
            except (ValueError, IOError) as e:
                self._write_to_terminal(f"Warning: Failed to write to log file: {e}\n")

    def _direct_print(self, message):
        """Direct print to terminal, bypassing any log redirection"""
        sys.__stdout__.write(f"{message}\n")
        sys.__stdout__.flush()

    def flush(self) -> None:
        """Flush both outputs"""
        self.terminal.flush()
        if not self.closed and self.log_file:
            try:
                self.log_file.flush()
            except (ValueError, IOError):
                pass

    def close(self) -> None:
        """Explicitly close log file"""
        if not self.closed and self.log_file:
            try:
                # Record session end in file
                session_end = f"\nSession ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*50}\n"
                self.log_file.write(session_end)
                self.log_file.close()
                self.closed = True
                if self.enable_file_logging:
                    self._write_to_terminal(
                        f"Log file {self.filename} has been closed.\n"
                    )
            except (ValueError, IOError) as e:
                self._write_to_terminal(f"Warning: Failed to close log file: {e}\n")

    def __del__(self) -> None:
        """Cleanup work, close log file"""
        self.close()


class SimpleConsoleLogger:
    """
    Simpler console logger that doesn't create any files
    """

    def __init__(self):
        self.terminal = sys.stdout

    def write(self, message: str) -> None:
        """Write only to terminal"""
        self.terminal.write(message)
        self.terminal.flush()

    def flush(self) -> None:
        """Flush output"""
        self.terminal.flush()

    def close(self) -> None:
        """No close operation needed"""
        pass

    def __del__(self) -> None:
        """No cleanup operation needed"""
        pass
