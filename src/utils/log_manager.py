#!/usr/bin/env python3
"""
Log management tool
Provides log viewing, cleanup, configuration and other functions
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from .logging_config import (
    cleanup_old_logs,
    get_log_stats,
    set_console_level,
    setup_global_logging,
)


def show_log_stats():
    """Display log statistics"""
    stats = get_log_stats()

    if not stats:
        print("No log files found")
        return

    print("=== Log File Statistics ===")
    print(f"Total files: {stats.get('file_count', 0)}")
    print(f"Total size: {stats.get('total_size_mb', 0)} MB")
    print()

    print("File details:")
    for filename, info in stats.items():
        if isinstance(info, dict):
            print(f"  {filename}: {info['size_mb']} MB (Modified: {info['modified']})")


def view_log(filename: Optional[str] = None, lines: int = 50):
    """View log file content"""
    log_dir = Path(__file__).parent.parent.parent / "logs"

    if filename is None:
        # Display the latest log file
        log_files = list(log_dir.glob("*.log*"))
        if not log_files:
            print("No log files found")
            return

        # Sort by modification time, get the latest
        latest_file = max(log_files, key=lambda f: f.stat().st_mtime)
        filename = latest_file.name

    log_file = log_dir / filename

    if not log_file.exists():
        print(f"Log file does not exist: {log_file}")
        return

    print(f"=== Viewing log file: {filename} (last {lines} lines) ===")

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            all_lines = f.readlines()

        # Display last N lines
        display_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines

        for line in display_lines:
            print(line.rstrip())

    except Exception as e:
        print(f"Failed to read log file: {e}")


def clean_logs(days: int = 7, confirm: bool = True):
    """Clean up old log files"""
    if confirm:
        response = input(
            f"Are you sure you want to delete log files older than {days} days? (y/N): "
        )
        if response.lower() != "y":
            print("Operation cancelled")
            return

    cleaned_count = cleanup_old_logs(days)
    print(f"Cleaned up {cleaned_count} log files")


def set_log_level(level: str):
    """Set console log level"""
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    level_upper = level.upper()
    if level_upper not in level_map:
        print(f"Invalid log level: {level}")
        print(f"Available levels: {', '.join(level_map.keys())}")
        return

    set_console_level(level_map[level_upper])
    print(f"Console log level set to: {level_upper}")


def list_log_files():
    """List all log files"""
    log_dir = Path(__file__).parent.parent.parent / "logs"

    if not log_dir.exists():
        print("Log directory does not exist")
        return

    log_files = list(log_dir.glob("*.log*"))

    if not log_files:
        print("No log files found")
        return

    print("=== Log File List ===")

    # Sort by modification time
    log_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    for log_file in log_files:
        stat = log_file.stat()
        size_mb = round(stat.st_size / (1024 * 1024), 2)
        modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {log_file.name} ({size_mb} MB, {modified})")


def main():
    """Command line entry point"""
    parser = argparse.ArgumentParser(description="Log management tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # stats command
    subparsers.add_parser("stats", help="Display log statistics")

    # view command
    view_parser = subparsers.add_parser("view", help="View log file")
    view_parser.add_argument("--file", "-f", help="Specify log file name")
    view_parser.add_argument(
        "--lines",
        "-n",
        type=int,
        default=50,
        help="Number of lines to display (default: 50)",
    )

    # clean command
    clean_parser = subparsers.add_parser("clean", help="Clean up old log files")
    clean_parser.add_argument(
        "--days",
        "-d",
        type=int,
        default=7,
        help="Clean files older than how many days (default: 7)",
    )
    clean_parser.add_argument(
        "--yes", "-y", action="store_true", help="Do not ask for confirmation"
    )

    # level command
    level_parser = subparsers.add_parser("level", help="Set console log level")
    level_parser.add_argument(
        "level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level",
    )

    # list command
    subparsers.add_parser("list", help="List all log files")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize logging system
    setup_global_logging()

    if args.command == "stats":
        show_log_stats()
    elif args.command == "view":
        view_log(args.file, args.lines)
    elif args.command == "clean":
        clean_logs(args.days, not args.yes)
    elif args.command == "level":
        set_log_level(args.level)
    elif args.command == "list":
        list_log_files()


if __name__ == "__main__":
    main()
