#!/usr/bin/env python3
"""
日志管理工具
提供日志查看、清理、配置等功能
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from .logging_config import (
    setup_global_logging, 
    set_console_level, 
    get_log_stats, 
    cleanup_old_logs
)


def show_log_stats():
    """显示日志统计信息"""
    stats = get_log_stats()
    
    if not stats:
        print("没有找到日志文件")
        return
    
    print("=== 日志文件统计 ===")
    print(f"文件总数: {stats.get('file_count', 0)}")
    print(f"总大小: {stats.get('total_size_mb', 0)} MB")
    print()
    
    print("文件详情:")
    for filename, info in stats.items():
        if isinstance(info, dict):
            print(f"  {filename}: {info['size_mb']} MB (修改时间: {info['modified']})")


def view_log(filename: Optional[str] = None, lines: int = 50):
    """查看日志文件内容"""
    log_dir = Path(__file__).parent.parent.parent / 'logs'
    
    if filename is None:
        # 显示最新的日志文件
        log_files = list(log_dir.glob('*.log*'))
        if not log_files:
            print("没有找到日志文件")
            return
        
        # 按修改时间排序，获取最新的
        latest_file = max(log_files, key=lambda f: f.stat().st_mtime)
        filename = latest_file.name
    
    log_file = log_dir / filename
    
    if not log_file.exists():
        print(f"日志文件不存在: {log_file}")
        return
    
    print(f"=== 查看日志文件: {filename} (最后 {lines} 行) ===")
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            
        # 显示最后N行
        display_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        for line in display_lines:
            print(line.rstrip())
            
    except Exception as e:
        print(f"读取日志文件失败: {e}")


def clean_logs(days: int = 7, confirm: bool = True):
    """清理旧日志文件"""
    if confirm:
        response = input(f"确定要删除 {days} 天前的日志文件吗? (y/N): ")
        if response.lower() != 'y':
            print("操作已取消")
            return
    
    cleaned_count = cleanup_old_logs(days)
    print(f"已清理 {cleaned_count} 个日志文件")


def set_log_level(level: str):
    """设置控制台日志级别"""
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    level_upper = level.upper()
    if level_upper not in level_map:
        print(f"无效的日志级别: {level}")
        print(f"可用级别: {', '.join(level_map.keys())}")
        return
    
    set_console_level(level_map[level_upper])
    print(f"控制台日志级别已设置为: {level_upper}")


def list_log_files():
    """列出所有日志文件"""
    log_dir = Path(__file__).parent.parent.parent / 'logs'
    
    if not log_dir.exists():
        print("日志目录不存在")
        return
    
    log_files = list(log_dir.glob('*.log*'))
    
    if not log_files:
        print("没有找到日志文件")
        return
    
    print("=== 日志文件列表 ===")
    
    # 按修改时间排序
    log_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    
    for log_file in log_files:
        stat = log_file.stat()
        size_mb = round(stat.st_size / (1024 * 1024), 2)
        modified = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  {log_file.name} ({size_mb} MB, {modified})")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='日志管理工具')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # stats 命令
    subparsers.add_parser('stats', help='显示日志统计信息')
    
    # view 命令
    view_parser = subparsers.add_parser('view', help='查看日志文件')
    view_parser.add_argument('--file', '-f', help='指定日志文件名')
    view_parser.add_argument('--lines', '-n', type=int, default=50, help='显示行数 (默认: 50)')
    
    # clean 命令
    clean_parser = subparsers.add_parser('clean', help='清理旧日志文件')
    clean_parser.add_argument('--days', '-d', type=int, default=7, help='清理多少天前的文件 (默认: 7)')
    clean_parser.add_argument('--yes', '-y', action='store_true', help='不询问确认')
    
    # level 命令
    level_parser = subparsers.add_parser('level', help='设置控制台日志级别')
    level_parser.add_argument('level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                             help='日志级别')
    
    # list 命令
    subparsers.add_parser('list', help='列出所有日志文件')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 初始化日志系统
    setup_global_logging()
    
    if args.command == 'stats':
        show_log_stats()
    elif args.command == 'view':
        view_log(args.file, args.lines)
    elif args.command == 'clean':
        clean_logs(args.days, not args.yes)
    elif args.command == 'level':
        set_log_level(args.level)
    elif args.command == 'list':
        list_log_files()


if __name__ == '__main__':
    main() 