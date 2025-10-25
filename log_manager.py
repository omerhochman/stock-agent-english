#!/usr/bin/env python3
"""
Log management tool entry script
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.log_manager import main

if __name__ == '__main__':
    main() 