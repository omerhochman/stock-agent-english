"""
Factor data API base classes and common functions
"""
from src.tools.data_source_adapter import DataAPI
from src.utils.logging_config import setup_logger

# Create data API instance
data_api = DataAPI()

# Setup logging
logger = setup_logger('factor_data_api')