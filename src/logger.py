# Logging configuration; unchanged from original
# Sets up structured logging with debug/info levels based on env

import logging
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get DEBUG setting from environment
DEBUG = os.getenv('DEBUG', 'False').lower() in ('true', '1', 'yes', 'on')

def setup_logger(name: str) -> logging.Logger:
    """Set up logger with formatted output"""
    logger = logging.getLogger(name)
    
    # Prevent adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Set level based on DEBUG
    logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG if DEBUG else logging.INFO)
    
    # Detailed format for debug mode
    if DEBUG:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Prevent propagation to avoid duplicate logs
    logger.propagate = False
    
    return logger