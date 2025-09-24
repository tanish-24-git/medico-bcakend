import logging
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get DEBUG setting from environment variable with default fallback
DEBUG = os.getenv('DEBUG', 'False').lower() in ('true', '1', 'yes', 'on')

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    
    # Prevent adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Set logger level based on DEBUG setting
    logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG if DEBUG else logging.INFO)
    
    # Create formatter with more detailed format for debug mode
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
