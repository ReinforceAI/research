# utils/logging_utils.py
import os
import logging
import datetime
from typing import Optional

def setup_detailed_logging(log_dir: str, name: str, level: str = "INFO") -> logging.Logger:
    """
    Set up a detailed logger with both file and console output.
    
    Args:
        log_dir: Directory to store log files
        name: Name for the logger
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for the log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    
    # Map string log level to actual level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    
    # Remove any existing handlers
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(numeric_level)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logger initialized: {name}")
    logger.info(f"Log level: {level}")
    logger.info(f"Log file: {log_file}")
    
    return logger

def log_exception(logger: logging.Logger, message: str, exception: Exception):
    """
    Log an exception with both the message and the traceback.
    
    Args:
        logger: Logger instance
        message: Error message
        exception: The exception that was caught
    """
    import traceback
    logger.error(f"{message}: {str(exception)}")
    logger.error(traceback.format_exc())

def get_function_logger(logger: logging.Logger, function_name: str) -> logging.Logger:
    """
    Get a child logger for a specific function.
    
    Args:
        logger: Parent logger
        function_name: Name of the function
        
    Returns:
        Child logger
    """
    return logger.getChild(function_name)