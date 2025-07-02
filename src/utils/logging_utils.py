"""
Logging utilities for src.

This module provides consistent logging configuration and utility functions
for the src package.
"""

import logging
import os
import sys
import re
from pathlib import Path


def sanitize_email_for_path(email: str) -> str:
    """
    Sanitize email address for use as a directory name.
    
    Args:
        email: User email address
        
    Returns:
        Sanitized email safe for use as directory name
    """
    # Replace invalid characters with underscores
    sanitized = re.sub(r'[<>:"/\\|?*@.]', '_', email.lower())
    # Remove any consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    return sanitized


def get_user_log_dir(user_email: str) -> Path:
    """
    Get the log directory for a specific user.
    
    Args:
        user_email: User email address
        
    Returns:
        Path to user's log directory
    """
    # Get project root (3 levels up from this file)
    project_root = Path(__file__).parent.parent.parent
    logs_root = project_root / "logs"
    
    # Create sanitized user directory
    sanitized_email = sanitize_email_for_path(user_email)
    user_log_dir = logs_root / sanitized_email
    
    # Ensure directory exists
    user_log_dir.mkdir(parents=True, exist_ok=True)
    
    return user_log_dir


def setup_user_logger(user_email: str, name=None, level=None):
    """
    Configure and return a user-specific logger with consistent formatting.

    Args:
        user_email: User email for creating user-specific log directory
        name: Logger name (defaults to root logger if None)
        level: Logging level (defaults to INFO if None or if env var not set)

    Returns:
        Configured logger instance
    """
    # Get logger level from environment or use default
    if level is None:
        level_name = os.environ.get("SRC_LOG_LEVEL", "INFO")
        level = getattr(logging, level_name.upper(), logging.INFO)

    # Create unique logger name for each user and component
    sanitized_email = sanitize_email_for_path(user_email)
    logger_name = f"user_{sanitized_email}_{name}" if name else f"user_{sanitized_email}"
    
    # Get or create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    # Prevent inheritance from parent loggers to avoid duplicate messages
    logger.propagate = False

    # Always clear existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatters
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # User-specific file handler
    user_log_dir = get_user_log_dir(user_email)
    log_filename = f"{name or 'general'}.log"
    log_file = user_log_dir / log_filename
    
    # Ensure the log file directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Log the setup for debugging
    logger.debug(f"User logger setup for {user_email} - Component: {name} - Log file: {log_file}")

    return logger


def get_user_logger(user_email: str, name=None):
    """
    Get a user-specific logger or create a new one with default settings.

    Args:
        user_email: User email for creating user-specific log directory
        name: Logger name (optional)

    Returns:
        User-specific logger instance
    """
    # Always setup the logger to ensure proper configuration
    return setup_user_logger(user_email, name)


def setup_logger(name=None, level=None, log_file=None):
    """
    Configure and return a logger with consistent formatting.

    Args:
        name: Logger name (defaults to root logger if None)
        level: Logging level (defaults to INFO if None or if env var not set)
        log_file: Optional file path to write logs to

    Returns:
        Configured logger instance
    """
    # Get logger level from environment or use default
    if level is None:
        level_name = os.environ.get("SRC_LOG_LEVEL", "INFO")
        level = getattr(logging, level_name.upper(), logging.INFO)

    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates when called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create formatters
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name=None):
    """
    Get an existing logger or create a new one with default settings.

    Args:
        name: Logger name (optional)

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # If logger doesn't have handlers, set it up
    if not logger.hasHandlers():
        # Check if log directory is set in environment
        log_dir = os.environ.get("SRC_LOG_DIR")
        log_file = None

        if log_dir:
            # Create path based on logger name
            log_filename = f"{name or 'src'}.log"
            log_file = Path(log_dir) / log_filename

        logger = setup_logger(name, log_file=log_file)

    return logger
