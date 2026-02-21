"""
Logger module with color coding and timestamps for AI inference workload insights.
"""
from datetime import datetime
from typing import Optional
import sys


class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    # Regular colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Bright colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


class LogLevel:
    """Log level definitions."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Logger:
    """
    A logger class with color coding and timestamps.
    
    Usage:
        from src.logger import logger
        
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")
    """
    
    def __init__(self, name: str = "AI-Inference", use_colors: bool = True):
        """
        Initialize the logger.
        
        Args:
            name: Name of the logger (default: "AI-Inference")
            use_colors: Whether to use color coding (default: True)
        """
        self.name = name
        self.use_colors = use_colors
        self.min_level = LogLevel.DEBUG
        
        # Define color mapping for each log level
        self.level_colors = {
            LogLevel.DEBUG: Colors.CYAN,
            LogLevel.INFO: Colors.BRIGHT_GREEN,
            LogLevel.WARNING: Colors.BRIGHT_YELLOW,
            LogLevel.ERROR: Colors.BRIGHT_RED,
            LogLevel.CRITICAL: Colors.BRIGHT_MAGENTA + Colors.BOLD,
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in a readable format."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    def _format_message(self, level: str, message: str, context: Optional[str] = None) -> str:
        """
        Format a log message with timestamp, level, and optional context.
        
        Args:
            level: Log level
            message: Log message
            context: Optional context (e.g., function name, class name)
        
        Returns:
            Formatted log message
        """
        timestamp = self._get_timestamp()
        context_str = f"[{context}]" if context else ""
        
        if self.use_colors:
            color = self.level_colors.get(level, Colors.WHITE)
            level_str = f"{color}{level:8s}{Colors.RESET}"
            time_str = f"{Colors.BRIGHT_BLUE}{timestamp}{Colors.RESET}"
            name_str = f"{Colors.MAGENTA}{self.name}{Colors.RESET}"
            context_str = f"{Colors.BRIGHT_CYAN}{context_str}{Colors.RESET}" if context else ""
        else:
            level_str = f"{level:8s}"
            time_str = timestamp
            name_str = self.name
        
        return f"{time_str} | {level_str} | {name_str} {context_str} | {message}"
    
    def _log(self, level: str, message: str, context: Optional[str] = None, stream=None):
        """
        Internal logging method.
        
        Args:
            level: Log level
            message: Log message
            context: Optional context
            stream: Output stream (default: sys.stdout for INFO/DEBUG, sys.stderr for others)
        """
        if stream is None:
            stream = sys.stderr if level in [LogLevel.ERROR, LogLevel.CRITICAL] else sys.stdout
        
        formatted_message = self._format_message(level, message, context)
        print(formatted_message, file=stream)
    
    def debug(self, message: str, context: Optional[str] = None):
        """Log a debug message."""
        self._log(LogLevel.DEBUG, message, context)
    
    def info(self, message: str, context: Optional[str] = None):
        """Log an info message."""
        self._log(LogLevel.INFO, message, context)
    
    def warning(self, message: str, context: Optional[str] = None):
        """Log a warning message."""
        self._log(LogLevel.WARNING, message, context)
    
    def error(self, message: str, context: Optional[str] = None):
        """Log an error message."""
        self._log(LogLevel.ERROR, message, context)
    
    def critical(self, message: str, context: Optional[str] = None):
        """Log a critical message."""
        self._log(LogLevel.CRITICAL, message, context)
    
    def success(self, message: str, context: Optional[str] = None):
        """Log a success message (alias for info with special formatting)."""
        if self.use_colors:
            formatted_msg = f"{Colors.BRIGHT_GREEN}✓{Colors.RESET} {message}"
        else:
            formatted_msg = f"✓ {message}"
        self._log(LogLevel.INFO, formatted_msg, context)
    
    def set_name(self, name: str):
        """Change the logger name."""
        self.name = name
    
    def disable_colors(self):
        """Disable color coding."""
        self.use_colors = False
    
    def enable_colors(self):
        """Enable color coding."""
        self.use_colors = True


# Create a default logger instance that can be imported
logger = Logger()


# Convenience function to create a custom logger
def get_logger(name: str, use_colors: bool = True) -> Logger:
    """
    Get a logger instance with a custom name.
    
    Args:
        name: Name of the logger
        use_colors: Whether to use color coding (default: True)
    
    Returns:
        Logger instance
    """
    return Logger(name, use_colors)
