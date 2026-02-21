"""
Demo script to test the logger functionality.
"""
from logger import logger, get_logger
from src.model_init.evaluator import Evaluator


def main():
    """Demonstrate logger functionality."""
    
    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    logger.success("This is a success message")
    
    print("\n" + "="*80 + "\n")
    
    # Test with context
    logger.info("Message with context", context="main")
    logger.debug("Debug with context", context="main")
    
    print("\n" + "="*80 + "\n")
    
    # Test evaluator with logging
    evaluator = Evaluator()
    evaluator.evaluate("dummy_model")
    
    print("\n" + "="*80 + "\n")
    
    # Test custom logger
    custom_logger = get_logger("CustomModule")
    custom_logger.info("This is from a custom logger")
    custom_logger.warning("Custom logger warning")


if __name__ == "__main__":
    main()
