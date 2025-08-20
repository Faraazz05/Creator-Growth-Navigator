"""
Creator Growth Navigator - Logging Module

Centralized logging setup with structured output, multiple handlers, and 
context-aware logging for debugging, monitoring, and error tracking.
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json

# Import configuration
from ..config.config import LOGGING_CONFIG, LOGS_DIR, BASE_DIR


class ContextLogger:
    """Enhanced logger with context tracking and structured logging."""
    
    def __init__(self, name: str = "creator_growth_navigator"):
        self.logger = logging.getLogger(name)
        self.context: Dict[str, Any] = {}
    
    def set_context(self, **kwargs) -> None:
        """Set context variables that will be included in all log messages."""
        self.context.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear all context variables."""
        self.context.clear()
    
    def _format_message(self, message: str, extra_context: Optional[Dict] = None) -> str:
        """Format message with context information."""
        full_context = {**self.context}
        if extra_context:
            full_context.update(extra_context)
        
        if full_context:
            context_str = " | ".join([f"{k}={v}" for k, v in full_context.items()])
            return f"{message} | {context_str}"
        return message
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with context."""
        formatted_msg = self._format_message(message, kwargs)
        self.logger.debug(formatted_msg)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message with context."""
        formatted_msg = self._format_message(message, kwargs)
        self.logger.info(formatted_msg)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with context."""
        formatted_msg = self._format_message(message, kwargs)
        self.logger.warning(formatted_msg)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message with context."""
        formatted_msg = self._format_message(message, kwargs)
        self.logger.error(formatted_msg)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message with context."""
        formatted_msg = self._format_message(message, kwargs)
        self.logger.critical(formatted_msg)


class DataProcessingLogger(ContextLogger):
    """Specialized logger for data processing operations."""
    
    def __init__(self):
        super().__init__("creator_growth_navigator.data")
    
    def log_data_load(self, filepath: str, rows: int, columns: int) -> None:
        """Log data loading operation."""
        self.info("Data loaded successfully", 
                 filepath=filepath, rows=rows, columns=columns)
    
    def log_data_validation(self, total_rows: int, valid_rows: int, 
                           issues: Optional[Dict] = None) -> None:
        """Log data validation results."""
        if issues:
            self.warning("Data validation completed with issues",
                        total_rows=total_rows, valid_rows=valid_rows, 
                        issues=issues)
        else:
            self.info("Data validation passed", 
                     total_rows=total_rows, valid_rows=valid_rows)
    
    def log_feature_engineering(self, input_features: int, output_features: int,
                               transformation: str) -> None:
        """Log feature engineering step."""
        self.info("Feature engineering completed",
                 transformation=transformation, 
                 input_features=input_features,
                 output_features=output_features)
    
    def log_data_save(self, filepath: str, rows: int) -> None:
        """Log data saving operation."""
        self.info("Data saved successfully", filepath=filepath, rows=rows)


class ModelLogger(ContextLogger):
    """Specialized logger for model operations."""
    
    def __init__(self):
        super().__init__("creator_growth_navigator.model")
    
    def log_training_start(self, model_type: str, train_samples: int, 
                          features: int) -> None:
        """Log model training start."""
        self.info("Model training started",
                 model_type=model_type, train_samples=train_samples, 
                 features=features)
    
    def log_training_complete(self, model_type: str, duration_seconds: float,
                             performance_metrics: Dict[str, float]) -> None:
        """Log model training completion."""
        self.info("Model training completed",
                 model_type=model_type, duration=f"{duration_seconds:.2f}s",
                 **performance_metrics)
    
    def log_validation_results(self, validation_type: str, 
                              metrics: Dict[str, float]) -> None:
        """Log validation results."""
        self.info("Model validation completed",
                 validation_type=validation_type, **metrics)
    
    def log_prediction(self, n_predictions: int, confidence_level: float) -> None:
        """Log prediction operation."""
        self.info("Predictions generated",
                 n_predictions=n_predictions, 
                 confidence_level=confidence_level)
    
    def log_model_save(self, model_path: str, version: str) -> None:
        """Log model saving."""
        self.info("Model saved successfully", 
                 model_path=model_path, version=version)


class StreamlitLogger(ContextLogger):
    """Specialized logger for Streamlit app operations."""
    
    def __init__(self):
        super().__init__("creator_growth_navigator.streamlit")
    
    def log_user_interaction(self, component: str, action: str, 
                            parameters: Optional[Dict] = None) -> None:
        """Log user interactions in Streamlit app."""
        context = {"component": component, "action": action}
        if parameters:
            context.update(parameters)
        self.info("User interaction", **context)
    
    def log_chart_render(self, chart_type: str, data_points: int) -> None:
        """Log chart rendering."""
        self.debug("Chart rendered", chart_type=chart_type, 
                  data_points=data_points)
    
    def log_error_display(self, error_type: str, error_message: str) -> None:
        """Log errors displayed to user."""
        self.error("Error displayed to user", 
                  error_type=error_type, message=error_message)


class PerformanceLogger:
    """Performance monitoring and timing logger."""
    
    def __init__(self, logger: ContextLogger):
        self.logger = logger
        self.timers: Dict[str, datetime] = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.timers[operation] = datetime.now()
        self.logger.debug(f"Started timing: {operation}")
    
    def end_timer(self, operation: str) -> float:
        """End timing and log duration."""
        if operation not in self.timers:
            self.logger.warning(f"No timer started for: {operation}")
            return 0.0
        
        start_time = self.timers.pop(operation)
        duration = (datetime.now() - start_time).total_seconds()
        
        self.logger.info(f"Operation completed: {operation}", 
                        duration=f"{duration:.3f}s")
        return duration
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Log any remaining timers
        for operation in list(self.timers.keys()):
            self.end_timer(operation)


# =============================================================================
# LOGGING SETUP FUNCTIONS
# =============================================================================

def setup_logging(log_level: str = "INFO", 
                 log_to_file: bool = True,
                 log_to_console: bool = True) -> None:
    """Set up logging configuration for the entire application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
    """
    # Ensure logs directory exists
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create modified config based on parameters
    config = LOGGING_CONFIG.copy()
    
    # Adjust handlers based on parameters
    handlers = []
    if log_to_console:
        handlers.append("console")
    if log_to_file:
        handlers.append("file")
    
    # Update handler levels
    if "console" in config["handlers"]:
        config["handlers"]["console"]["level"] = log_level
    if "file" in config["handlers"]:
        config["handlers"]["file"]["level"] = log_level
    
    # Update root and main logger
    config["root"]["handlers"] = handlers
    config["loggers"]["creator_growth_navigator"]["handlers"] = handlers
    config["loggers"]["creator_growth_navigator"]["level"] = log_level
    
    # Apply configuration
    logging.config.dictConfig(config)


def get_logger(name: str = "creator_growth_navigator") -> ContextLogger:
    """Get a configured logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Configured ContextLogger instance
    """
    return ContextLogger(name)


def get_data_logger() -> DataProcessingLogger:
    """Get specialized logger for data processing."""
    return DataProcessingLogger()


def get_model_logger() -> ModelLogger:
    """Get specialized logger for model operations."""
    return ModelLogger()


def get_streamlit_logger() -> StreamlitLogger:
    """Get specialized logger for Streamlit operations."""
    return StreamlitLogger()


# =============================================================================
# DECORATORS
# =============================================================================

def log_execution_time(logger: Optional[ContextLogger] = None):
    """Decorator to log function execution time."""
    if logger is None:
        logger = get_logger()
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            logger.debug(f"Starting execution: {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"Completed execution: {func.__name__}",
                           duration=f"{duration:.3f}s")
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(f"Error in execution: {func.__name__}",
                           duration=f"{duration:.3f}s", error=str(e))
                raise
        
        return wrapper
    return decorator


def log_function_call(logger: Optional[ContextLogger] = None):
    """Decorator to log function calls with parameters."""
    if logger is None:
        logger = get_logger()
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Log function call (but be careful with sensitive data)
            safe_kwargs = {k: v for k, v in kwargs.items() 
                          if not k.lower().startswith(('password', 'token', 'key'))}
            
            logger.debug(f"Function call: {func.__name__}",
                        args_count=len(args), kwargs=safe_kwargs)
            
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Function completed: {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"Function failed: {func.__name__}", error=str(e))
                raise
        
        return wrapper
    return decorator


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def log_system_info() -> None:
    """Log system information for debugging."""
    logger = get_logger()
    
    logger.info("System Information",
               python_version=sys.version.split()[0],
               platform=sys.platform,
               base_dir=str(BASE_DIR))


def log_configuration(config_dict: Dict[str, Any]) -> None:
    """Log configuration information (sanitized)."""
    logger = get_logger()
    
    # Remove sensitive information
    safe_config = json.loads(json.dumps(config_dict, default=str))
    
    logger.info("Configuration loaded", config_keys=list(safe_config.keys()))


def create_log_context(session_id: str, user_id: str = "anonymous") -> ContextLogger:
    """Create a logger with session context."""
    logger = get_logger()
    logger.set_context(session_id=session_id, user_id=user_id)
    return logger


# =============================================================================
# INITIALIZATION
# =============================================================================

# Initialize logging when module is imported
try:
    setup_logging()
    logger = get_logger()
    logger.info("Logging system initialized successfully")
    log_system_info()
except Exception as e:
    # Fallback to basic logging if setup fails
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    logging.error(f"Failed to initialize advanced logging: {e}")


# Export commonly used loggers
__all__ = [
    'ContextLogger',
    'DataProcessingLogger', 
    'ModelLogger',
    'StreamlitLogger',
    'PerformanceLogger',
    'setup_logging',
    'get_logger',
    'get_data_logger',
    'get_model_logger', 
    'get_streamlit_logger',
    'log_execution_time',
    'log_function_call',
    'log_system_info',
    'create_log_context'
]
