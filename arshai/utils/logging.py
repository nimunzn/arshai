"""
Simple logging utilities for Arshai framework with OpenTelemetry integration.

This module provides logging support similar to Taloan project - simple and clean,
with optional OpenTelemetry integration when available.
"""

import logging
import os
import sys
import json
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, Union, List, Set, Protocol, runtime_checkable

# Try to import loguru, fall back to standard logging if not available
try:
    from loguru import logger as loguru_logger
    LOGURU_AVAILABLE = True
    # Remove default logger handlers
    loguru_logger.remove()
except ImportError:
    LOGURU_AVAILABLE = False
    loguru_logger = None

class LogLevel(str, Enum):
    """Enum representing valid log levels"""
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    
    @classmethod
    def validate(cls, level: str) -> bool:
        """Validate if a string is a valid log level"""
        try:
            return level.upper() in cls.__members__
        except (AttributeError, TypeError):
            return False

class LoggingFormat(str, Enum):
    """Enum representing logging formats"""
    DEFAULT = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{extra[name]}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    JSON = "{message}"
    CORRELATION = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{extra[name]}</cyan> | <yellow>trace_id={extra[trace_id]}</yellow> | <yellow>span_id={extra[span_id]}</yellow> | <yellow>correlation_id={extra[correlation_id]}</yellow> - <level>{message}</level>"
    JSON_CORRELATION = "{message}"

@runtime_checkable
class LoggerInterface(Protocol):
    """Protocol defining the logger interface"""
    
    def debug(self, message: Any, **kwargs): ...
    def info(self, message: Any, **kwargs): ...
    def warning(self, message: Any, **kwargs): ...
    def error(self, message: Any, **kwargs): ...
    def critical(self, message: Any, **kwargs): ...
    def bind(self, **kwargs): ...

@runtime_checkable
class HandlerInterface(Protocol):
    """Protocol defining the handler interface"""
    
    def add_handler(self, sink, format: str, level: str, **kwargs): ...
    def remove_handler(self, handler_id): ...

class LoggingConfig:
    """Configuration container for logger settings"""
    
    # Environment variable names
    ENV_LOG_LEVEL = "LOG_LEVEL"
    ENV_LOG_FORMAT = "LOG_FORMAT"
    ENV_LOG_JSON = "LOG_JSON"
    
    # Default values
    DEFAULT_LOG_LEVEL = "INFO"  # String value instead of enum
    DEFAULT_LOG_FORMAT = LoggingFormat.DEFAULT
    DEFAULT_LOG_JSON = False
    
    @classmethod
    def get_log_level(cls) -> str:
        """Get log level from environment or default"""
        return os.getenv(cls.ENV_LOG_LEVEL, cls.DEFAULT_LOG_LEVEL)
    
    @classmethod
    def get_log_format(cls) -> str:
        """Get log format from environment or default"""
        return os.getenv(cls.ENV_LOG_FORMAT, cls.DEFAULT_LOG_FORMAT)
    
    @classmethod
    def is_json_logging(cls) -> bool:
        """Check if JSON logging is enabled"""
        return os.getenv(cls.ENV_LOG_JSON, "false").lower() in ("true", "1", "yes")


def get_logger(name: str):
    """
    Get a logger instance with the given name.
    
    Args:
        name: Name for the logger (typically __name__)
        
    Returns:
        A configured logger instance
    """
    if LOGURU_AVAILABLE:
        # Use loguru with OTEL integration if enabled
        if is_otel_enabled():
            return get_trace_logger(name)
        else:
            return get_simple_logger(name)
    else:
        # Fall back to standard logging
        logger = logging.getLogger(name)
        if not name.startswith("arshai.") and name != "arshai":
            logger.name = f"arshai.{name}"
        return logger


def get_simple_logger(name: str):
    """Get a simple loguru logger"""
    if not name.startswith("arshai.") and name != "arshai":
        name = f"arshai.{name}"
    return loguru_logger.bind(name=name)


def get_trace_logger(name: str, current_span=None):
    """Get a logger with OpenTelemetry trace context"""
    try:
        from opentelemetry import trace
        
        current_span = current_span or trace.get_current_span()
        span_context = current_span.get_span_context()
        
        trace_id = format(span_context.trace_id, '032x')
        span_id = format(span_context.span_id, '016x')
        correlation_id = trace_id if trace_id != '0' * 32 else str(uuid.uuid4())
        
        if not name.startswith("arshai.") and name != "arshai":
            name = f"arshai.{name}"
            
        return loguru_logger.bind(
            name=name,
            trace_id=trace_id,
            span_id=span_id,
            correlation_id=correlation_id
        )
    except (ImportError, AttributeError, Exception):
        # Fall back to simple logger
        return get_simple_logger(name)

class LoggerAdapter:
    """Adapter for loguru logger implementing LoggerInterface"""
    
    def __init__(self, logger_instance=loguru_logger):
        self._logger = logger_instance
        
    def debug(self, message: Any, **kwargs):
        return self._logger.debug(message, **kwargs)
        
    def info(self, message: Any, **kwargs):
        return self._logger.info(message, **kwargs)
        
    def warning(self, message: Any, **kwargs):
        return self._logger.warning(message, **kwargs)
        
    def error(self, message: Any, **kwargs):
        return self._logger.error(message, **kwargs)
        
    def critical(self, message: Any, **kwargs):
        return self._logger.critical(message, **kwargs)
        
    def bind(self, **kwargs):
        return self._logger.bind(**kwargs)

class LoguruHandler:
    """Handler for loguru logger implementing HandlerInterface"""
    
    def __init__(self, logger_instance=loguru_logger):
        self._logger = logger_instance
        self._handlers = {}
        
    def add_handler(self, sink=sys.stdout, format: str = LoggingFormat.DEFAULT, 
                  level: str = LoggingConfig.DEFAULT_LOG_LEVEL, **kwargs):
        handler_id = self._logger.add(
            sink, 
            format=format, 
            level=level, 
            **kwargs
        )
        self._handlers[handler_id] = {"sink": sink, "format": format, "level": level, **kwargs}
        return handler_id
        
    def remove_handler(self, handler_id):
        if handler_id in self._handlers:
            self._logger.remove(handler_id)
            del self._handlers[handler_id]
            return True
        return False
    
    def get_handlers(self):
        return self._handlers

class InterceptHandler(logging.Handler):
    """Handler to intercept standard library logging into loguru"""
    
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        # Bind the logger name to avoid KeyError in format string
        # Use the record's logger name, but ensure it has the arshai prefix
        logger_name = record.name
        if not logger_name.startswith("arshai.") and logger_name != "arshai":
            logger_name = f"arshai.{logger_name}"

        loguru_logger.bind(name=logger_name).opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

class LoggerFactory:
    """Factory for creating logger instances"""
    
    def __init__(self, adapter: LoggerInterface = None, handler: HandlerInterface = None):
        self._adapter = adapter or LoggerAdapter(loguru_logger)
        self._handler = handler or LoguruHandler(loguru_logger)
        
    def create_logger(self, name: str, **context) -> LoggerInterface:
        """Create a simple logger with name"""
        context["name"] = name
        return self._adapter.bind(**context)
    
    def create_correlation_logger(self, name: str, correlation_id: Optional[str] = None, 
                               trace_id: Optional[str] = None, span_id: Optional[str] = None) -> LoggerInterface:
        """Create a logger with correlation ID and trace context"""
        context = {"name": name}
        
        if correlation_id:
            context["correlation_id"] = correlation_id
        else:
            # Generate a correlation ID if not provided
            context["correlation_id"] = str(uuid.uuid4())
            
        if trace_id:
            context["trace_id"] = trace_id
        
        if span_id:
            context["span_id"] = span_id
            
        return self._adapter.bind(**context)
    
    def create_trace_logger(self, name: str, current_span=None) -> LoggerInterface:
        """Create a logger with correlation ID extracted from the current OpenTelemetry span"""
        try:
            from opentelemetry import trace
            current_span = current_span or trace.get_current_span()
            span_context = current_span.get_span_context()
            
            trace_id = format(span_context.trace_id, '032x')
            span_id = format(span_context.span_id, '016x')
            
            # Use trace ID as correlation ID if we have a valid trace
            correlation_id = trace_id if trace_id != '0' * 32 else str(uuid.uuid4())
            
            return self.create_correlation_logger(name, correlation_id, trace_id, span_id)
        except (ImportError, AttributeError):
            # Fall back to regular logger if OpenTelemetry is not available
            # or if current span does not exist
            return self.create_logger(name)

class LoggingService:
    """Central service for managing logging configuration"""
    
    _default_handler_id = None
    _otel_enabled = False  # Default value for OTEL
    
    def __init__(self, handler: HandlerInterface = None):
        self._handler = handler or LoguruHandler(loguru_logger)
    
    @classmethod
    def is_otel_enabled(cls) -> bool:
        """Check if OpenTelemetry is enabled"""
        return cls._otel_enabled
    
    @classmethod
    def set_otel_enabled(cls, enabled: bool):
        """Set OpenTelemetry enabled status"""
        cls._otel_enabled = enabled
        
    def configure(self, json_logging: bool = None, level: str = None):
        """Configure the logging service"""
        json_logging = LoggingConfig.is_json_logging() if json_logging is None else json_logging
        level = LoggingConfig.get_log_level() if level is None else level
        
        # Validate log level
        if isinstance(level, str) and not LogLevel.validate(level):
            valid_levels = ", ".join(LogLevel.__members__.keys())
            raise ValueError(f"Invalid log level: {level}. Valid levels are: {valid_levels}")
        
        # Remove existing default handler if present
        if LoggingService._default_handler_id is not None:
            self._handler.remove_handler(LoggingService._default_handler_id)
        
        # Add appropriate handler based on JSON logging setting
        if json_logging:
            format_str = LoggingFormat.JSON
            LoggingService._default_handler_id = self._handler.add_handler(
                sink=sys.stdout, 
                format=format_str,
                level=level,
                serialize=True,
                backtrace=True,
                diagnose=True
            )
        else:
            format_str = LoggingFormat.DEFAULT
            LoggingService._default_handler_id = self._handler.add_handler(
                sink=sys.stdout, 
                format=format_str,
                level=level,
                backtrace=True,
                diagnose=True
            )
        
        # Configure standard library logging
        self.setup_standard_logging(level)
        
        return LoggingService._default_handler_id
    
    def setup_standard_logging(self, level: str = None):
        """Configure standard library logging to use loguru"""
        level = LoggingConfig.get_log_level() if level is None else level
        
        # Intercept standard library logging
        logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
        
        # Update logger for all existing loggers from standard logging
        for name in logging.root.manager.loggerDict.keys():
            logging.getLogger(name).handlers = [InterceptHandler()]
            # Ensure common packages use the correct log level
            if name in ["uvicorn", "uvicorn.error", "fastapi"]:
                logging.getLogger(name).setLevel(level)
    
    def set_log_level(self, level: Union[str, int]):
        """Set global log level at runtime"""
        # Validate log level if string
        if isinstance(level, str) and not LogLevel.validate(level):
            valid_levels = ", ".join(LogLevel.__members__.keys())
            raise ValueError(f"Invalid log level: {level}. Valid levels are: {valid_levels}")
        
        # Reconfigure with new log level
        self.configure(level=level)
        
        # Also update standard library loggers
        for name in logging.root.manager.loggerDict.keys():
            logging.getLogger(name).setLevel(level)


def is_otel_enabled() -> bool:
    """Check if OpenTelemetry is enabled"""
    return os.getenv("OTEL_ENABLED", "false").lower() in ("true", "1", "yes")


# Legacy function for backward compatibility
def configure_logging(level: str = None, **kwargs):
    """
    Configure logging for the application (legacy compatibility).
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        **kwargs: Other arguments (ignored for compatibility)
    """
    _logging_service.configure(level=level)

class StructuredLogEvent:
    """Base class for structured log events"""
    
    def __init__(self, event_type: str, message: str):
        self.event_type = event_type
        self.message = message
        self.data = {
            "event": event_type,
            "message": message
        }
    
    def add_data(self, **kwargs):
        """Add data to the event"""
        self.data.update(kwargs)
        return self
    
    def get_data(self):
        """Get event data"""
        return self.data
    
    def log(self, logger_instance, level: str = "info"):
        """Log the event with the specified level"""
        if not hasattr(logger_instance, level):
            valid_levels = "debug, info, warning, error, critical, exception"
            raise ValueError(f"Invalid log level: {level}. Valid levels are: {valid_levels}")
        
        log_func = getattr(logger_instance, level)
        if LoggingConfig.is_json_logging():
            log_func(self.data)
        else:
            log_func(f"{self.event_type}: {self.message} | data={json.dumps(self.data)}")

class ConnectionEvent(StructuredLogEvent):
    """Event for connection/disconnection logs"""
    
    def __init__(self, state: str, connection_type: str, entity_id: str, message: str):
        super().__init__(f"{connection_type}_{state}", message)
        self.add_data(
            state=state,
            connection_type=connection_type,
            entity_id=entity_id
        )

class ApiEvent(StructuredLogEvent):
    """Event for API call logs"""
    
    def __init__(self, direction: str, endpoint: str, message: str, status: Optional[int] = None):
        super().__init__("api", message)
        self.add_data(
            direction=direction,
            endpoint=endpoint
        )
        
        if status:
            self.add_data(status=status)
    
    def set_duration(self, duration_ms: float):
        """Set the duration of the API call"""
        self.add_data(duration_ms=duration_ms)
        return self
    
    def log(self, logger_instance):
        """Log API event with appropriate level based on status"""
        status = self.data.get("status")
        level = "debug" if status and 200 <= status < 300 else "info"
        super().log(logger_instance, level)

class ErrorEvent(StructuredLogEvent):
    """Event for error logs"""
    
    def __init__(self, error_type: str, message: str, exception: Optional[Exception] = None):
        super().__init__("error", message)
        self.add_data(error_type=error_type)
        
        if exception:
            self.add_data(
                exception=str(exception),
                exception_type=type(exception).__name__
            )
    
    def log(self, logger_instance):
        """Log error event with error level"""
        super().log(logger_instance, "error")

class MetricEvent(StructuredLogEvent):
    """Event for metric logs"""
    
    def __init__(self, metric_name: str, value: Union[int, float], unit: str = ""):
        super().__init__("metric", f"{metric_name}={value}{unit}")
        self.add_data(
            metric_name=metric_name,
            value=value
        )
        
        if unit:
            self.add_data(unit=unit)

# Initialize singleton instances
_logger_factory = LoggerFactory()
_logging_service = LoggingService()

# Configure logging on module import
_logging_service.configure()

# Backward compatible functions
def get_logger(name: str):
    """
    Get a named logger instance with proper context
    
    Args:
        name: The logger name, typically __name__ from calling module
        
    Returns:
        A loguru logger instance bound with context data
    """
    return _logger_factory.create_logger(name)

def get_logger_with_correlation(name: str, correlation_id: Optional[str] = None, 
                             trace_id: Optional[str] = None, span_id: Optional[str] = None):
    """
    Get a logger with correlation ID and trace context bound to the context.
    This is ideal for distributed tracing scenarios where logs need to be
    correlated across services.
    
    Args:
        name: The logger name
        correlation_id: Optional correlation ID for distributed tracing
        trace_id: Optional OpenTelemetry trace ID
        span_id: Optional OpenTelemetry span ID
        
    Returns:
        A loguru logger with context bound
    """
    return _logger_factory.create_correlation_logger(name, correlation_id, trace_id, span_id)

def get_correlation_logger_from_trace(name: str, current_span=None):
    """
    Create a logger with correlation ID extracted from the current OpenTelemetry span.
    If no current span exists, creates a regular logger.
    
    Args:
        name: The logger name
        current_span: Optional current span to extract context from
        
    Returns:
        A loguru logger with trace context bound
    """
    return _logger_factory.create_trace_logger(name, current_span)

def setup_logging():
    """
    Configure loguru as the primary logging tool,
    intercept standard library logging
    """
    _logging_service.setup_standard_logging()

def set_log_level(level: Union[str, int]):
    """
    Set the global logging level at runtime
    
    Args:
        level: The log level to set (e.g., "DEBUG", "INFO", "WARNING", "ERROR")
    """
    _logging_service.set_log_level(level)

def log_event(logger_instance, level: str, event_type: str, message: str, data: Optional[Dict[str, Any]] = None):
    """
    Log an event with consistent structure and optional context data
    
    Args:
        logger_instance: The logger instance to use
        level: The log level (debug, info, warning, error, critical)
        event_type: Type of event (e.g., 'connection', 'redis', 'api_call')
        message: Human-readable message
        data: Optional dictionary with additional context data
    """
    event = StructuredLogEvent(event_type, message)
    if data:
        event.add_data(**data)
    event.log(logger_instance, level)

def log_connection(logger_instance, state: str, connection_type: str, 
                 entity_id: str, message: str, extra_data: Optional[Dict[str, Any]] = None):
    """
    Log connection/disconnection events consistently
    
    Args:
        logger_instance: The logger instance to use
        state: Connection state ('connected', 'disconnected', etc.)
        connection_type: Type of connection ('sse', 'websocket', 'redis', etc.)
        entity_id: ID of the connected entity (session ID, user ID, etc.)
        message: Human-readable message
        extra_data: Optional dictionary with additional context data
    """
    event = ConnectionEvent(state, connection_type, entity_id, message)
    if extra_data:
        event.add_data(**extra_data)
    event.log(logger_instance, "info")

def log_api(logger_instance, direction: str, endpoint: str, 
          status: Optional[int] = None, message: str = "", 
          extra_data: Optional[Dict[str, Any]] = None,
          duration_ms: Optional[float] = None):
    """
    Log API calls consistently
    
    Args:
        logger_instance: The logger instance to use
        direction: API direction ('request', 'response')
        endpoint: API endpoint path
        status: Optional HTTP status code (for responses)
        message: Human-readable message
        extra_data: Optional dictionary with additional context data
        duration_ms: Optional duration of the API call in milliseconds
    """
    event = ApiEvent(direction, endpoint, message, status)
    
    if duration_ms is not None:
        event.set_duration(duration_ms)
    
    if extra_data:
        event.add_data(**extra_data)
    
    event.log(logger_instance)

def log_error(logger_instance, error_type: str, message: str, 
            exception: Optional[Exception] = None, 
            extra_data: Optional[Dict[str, Any]] = None):
    """
    Log errors consistently
    
    Args:
        logger_instance: The logger instance to use
        error_type: Category of error
        message: Human-readable error message
        exception: Optional exception object
        extra_data: Optional dictionary with additional context data
    """
    event = ErrorEvent(error_type, message, exception)
    
    if extra_data:
        event.add_data(**extra_data)
    
    event.log(logger_instance)

def log_metric(logger_instance, metric_name: str, value: Union[int, float], 
              unit: str = "", context: Optional[Dict[str, Any]] = None):
    """
    Log metrics in a consistent format. This is complementary to OTEL metrics
    but useful for ad-hoc metrics that don't need formal instrumentation.
    
    Args:
        logger_instance: The logger instance to use
        metric_name: Name of the metric
        value: Numeric value of the metric
        unit: Optional unit of the metric
        context: Optional context data for the metric
    """
    event = MetricEvent(metric_name, value, unit)
    
    if context:
        event.add_data(**context)
    
    event.log(logger_instance, "info")