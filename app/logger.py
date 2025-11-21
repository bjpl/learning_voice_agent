"""
Structured logging configuration for Learning Voice Agent
PATTERN: Centralized logging with structured output using structlog
WHY: Better observability, searchability, and debugging in production

Features:
- Structured JSON logging with contextual information
- Request ID tracking for distributed tracing
- Environment-based configuration (dev vs prod)
- Performance-friendly processors
- Correlation IDs for request tracing
"""
import logging
import sys
import uuid
from typing import Optional, Dict, Any
from contextvars import ContextVar
import structlog
from structlog.types import EventDict, Processor

# Context variable for request tracking
request_context: ContextVar[Dict[str, Any]] = ContextVar('request_context', default={})

def add_request_id(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """
    Add request ID to all log events for distributed tracing
    PATTERN: Context enrichment processor
    WHY: Track requests across the system
    """
    ctx = request_context.get()
    if ctx and 'request_id' in ctx:
        event_dict['request_id'] = ctx['request_id']
    return event_dict


def add_service_info(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """
    Add service metadata to log events
    PATTERN: Static context enrichment
    WHY: Identify log source in distributed systems
    """
    event_dict['service'] = 'learning_voice_agent'
    event_dict['version'] = '1.0.0'
    return event_dict


def drop_color_message_key(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """
    Remove color-related keys from structlog in production
    WHY: Clean JSON output without ANSI codes
    """
    event_dict.pop('color_message', None)
    return event_dict


def configure_structlog(
    log_level: str = "INFO",
    json_output: bool = False,
    show_locals: bool = False
) -> None:
    """
    Configure structlog with appropriate processors for the environment

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: Use JSON formatter (True for prod, False for dev)
        show_locals: Include local variables in exception logs (dev only)

    PATTERN: Environment-based configuration
    WHY: Human-readable logs in dev, structured JSON in production
    """
    # Shared processors for all environments
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        add_request_id,
        add_service_info,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    # Always include exception info formatting
    shared_processors.append(structlog.processors.format_exc_info)

    # Environment-specific processors
    if json_output:
        # Production: JSON output
        shared_processors.extend([
            drop_color_message_key,
            structlog.processors.JSONRenderer()
        ])
    else:
        # Development: Colored console output
        shared_processors.extend([
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(colors=True)
        ])

    # Configure structlog
    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )


def get_logger(
    name: Optional[str] = None,
    **initial_values: Any
) -> structlog.stdlib.BoundLogger:
    """
    Get a configured logger instance with optional initial context

    Args:
        name: Logger name (usually __name__ of the module)
        **initial_values: Initial context to bind to the logger

    Returns:
        Configured structlog logger with bound context

    Example:
        >>> logger = get_logger(__name__, component="audio_pipeline")
        >>> logger.info("processing_audio", duration_ms=123, format="wav")

    PATTERN: Factory with context binding
    WHY: Each logger can have module-specific context
    """
    logger = structlog.get_logger(name)
    if initial_values:
        logger = logger.bind(**initial_values)
    return logger


def set_request_context(
    request_id: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    **extra: Any
) -> Dict[str, Any]:
    """
    Set request context for distributed tracing

    Args:
        request_id: Unique request identifier (auto-generated if not provided)
        session_id: User session identifier
        user_id: User identifier
        **extra: Additional context fields

    Returns:
        The set context dictionary

    Example:
        >>> ctx = set_request_context(session_id="abc123")
        >>> logger.info("handling_request")  # Will include request_id and session_id

    PATTERN: Context propagation
    WHY: Trace requests across async operations
    """
    ctx = {
        'request_id': request_id or str(uuid.uuid4()),
        **({"session_id": session_id} if session_id else {}),
        **({"user_id": user_id} if user_id else {}),
        **extra
    }
    request_context.set(ctx)
    return ctx


def clear_request_context() -> None:
    """
    Clear the current request context
    WHY: Prevent context leakage between requests
    """
    request_context.set({})


def get_request_context() -> Dict[str, Any]:
    """
    Get the current request context
    Returns:
        Current context dictionary
    """
    return request_context.get()


# Initialize structlog based on environment
# Auto-detect environment (could be moved to config)
import os
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
JSON_LOGS = ENVIRONMENT.lower() in ("production", "prod", "staging")
SHOW_LOCALS = ENVIRONMENT.lower() in ("development", "dev")

configure_structlog(
    log_level=LOG_LEVEL,
    json_output=JSON_LOGS,
    show_locals=SHOW_LOCALS
)

# Specialized loggers for different components
api_logger = get_logger("voice_agent.api", component="api")
audio_logger = get_logger("voice_agent.audio", component="audio_pipeline")
db_logger = get_logger("voice_agent.database", component="database")
conversation_logger = get_logger("voice_agent.conversation", component="conversation")
twilio_logger = get_logger("voice_agent.twilio", component="twilio")
state_logger = get_logger("voice_agent.state", component="state_manager")

# Default application logger
logger = get_logger("voice_agent")


# Helper function for backwards compatibility
def setup_logger(
    name: str = "voice_agent",
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> structlog.stdlib.BoundLogger:
    """
    Legacy compatibility function
    DEPRECATED: Use get_logger() instead

    Args:
        name: Logger name
        level: Log level (ignored, use LOG_LEVEL env var)
        format_string: Format string (ignored, structlog handles formatting)

    Returns:
        Structured logger instance
    """
    return get_logger(name)
