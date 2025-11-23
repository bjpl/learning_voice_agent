"""
Structured Logging Configuration - Week 1 Infrastructure
PATTERN: Centralized structlog configuration with JSON output support
WHY: Structured logs are machine-parseable for monitoring and debugging
"""
import sys
import logging
from typing import Optional
from functools import lru_cache

try:
    import structlog
    from structlog.types import Processor
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False


def _add_log_level(logger: str, method_name: str, event_dict: dict) -> dict:
    """Add log level to event dict for filtering."""
    event_dict["level"] = method_name.upper()
    return event_dict


def _add_timestamp(logger: str, method_name: str, event_dict: dict) -> dict:
    """Add ISO format timestamp."""
    import datetime
    event_dict["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
    return event_dict


@lru_cache(maxsize=1)
def configure_structlog(
    json_output: bool = False,
    log_level: str = "INFO"
) -> None:
    """
    Configure structlog with appropriate processors.

    Args:
        json_output: If True, output JSON format (for production)
        log_level: Minimum log level to output

    PATTERN: Single configuration point for all logging
    WHY: Consistency across all modules
    """
    if not STRUCTLOG_AVAILABLE:
        # Fallback to standard logging if structlog not installed
        return

    # Common processors for all outputs
    shared_processors: list[Processor] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_output:
        # Production: JSON output for log aggregation
        renderer = structlog.processors.JSONRenderer()
    else:
        # Development: Pretty console output
        renderer = structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.plain_traceback
        )

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging to work with structlog
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=renderer,
            foreign_pre_chain=shared_processors,
        )
    )

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(getattr(logging, log_level.upper()))


def get_logger(name: str = "voice_agent") -> "structlog.stdlib.BoundLogger":
    """
    Get a configured structlog logger instance.

    Args:
        name: Logger name for identification

    Returns:
        Configured structlog BoundLogger instance

    PATTERN: Factory function for logger instances
    WHY: Easy to get loggers with consistent configuration

    Usage:
        logger = get_logger("my_module")
        logger.info("event_name", key1="value1", key2=123)
        logger.error("api_failed", error=str(e), endpoint="/api/test")
    """
    if not STRUCTLOG_AVAILABLE:
        # Fallback to standard logging
        return _get_stdlib_logger(name)

    # Ensure structlog is configured
    configure_structlog()

    return structlog.get_logger(name)


def _get_stdlib_logger(name: str) -> logging.Logger:
    """Fallback to standard library logging if structlog unavailable."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger


# Pre-configured loggers for different components
# PATTERN: Module-specific loggers for granular control
logger = get_logger("voice_agent")
api_logger = get_logger("voice_agent.api")
audio_logger = get_logger("voice_agent.audio")
db_logger = get_logger("voice_agent.database")
conversation_logger = get_logger("voice_agent.conversation")
state_logger = get_logger("voice_agent.state")


# Legacy compatibility - setup_logger function
def setup_logger(
    name: str = "voice_agent",
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Legacy function for backwards compatibility.
    Prefer using get_logger() for new code.
    """
    return _get_stdlib_logger(name)
