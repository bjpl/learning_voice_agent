"""
Structured Logging Configuration - Week 1 Infrastructure
PATTERN: Centralized structlog configuration with JSON output support + Log Aggregation
WHY: Structured logs are machine-parseable for monitoring and debugging
GOAP: Supports DataDog, CloudWatch, and ELK log aggregation with correlation IDs
"""
import sys
import logging
import uuid
import os
from typing import Optional, Any, Callable
from functools import lru_cache
from contextvars import ContextVar

try:
    import structlog
    from structlog.types import Processor
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

# Correlation ID context for request tracing
# PATTERN: Context variables for async-safe request tracking
correlation_id_var: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


def _add_log_level(logger: str, method_name: str, event_dict: dict) -> dict:
    """Add log level to event dict for filtering."""
    event_dict["level"] = method_name.upper()
    return event_dict


def _add_timestamp(logger: str, method_name: str, event_dict: dict) -> dict:
    """Add ISO format timestamp."""
    import datetime
    event_dict["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
    return event_dict


def _add_correlation_id(logger: str, method_name: str, event_dict: dict) -> dict:
    """
    Add correlation ID to event dict for request tracing.

    PATTERN: Decorator pattern for log enrichment
    WHY: Track requests across distributed systems and async operations
    """
    correlation_id = correlation_id_var.get()
    if correlation_id:
        event_dict["correlation_id"] = correlation_id
    return event_dict


def _add_service_context(logger: str, method_name: str, event_dict: dict) -> dict:
    """
    Add service context metadata for log aggregation.

    PATTERN: Enrichment with environment context
    WHY: Identify service instance in multi-instance deployments
    """
    from app.config import settings

    event_dict["service"] = settings.datadog_service_name
    event_dict["environment"] = settings.environment

    # Add hostname for instance tracking
    import socket
    event_dict["host"] = socket.gethostname()

    return event_dict


def _setup_log_aggregator() -> Optional[logging.Handler]:
    """
    Setup log aggregator handler based on configuration.

    PATTERN: Factory pattern for handler creation
    WHY: Decouple handler creation from configuration logic

    Returns:
        Optional logging.Handler for the configured aggregator
    """
    from app.config import settings

    aggregator = settings.log_aggregator.lower()

    if aggregator == "datadog":
        return _setup_datadog_handler()
    elif aggregator == "cloudwatch":
        return _setup_cloudwatch_handler()
    elif aggregator == "elk":
        return _setup_elk_handler()

    return None


def _setup_datadog_handler() -> Optional[logging.Handler]:
    """
    Setup DataDog log handler.

    PATTERN: Optional dependency with graceful fallback
    WHY: Don't require DataDog in development
    """
    try:
        from app.config import settings

        if not settings.datadog_api_key:
            logging.warning("DataDog API key not configured, skipping DataDog logging")
            return None

        # DataDog integration via HTTP handler
        # PATTERN: HTTP-based log shipping for cloud services
        import logging.handlers

        handler = logging.handlers.HTTPHandler(
            host=f"http-intake.logs.{settings.datadog_site}",
            url=f"/api/v2/logs",
            method="POST",
        )

        # Add DataDog-specific metadata
        # Note: In production, use official datadog library for better integration
        logging.info("DataDog log handler configured")
        return handler

    except ImportError:
        logging.warning("DataDog dependencies not installed, skipping DataDog logging")
        return None
    except Exception as e:
        logging.error(f"Failed to setup DataDog handler: {e}")
        return None


def _setup_cloudwatch_handler() -> Optional[logging.Handler]:
    """
    Setup AWS CloudWatch log handler.

    PATTERN: AWS SDK integration for CloudWatch Logs
    WHY: Native AWS logging for services deployed on AWS
    """
    try:
        from app.config import settings

        # Check if AWS credentials are available
        if not (os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_PROFILE")):
            logging.warning("AWS credentials not configured, skipping CloudWatch logging")
            return None

        try:
            import boto3
            import watchtower
        except ImportError:
            logging.warning("boto3 or watchtower not installed, skipping CloudWatch logging")
            return None

        # Create CloudWatch Logs handler using watchtower
        # PATTERN: Boto3 integration for AWS services
        import socket
        log_stream_name = f"{settings.cloudwatch_log_stream_prefix}-{socket.gethostname()}"

        handler = watchtower.CloudWatchLogHandler(
            log_group_name=settings.cloudwatch_log_group,
            stream_name=log_stream_name,
            use_queues=True,  # Async log shipping
            create_log_group=True,
        )

        logging.info(f"CloudWatch log handler configured: {settings.cloudwatch_log_group}/{log_stream_name}")
        return handler

    except Exception as e:
        logging.error(f"Failed to setup CloudWatch handler: {e}")
        return None


def _setup_elk_handler() -> Optional[logging.Handler]:
    """
    Setup ELK (Elasticsearch) log handler.

    PATTERN: Elasticsearch integration for self-hosted logging
    WHY: Support on-premise and self-hosted ELK stacks
    """
    try:
        from app.config import settings

        if not settings.elk_host:
            logging.warning("ELK host not configured, skipping ELK logging")
            return None

        try:
            from cmreslogging.handlers import CMRESHandler
        except ImportError:
            logging.warning("cmreslogging not installed, skipping ELK logging")
            return None

        # Create Elasticsearch handler
        # PATTERN: CMRESHandler for Elasticsearch log shipping
        auth = None
        if settings.elk_username and settings.elk_password:
            auth = (settings.elk_username, settings.elk_password)

        handler = CMRESHandler(
            hosts=[{
                'host': settings.elk_host,
                'port': settings.elk_port,
                'use_ssl': settings.elk_use_ssl
            }],
            auth_type=CMRESHandler.AuthType.BASIC_AUTH if auth else CMRESHandler.AuthType.NO_AUTH,
            auth_details=auth,
            es_index_name=settings.elk_index_prefix,
            es_additional_fields={
                'service': settings.datadog_service_name,
                'environment': settings.environment
            }
        )

        logging.info(f"ELK log handler configured: {settings.elk_host}:{settings.elk_port}")
        return handler

    except Exception as e:
        logging.error(f"Failed to setup ELK handler: {e}")
        return None


@lru_cache(maxsize=1)
def configure_structlog(
    json_output: bool = False,
    log_level: str = "INFO"
) -> None:
    """
    Configure structlog with appropriate processors and log aggregation.

    Args:
        json_output: If True, output JSON format (for production)
        log_level: Minimum log level to output

    PATTERN: Single configuration point for all logging with optional aggregation
    WHY: Consistency across all modules + production observability
    """
    if not STRUCTLOG_AVAILABLE:
        # Fallback to standard logging if structlog not installed
        return

    # Common processors for all outputs
    # PATTERN: Processor chain with enrichment decorators
    shared_processors: list[Processor] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        _add_correlation_id,  # Add correlation ID for request tracing
        _add_service_context,  # Add service metadata
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
    handlers = []

    # Always add stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=renderer,
            foreign_pre_chain=shared_processors,
        )
    )
    handlers.append(stdout_handler)

    # Add log aggregator handler if configured
    # PATTERN: Multi-destination logging (stdout + aggregator)
    aggregator_handler = _setup_log_aggregator()
    if aggregator_handler:
        # Use JSON format for aggregator
        aggregator_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.processors.JSONRenderer(),
                foreign_pre_chain=shared_processors,
            )
        )
        handlers.append(aggregator_handler)

    root_logger = logging.getLogger()
    root_logger.handlers = handlers
    root_logger.setLevel(getattr(logging, log_level.upper()))


def get_logger(name: str = "voice_agent", **bind_kwargs) -> "structlog.stdlib.BoundLogger":
    """
    Get a configured structlog logger instance.

    Args:
        name: Logger name for identification
        **bind_kwargs: Additional context to bind to this logger instance
                      (e.g., agent_id="abc123", session_id="xyz")

    Returns:
        Configured structlog BoundLogger instance

    PATTERN: Factory function for logger instances with context binding
    WHY: Easy to get loggers with consistent configuration and bound context

    Usage:
        logger = get_logger("my_module")
        logger.info("event_name", key1="value1", key2=123)

        # With bound context (appears in all log messages from this logger)
        logger = get_logger("agent.coordinator", agent_id="abc123")
        logger.info("processing")  # Will include agent_id in output
    """
    if not STRUCTLOG_AVAILABLE:
        # Fallback to standard logging with compat wrapper
        logger = _get_stdlib_logger(name)
        # Bind context if provided
        if bind_kwargs:
            logger = logger.bind(**bind_kwargs)
        return logger

    # Ensure structlog is configured with production settings
    from app.config import settings
    configure_structlog(
        json_output=settings.log_json_format,
        log_level=settings.log_level
    )

    logger = structlog.get_logger(name)

    # Bind additional context if provided
    if bind_kwargs:
        logger = logger.bind(**bind_kwargs)

    return logger


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """
    Set correlation ID for the current context (request/task).

    Args:
        correlation_id: Optional correlation ID, generates UUID if not provided

    Returns:
        The correlation ID that was set

    PATTERN: Context variable management for async-safe tracking
    WHY: Track requests across distributed systems and async operations

    Usage:
        # In FastAPI middleware or request handler
        correlation_id = set_correlation_id(request.headers.get("X-Correlation-ID"))
        logger = get_logger("api")
        logger.info("processing_request")  # Will include correlation_id
    """
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())

    correlation_id_var.set(correlation_id)
    return correlation_id


def get_correlation_id() -> Optional[str]:
    """
    Get the current correlation ID from context.

    Returns:
        Current correlation ID or None if not set

    PATTERN: Context variable accessor
    WHY: Allow retrieval of correlation ID for manual logging
    """
    return correlation_id_var.get()


def clear_correlation_id() -> None:
    """
    Clear the correlation ID from context.

    PATTERN: Context cleanup
    WHY: Prevent correlation ID leakage between requests in async contexts

    Usage:
        # In FastAPI middleware cleanup
        try:
            await call_next(request)
        finally:
            clear_correlation_id()
    """
    correlation_id_var.set(None)


class _StructlogCompatibleLogger:
    """
    Wrapper for stdlib logging.Logger to provide structlog-compatible interface.

    PATTERN: Adapter pattern for API compatibility
    WHY: Allow code to use structlog API even when structlog is unavailable
    """

    def __init__(self, logger: logging.Logger):
        self._logger = logger
        self._bound_context = {}

    def bind(self, **kwargs):
        """Bind context to logger (simulated for stdlib logger)."""
        new_logger = _StructlogCompatibleLogger(self._logger)
        new_logger._bound_context = {**self._bound_context, **kwargs}
        return new_logger

    def _format_message(self, event: str, **kwargs) -> str:
        """Format message with bound context and kwargs."""
        context = {**self._bound_context, **kwargs}
        if context:
            context_str = " ".join(f"{k}={v}" for k, v in context.items())
            return f"{event} {context_str}"
        return event

    def debug(self, event: str, **kwargs):
        """Log debug message with kwargs support."""
        self._logger.debug(self._format_message(event, **kwargs))

    def info(self, event: str, **kwargs):
        """Log info message with kwargs support."""
        self._logger.info(self._format_message(event, **kwargs))

    def warning(self, event: str, **kwargs):
        """Log warning message with kwargs support."""
        self._logger.warning(self._format_message(event, **kwargs))

    def error(self, event: str, **kwargs):
        """Log error message with kwargs support."""
        exc_info = kwargs.pop('exc_info', False)
        self._logger.error(self._format_message(event, **kwargs), exc_info=exc_info)

    def critical(self, event: str, **kwargs):
        """Log critical message with kwargs support."""
        self._logger.critical(self._format_message(event, **kwargs))

    def exception(self, event: str, **kwargs):
        """Log exception with kwargs support."""
        self._logger.exception(self._format_message(event, **kwargs))


def _get_stdlib_logger(name: str) -> _StructlogCompatibleLogger:
    """
    Fallback to standard library logging if structlog unavailable.

    Returns a wrapper that provides structlog-compatible API.
    """
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

    return _StructlogCompatibleLogger(logger)


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
