"""
CORS Configuration - Plan A Security Fix

SPARC Implementation:
- Specification: Replace wildcard ["*"] with explicit origins
- Architecture: Environment-based configuration
- Security: Proper credentials handling
"""

import os
import logging
from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)


class CORSConfig:
    """
    CORS configuration manager.

    Provides environment-aware CORS settings with secure defaults.
    """

    # Development origins (localhost variants)
    DEVELOPMENT_ORIGINS = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8080",
    ]

    # Default production origins (should be overridden via environment)
    DEFAULT_PRODUCTION_ORIGINS: List[str] = []

    @classmethod
    def get_allowed_origins(cls) -> List[str]:
        """
        Get allowed CORS origins based on environment.

        Priority:
        1. CORS_ORIGINS environment variable (comma-separated)
        2. Environment-specific defaults

        Returns:
            List of allowed origin URLs
        """
        # Check for explicit configuration
        env_origins = os.getenv("CORS_ORIGINS", "")
        if env_origins:
            origins = [o.strip() for o in env_origins.split(",") if o.strip()]
            # Validate origins
            validated = []
            for origin in origins:
                if cls._is_valid_origin(origin):
                    validated.append(origin)
                else:
                    logger.warning(f"Invalid CORS origin ignored: {origin}")
            if validated:
                return validated

        # Environment-based defaults
        environment = os.getenv("ENVIRONMENT", "development").lower()

        if environment == "production":
            if not cls.DEFAULT_PRODUCTION_ORIGINS:
                logger.warning(
                    "CORS_ORIGINS not set in production! "
                    "Set CORS_ORIGINS environment variable to allowed domains."
                )
                return []
            return cls.DEFAULT_PRODUCTION_ORIGINS

        elif environment == "staging":
            staging_origins = os.getenv("STAGING_CORS_ORIGINS", "")
            if staging_origins:
                return [o.strip() for o in staging_origins.split(",")]
            return cls.DEVELOPMENT_ORIGINS

        else:  # development or test
            return cls.DEVELOPMENT_ORIGINS

    @classmethod
    def _is_valid_origin(cls, origin: str) -> bool:
        """Validate origin format."""
        if origin == "*":
            logger.warning("Wildcard CORS origin '*' is not allowed in production")
            return False

        if not origin.startswith(("http://", "https://")):
            return False

        # Block obviously wrong patterns
        if "localhost" in origin and os.getenv("ENVIRONMENT") == "production":
            logger.warning(f"Localhost origin not allowed in production: {origin}")
            return False

        return True

    @classmethod
    def get_cors_config(cls) -> dict:
        """
        Get complete CORS configuration.

        Returns:
            Dict with all CORS middleware parameters
        """
        origins = cls.get_allowed_origins()
        environment = os.getenv("ENVIRONMENT", "development").lower()

        # Allow credentials only if we have specific origins
        allow_credentials = len(origins) > 0 and "*" not in origins

        return {
            "allow_origins": origins,
            "allow_credentials": allow_credentials,
            "allow_methods": cls._get_allowed_methods(),
            "allow_headers": cls._get_allowed_headers(),
            "expose_headers": cls._get_expose_headers(),
            "max_age": 600 if environment == "production" else 60,  # Cache preflight
        }

    @classmethod
    def _get_allowed_methods(cls) -> List[str]:
        """Get allowed HTTP methods."""
        return ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]

    @classmethod
    def _get_allowed_headers(cls) -> List[str]:
        """Get allowed request headers."""
        return [
            "Authorization",
            "Content-Type",
            "Accept",
            "Origin",
            "X-Request-ID",
            "X-Requested-With",
        ]

    @classmethod
    def _get_expose_headers(cls) -> List[str]:
        """Get headers exposed to the browser."""
        return [
            "X-Request-ID",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
            "X-Response-Time",
        ]


def configure_cors(app: FastAPI) -> None:
    """
    Configure CORS middleware with secure settings.

    This replaces the insecure ["*"] wildcard configuration
    with environment-aware explicit origins.

    Args:
        app: FastAPI application instance
    """
    config = CORSConfig.get_cors_config()

    if not config["allow_origins"]:
        logger.error(
            "No CORS origins configured! API will reject cross-origin requests. "
            "Set CORS_ORIGINS environment variable."
        )

    logger.info(f"CORS configured with origins: {config['allow_origins']}")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config["allow_origins"],
        allow_credentials=config["allow_credentials"],
        allow_methods=config["allow_methods"],
        allow_headers=config["allow_headers"],
        expose_headers=config["expose_headers"],
        max_age=config["max_age"],
    )


def get_cors_origins() -> List[str]:
    """
    Utility function to get current CORS origins.

    Use this when you need to check configured origins
    (e.g., for WebSocket origin validation).
    """
    return CORSConfig.get_allowed_origins()
