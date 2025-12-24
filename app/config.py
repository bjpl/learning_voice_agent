"""
Configuration Management
PATTERN: Singleton configuration with environment validation and secrets management
"""
import os
import logging
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

# Load .env file (backward compatibility)
load_dotenv()


# ============================================================================
# SECRETS PROVIDER ABSTRACTION
# PATTERN: Provider pattern for external secrets management
# WHY: Support multiple secrets backends (env, Railway, AWS Secrets Manager)
# ============================================================================

class SecretsProvider(ABC):
    """
    Abstract base class for secrets providers.

    PATTERN: Strategy pattern for secrets retrieval
    WHY: Enable different secrets backends without changing config logic
    """

    @abstractmethod
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieve a secret value by key."""
        pass

    @abstractmethod
    def validate_available(self) -> bool:
        """Check if the provider is available and properly configured."""
        pass


class EnvSecretsProvider(SecretsProvider):
    """
    Environment variable secrets provider.

    PATTERN: Default provider using environment variables
    WHY: Backward compatibility with existing .env setup
    """

    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieve secret from environment variables."""
        return os.getenv(key, default)

    def validate_available(self) -> bool:
        """Environment variables are always available."""
        return True


class RailwaySecretsProvider(SecretsProvider):
    """
    Railway secrets provider.

    PATTERN: Railway-specific environment variable pattern
    WHY: Railway automatically injects secrets as environment variables
    SECURITY: Railway encrypts secrets at rest and in transit
    """

    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Retrieve secret from Railway environment.

        Railway injects secrets as standard environment variables,
        but we validate Railway-specific markers to ensure proper context.
        """
        # Check if running on Railway
        if not self.validate_available():
            logger.warning("RailwaySecretsProvider used outside Railway environment")

        return os.getenv(key, default)

    def validate_available(self) -> bool:
        """Check if running in Railway environment."""
        # Railway sets RAILWAY_ENVIRONMENT variable
        return os.getenv("RAILWAY_ENVIRONMENT") is not None


class AWSSecretsProvider(SecretsProvider):
    """
    AWS Secrets Manager provider.

    PATTERN: Cloud-native secrets management
    WHY: Enterprise-grade secrets management with rotation support
    SECURITY: AWS handles encryption, access control, and audit logging
    """

    def __init__(self):
        self._secrets_cache: Dict[str, str] = {}
        self._client = None

    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Retrieve secret from AWS Secrets Manager.

        PATTERN: Lazy initialization with caching
        WHY: Minimize AWS API calls and startup time
        """
        # Check cache first
        if key in self._secrets_cache:
            return self._secrets_cache[key]

        # Lazy-load boto3 only if needed
        if self._client is None:
            try:
                import boto3
                from botocore.exceptions import ClientError

                self._client = boto3.client('secretsmanager')
                self._client_error = ClientError
            except ImportError:
                logger.error("boto3 not installed. Install with: pip install boto3")
                return default

        # Retrieve from AWS
        try:
            # AWS Secrets Manager key format: app/learning-voice-agent/{key}
            secret_name = f"app/learning-voice-agent/{key}"
            response = self._client.get_secret_value(SecretId=secret_name)

            # AWS stores secrets as JSON or string
            if 'SecretString' in response:
                secret_value = response['SecretString']
                self._secrets_cache[key] = secret_value
                return secret_value
        except self._client_error as e:
            logger.warning(f"Failed to retrieve secret '{key}' from AWS: {e}")
            return default
        except Exception as e:
            logger.error(f"Unexpected error retrieving secret '{key}': {e}")
            return default

        return default

    def validate_available(self) -> bool:
        """Check if AWS credentials are configured."""
        try:
            import boto3
            # Check if AWS credentials are available
            session = boto3.Session()
            credentials = session.get_credentials()
            return credentials is not None
        except (ImportError, Exception):
            return False


class SecretsManager:
    """
    Unified secrets management interface.

    PATTERN: Facade pattern with provider selection
    WHY: Single interface for multiple secrets backends
    """

    def __init__(self, provider_type: str = "env"):
        """
        Initialize secrets manager with specified provider.

        Args:
            provider_type: Provider type (env, railway, aws)
        """
        self.provider, self.provider_type = self._create_provider(provider_type)

        # Validate provider is available
        if not self.provider.validate_available():
            logger.warning(
                f"Secrets provider '{self.provider_type}' not fully available. "
                f"Falling back to environment variables."
            )
            if self.provider_type != "env":
                self.provider = EnvSecretsProvider()
                self.provider_type = "env"

    def _create_provider(self, provider_type: str) -> tuple[SecretsProvider, str]:
        """
        Factory method to create secrets provider.

        PATTERN: Factory pattern
        WHY: Encapsulate provider creation logic

        Returns:
            Tuple of (provider_instance, actual_provider_type)
        """
        providers = {
            "env": EnvSecretsProvider,
            "railway": RailwaySecretsProvider,
            "aws": AWSSecretsProvider,
        }

        provider_class = providers.get(provider_type)
        if provider_class is None:
            logger.warning(
                f"Unknown secrets provider '{provider_type}'. "
                f"Falling back to 'env'. Valid options: {list(providers.keys())}"
            )
            provider_class = EnvSecretsProvider
            provider_type = "env"

        return provider_class(), provider_type

    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieve secret value."""
        return self.provider.get_secret(key, default)


# Global secrets manager instance
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager() -> SecretsManager:
    """
    Get or create global secrets manager instance.

    PATTERN: Lazy singleton
    WHY: Initialize once, use everywhere
    """
    global _secrets_manager
    if _secrets_manager is None:
        provider_type = os.getenv("SECRETS_PROVIDER", "env").lower()
        _secrets_manager = SecretsManager(provider_type)
        logger.info(f"Initialized secrets manager with provider: {_secrets_manager.provider_type}")
    return _secrets_manager

class Settings(BaseSettings):
    # API Keys
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    twilio_account_sid: Optional[str] = Field(None, env="TWILIO_ACCOUNT_SID")
    twilio_auth_token: Optional[str] = Field(None, env="TWILIO_AUTH_TOKEN")
    twilio_phone_number: Optional[str] = Field(None, env="TWILIO_PHONE_NUMBER")

    # Database
    # Supports: sqlite:///path/to/db.db OR postgresql://user:pass@host:port/dbname
    database_url: str = Field("sqlite:///./learning_captures.db", env="DATABASE_URL")

    @property
    def database_type(self) -> str:
        """Extract database type from DATABASE_URL"""
        return self.database_url.split("://")[0].lower()

    @property
    def is_postgres(self) -> bool:
        """Check if using PostgreSQL"""
        return self.database_type in ("postgresql", "postgres")

    # Redis
    redis_url: str = Field("redis://localhost:6379/1", env="REDIS_URL")
    redis_ttl: int = Field(1800, env="REDIS_TTL")  # 30 minutes

    # Server
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")
    cors_origins: list = Field(["*"], env="CORS_ORIGINS")

    # Audio Processing
    whisper_model: str = Field("whisper-1", env="WHISPER_MODEL")
    max_audio_duration: int = Field(60, env="MAX_AUDIO_DURATION")  # seconds

    # Claude Configuration
    claude_model: str = Field("claude-3-5-sonnet-20241022", env="CLAUDE_MODEL")
    claude_max_tokens: int = Field(150, env="CLAUDE_MAX_TOKENS")
    claude_temperature: float = Field(0.7, env="CLAUDE_TEMPERATURE")

    # Session Management
    session_timeout: int = Field(180, env="SESSION_TIMEOUT")  # 3 minutes
    max_context_exchanges: int = Field(5, env="MAX_CONTEXT_EXCHANGES")

    # Week 3: Vector Database Configuration
    chroma_persist_directory: str = Field("./chroma_db", env="CHROMA_PERSIST_DIR")
    embedding_model: str = Field("all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    semantic_search_threshold: float = Field(0.5, env="SEMANTIC_SEARCH_THRESHOLD")
    enable_vector_search: bool = Field(True, env="ENABLE_VECTOR_SEARCH")

    # RuVector Configuration (Phase 1)
    vector_backend: str = Field("auto", env="VECTOR_BACKEND")  # auto, chromadb, ruvector
    ruvector_persist_directory: str = Field("./ruvector_db", env="RUVECTOR_PERSIST_DIR")
    ruvector_enable_learning: bool = Field(True, env="RUVECTOR_ENABLE_LEARNING")
    ruvector_enable_compression: bool = Field(True, env="RUVECTOR_ENABLE_COMPRESSION")
    ruvector_compression_tiers: str = Field(
        "hot:f32:24h,warm:f16:168h,cold:i8",
        env="RUVECTOR_COMPRESSION_TIERS"
    )
    ruvector_gnn_enabled: bool = Field(True, env="RUVECTOR_GNN_ENABLED")
    ruvector_attention_heads: int = Field(8, env="RUVECTOR_ATTENTION_HEADS")

    # Feature Flags for A/B Testing
    vector_ab_test_enabled: bool = Field(False, env="VECTOR_AB_TEST_ENABLED")
    vector_ab_test_ruvector_percentage: int = Field(50, env="VECTOR_AB_TEST_RUVECTOR_PCT")

    # Graph Query Configuration (Phase 2 prep)
    enable_graph_queries: bool = Field(False, env="ENABLE_GRAPH_QUERIES")
    graph_query_timeout_ms: int = Field(500, env="GRAPH_QUERY_TIMEOUT_MS")

    # Week 3: Advanced Prompts Configuration
    use_chain_of_thought: bool = Field(True, env="USE_CHAIN_OF_THOUGHT")
    use_few_shot: bool = Field(True, env="USE_FEW_SHOT")
    prompt_strategy: str = Field("adaptive", env="PROMPT_STRATEGY")  # basic, few_shot, chain_of_thought, adaptive

    # Week 3: Offline/PWA Configuration
    offline_cache_max_entries: int = Field(100, env="OFFLINE_CACHE_MAX_ENTRIES")
    enable_offline_mode: bool = Field(True, env="ENABLE_OFFLINE_MODE")

    # Security Configuration (Plan A)
    jwt_secret_key: str = Field("dev-secret-key-change-in-production", env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field("HS256", env="JWT_ALGORITHM")
    jwt_access_expire_minutes: int = Field(15, env="JWT_ACCESS_EXPIRE_MINUTES")
    jwt_refresh_expire_days: int = Field(7, env="JWT_REFRESH_EXPIRE_DAYS")

    # Rate Limiting
    rate_limit_enabled: bool = Field(True, env="RATE_LIMIT_ENABLED")
    rate_limit_requests_per_minute: int = Field(100, env="RATE_LIMIT_REQUESTS_PER_MINUTE")
    rate_limit_auth_requests_per_minute: int = Field(10, env="RATE_LIMIT_AUTH_REQUESTS_PER_MINUTE")

    # Security Headers Configuration
    security_headers_enabled: bool = Field(True, env="SECURITY_HEADERS_ENABLED")
    security_csp_enabled: bool = Field(True, env="SECURITY_CSP_ENABLED")
    security_csp_report_only: bool = Field(False, env="SECURITY_CSP_REPORT_ONLY")
    security_csp_report_uri: Optional[str] = Field(None, env="SECURITY_CSP_REPORT_URI")
    security_hsts_enabled: bool = Field(True, env="SECURITY_HSTS_ENABLED")
    security_hsts_max_age: int = Field(31536000, env="SECURITY_HSTS_MAX_AGE")
    security_hsts_preload: bool = Field(False, env="SECURITY_HSTS_PRELOAD")
    security_frame_options: str = Field("DENY", env="SECURITY_FRAME_OPTIONS")
    security_referrer_policy: str = Field("strict-origin-when-cross-origin", env="SECURITY_REFERRER_POLICY")
    security_permissions_policy: str = Field(
        "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
        "magnetometer=(), microphone=(self), payment=(), usb=()",
        env="SECURITY_PERMISSIONS_POLICY"
    )

    # WebSocket Security
    websocket_origin_validation: bool = Field(True, env="WEBSOCKET_ORIGIN_VALIDATION")

    # Logging and Observability Configuration
    log_aggregator: str = Field("none", env="LOG_AGGREGATOR")  # none, datadog, cloudwatch, elk
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_json_format: bool = Field(False, env="LOG_JSON_FORMAT")

    # DataDog Configuration
    datadog_api_key: Optional[str] = Field(None, env="DATADOG_API_KEY")
    datadog_app_key: Optional[str] = Field(None, env="DATADOG_APP_KEY")
    datadog_site: str = Field("datadoghq.com", env="DATADOG_SITE")
    datadog_service_name: str = Field("learning-voice-agent", env="DATADOG_SERVICE_NAME")

    # CloudWatch Configuration
    aws_region: str = Field("us-east-1", env="AWS_REGION")
    cloudwatch_log_group: str = Field("/learning-voice-agent", env="CLOUDWATCH_LOG_GROUP")
    cloudwatch_log_stream_prefix: str = Field("app", env="CLOUDWATCH_LOG_STREAM_PREFIX")

    # ELK Configuration
    elk_host: Optional[str] = Field(None, env="ELK_HOST")
    elk_port: int = Field(9200, env="ELK_PORT")
    elk_index_prefix: str = Field("learning-voice-agent", env="ELK_INDEX_PREFIX")
    elk_use_ssl: bool = Field(False, env="ELK_USE_SSL")
    elk_username: Optional[str] = Field(None, env="ELK_USERNAME")
    elk_password: Optional[str] = Field(None, env="ELK_PASSWORD")

    # Environment
    environment: str = Field("development", env="ENVIRONMENT")

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._log_secrets_configuration()
        self._validate_production_settings()

    def _log_secrets_configuration(self) -> None:
        """
        Log secrets provider configuration.

        PATTERN: Transparency logging
        WHY: Make secrets configuration visible for debugging
        """
        secrets_manager = get_secrets_manager()
        logger.info(
            f"Secrets configuration - Provider: {secrets_manager.provider_type}, "
            f"Environment: {self.environment}"
        )

        # Warn if using development secrets in production
        if self.environment == "production":
            self._check_development_secrets_in_production()

    def _check_development_secrets_in_production(self) -> None:
        """
        Check for development secrets being used in production.

        PATTERN: Security audit on startup
        WHY: Prevent accidental use of development credentials
        """
        development_indicators = [
            ("ANTHROPIC_API_KEY", "sk-ant-"),
            ("OPENAI_API_KEY", "sk-"),
            ("DATABASE_URL", "sqlite://"),
            ("REDIS_URL", "localhost"),
        ]

        warnings_found = []

        for secret_name, dev_pattern in development_indicators:
            value = os.getenv(secret_name, "")
            if value:
                # Check for obvious development patterns
                if secret_name == "DATABASE_URL" and dev_pattern in value:
                    warnings_found.append(
                        f"{secret_name} appears to be using SQLite (development database)"
                    )
                elif secret_name == "REDIS_URL" and dev_pattern in value:
                    warnings_found.append(
                        f"{secret_name} appears to be using localhost (development Redis)"
                    )

        # Log all warnings
        if warnings_found:
            logger.warning(
                "Development secrets detected in production environment:\n" +
                "\n".join(f"  - {w}" for w in warnings_found) +
                "\n\nConsider using production-grade services and secrets management."
            )

    def _validate_production_settings(self) -> None:
        """
        Validate critical settings for production environment.

        PATTERN: Fail-fast validation with comprehensive checks
        WHY: Prevent deployment with insecure defaults
        SECURITY: Multi-layer validation for critical secrets
        """
        if self.environment != "production":
            return

        # ====================================================================
        # CRITICAL VALIDATION: JWT Secret Key
        # ====================================================================
        insecure_jwt_defaults = [
            "dev-secret-key-change-in-production",
            "your-secret-key-here",
            "changeme",
            "secret",
            "dev-secret",
            "test",
            "password",
            "secret-key",
        ]

        if not self.jwt_secret_key:
            raise ValueError(
                "SECURITY ERROR: JWT_SECRET_KEY must be set for production environment. "
                "Set SECRETS_PROVIDER=railway or SECRETS_PROVIDER=aws for managed secrets."
            )

        if self.jwt_secret_key.lower() in [d.lower() for d in insecure_jwt_defaults]:
            raise ValueError(
                f"SECURITY ERROR: JWT_SECRET_KEY is using an insecure default value. "
                f"Current value matches known insecure pattern. "
                f"Generate a secure key with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
            )

        if len(self.jwt_secret_key) < 32:
            raise ValueError(
                f"SECURITY ERROR: JWT_SECRET_KEY must be at least 32 characters for production security. "
                f"Current length: {len(self.jwt_secret_key)}. "
                f"Generate a secure key with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
            )

        # ====================================================================
        # CRITICAL VALIDATION: Required API Keys
        # ====================================================================
        required_secrets = {
            "ANTHROPIC_API_KEY": self.anthropic_api_key,
        }

        missing_secrets = [
            name for name, value in required_secrets.items()
            if not value or value in ["", "your_anthropic_api_key_here"]
        ]

        if missing_secrets:
            raise ValueError(
                f"SECURITY ERROR: Required secrets not set for production: {', '.join(missing_secrets)}. "
                f"Configure secrets using SECRETS_PROVIDER (env/railway/aws)."
            )

        # ====================================================================
        # WARNING: CORS Configuration
        # ====================================================================
        if self.cors_origins == ["*"]:
            logger.warning(
                "SECURITY WARNING: CORS_ORIGINS is set to wildcard ['*'] in production. "
                "This allows requests from any domain. "
                "Set CORS_ORIGINS to specific domains: CORS_ORIGINS=[\"https://yourdomain.com\"]"
            )

        # ====================================================================
        # WARNING: Database Configuration
        # ====================================================================
        if "sqlite://" in self.database_url.lower():
            logger.warning(
                "PRODUCTION WARNING: Using SQLite database in production. "
                "Consider migrating to PostgreSQL or MySQL for production workloads. "
                "Railway provides managed PostgreSQL: https://railway.app/databases"
            )

        # ====================================================================
        # VALIDATION: Security Headers
        # ====================================================================
        if not self.security_headers_enabled:
            logger.warning(
                "SECURITY WARNING: Security headers disabled in production. "
                "Enable with: SECURITY_HEADERS_ENABLED=true"
            )

        if not self.rate_limit_enabled:
            logger.warning(
                "SECURITY WARNING: Rate limiting disabled in production. "
                "Enable with: RATE_LIMIT_ENABLED=true"
            )

        # ====================================================================
        # INFO: Secrets Provider Status
        # ====================================================================
        secrets_manager = get_secrets_manager()
        logger.info(
            f"Production deployment using secrets provider: {secrets_manager.provider_type}"
        )

# Singleton instance
settings = Settings()