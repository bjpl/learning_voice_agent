"""
Configuration Management
PATTERN: Singleton configuration with environment validation
"""
import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # API Keys
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    twilio_account_sid: Optional[str] = Field(None, env="TWILIO_ACCOUNT_SID")
    twilio_auth_token: Optional[str] = Field(None, env="TWILIO_AUTH_TOKEN")
    twilio_phone_number: Optional[str] = Field(None, env="TWILIO_PHONE_NUMBER")

    # Database
    database_url: str = Field("sqlite:///./learning_captures.db", env="DATABASE_URL")

    # Redis
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    redis_ttl: int = Field(1800, env="REDIS_TTL")  # 30 minutes

    # Server
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")
    cors_origins: list = Field(["*"], env="CORS_ORIGINS")

    # Audio Processing
    whisper_model: str = Field("whisper-1", env="WHISPER_MODEL")
    max_audio_duration: int = Field(60, env="MAX_AUDIO_DURATION")  # seconds

    # Claude Configuration
    claude_model: str = Field("claude-3-haiku-20240307", env="CLAUDE_MODEL")
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

    # Environment
    environment: str = Field("development", env="ENVIRONMENT")

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validate_production_settings()

    def _validate_production_settings(self) -> None:
        """
        Validate critical settings for production environment.

        PATTERN: Fail-fast validation
        WHY: Prevent deployment with insecure defaults
        """
        if self.environment == "production":
            # JWT Secret Key validation
            insecure_defaults = [
                "dev-secret-key-change-in-production",
                "your-secret-key-here",
                "changeme",
                "secret",
                "dev-secret",
            ]

            if not self.jwt_secret_key:
                raise ValueError(
                    "JWT_SECRET_KEY must be set for production environment"
                )

            if self.jwt_secret_key.lower() in [d.lower() for d in insecure_defaults]:
                raise ValueError(
                    "JWT_SECRET_KEY must not use insecure default values in production"
                )

            if len(self.jwt_secret_key) < 32:
                raise ValueError(
                    "JWT_SECRET_KEY must be at least 32 characters for production security"
                )

            # Validate CORS is not wildcard in production
            if self.cors_origins == ["*"]:
                import logging
                logging.warning(
                    "CORS_ORIGINS is set to wildcard ['*'] in production. "
                    "Consider restricting to specific domains."
                )

# Singleton instance
settings = Settings()