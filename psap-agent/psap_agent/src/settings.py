"""Settings configuration for the PSAP agent.

This module provides centralized configuration management using Pydantic
BaseSettings for environment variable loading, validation, and default
value handling for the PSAP agent service.
"""

from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

from psap_agent.src.core.exceptions.exceptions import AppException, AppExceptionCode
from psap_agent.utils.pylogger import get_python_logger

# Initialize logger
logger = get_python_logger()

# Load environment variables with error handling
try:
    load_dotenv()
except Exception as e:
    # Log error but don't fail - environment variables might be set directly
    logger.warning(f"Could not load .env file: {e}")


class Settings(BaseSettings):
    """Configuration settings for the PSAP agent.

    Uses Pydantic BaseSettings to load and validate configuration from
    environment variables. Provides default values for optional settings
    and validation for required ones.

    The settings are organized into logical groups:
    - Server Configuration: Host, port, SSL settings
    - Database Configuration: PostgreSQL connection parameters
    - Langfuse Configuration: Tracing and analytics settings
    - Google Configuration: Service account credentials
    - SSO Configuration: Authentication settings
    """

    # Server Configuration
    AGENT_HOST: str = Field(default="0.0.0.0", json_schema_extra={"env": "AGENT_HOST"})
    AGENT_PORT: int = Field(default=8081, json_schema_extra={"env": "AGENT_PORT"})
    AGENT_SSL_KEYFILE: Optional[str] = Field(
        default=None, json_schema_extra={"env": "AGENT_SSL_KEYFILE"}
    )
    AGENT_SSL_CERTFILE: Optional[str] = Field(
        default=None, json_schema_extra={"env": "AGENT_SSL_CERTFILE"}
    )
    
    # MCP Server Configuration
    MCP_SERVER_URL: str = Field(
        default="http://localhost:5001/mcp/", 
        json_schema_extra={"env": "MCP_SERVER_URL"}
    )
    PYTHON_LOG_LEVEL: str = Field(
        default="INFO", json_schema_extra={"env": "PYTHON_LOG_LEVEL"}
    )
    USE_INMEMORY_SAVER: bool = Field(
        default=False, json_schema_extra={"env": "USE_INMEMORY_SAVER"}
    )

    # Database Configuration
    POSTGRES_USER: str = Field(
        default="pgvector", json_schema_extra={"env": "POSTGRES_USER"}
    )
    POSTGRES_PASSWORD: str = Field(
        default="pgvector", json_schema_extra={"env": "POSTGRES_PASSWORD"}
    )
    POSTGRES_DB: str = Field(
        default="pgvector", json_schema_extra={"env": "POSTGRES_DB"}
    )
    POSTGRES_HOST: str = Field(
        default="pgvector", json_schema_extra={"env": "POSTGRES_HOST"}
    )
    POSTGRES_PORT: int = Field(default=5432, json_schema_extra={"env": "POSTGRES_PORT"})

    # Google Service Account Configuration
    GOOGLE_SERVICE_ACCOUNT_FILE: Optional[str] = Field(
        default=None, json_schema_extra={"env": "GOOGLE_SERVICE_ACCOUNT_FILE"}
    )
    GOOGLE_API_KEY: Optional[str] = Field(
        default=None, json_schema_extra={"env": "GOOGLE_API_KEY"}
    )
    
    # LLM Model Configuration
    GEMINI_MODEL: str = Field(
        default="gemini-3-flash-preview", json_schema_extra={"env": "GEMINI_MODEL"}
    )

    # Context Caching Configuration
    ENABLE_PROMPT_CACHING: bool = Field(
        default=True, json_schema_extra={"env": "ENABLE_PROMPT_CACHING"}
    )
    CACHE_TTL_HOURS: float = Field(
        default=4.0, json_schema_extra={"env": "CACHE_TTL_HOURS"}
    )

    # Langfuse Configuration
    LANGFUSE_PUBLIC_KEY: Optional[str] = Field(
        default=None, json_schema_extra={"env": "LANGFUSE_PUBLIC_KEY"}
    )
    LANGFUSE_SECRET_KEY: Optional[str] = Field(
        default=None, json_schema_extra={"env": "LANGFUSE_SECRET_KEY"}
    )
    LANGFUSE_HOST: Optional[str] = Field(
        default=None, json_schema_extra={"env": "LANGFUSE_HOST"}
    )
    LANGFUSE_TRACING_ENVIRONMENT: str = Field(
        default="development", json_schema_extra={"env": "LANGFUSE_TRACING_ENVIRONMENT"}
    )

    # Snowflake Configuration
    SNOWFLAKE_ACCOUNT: Optional[str] = Field(
        default=None, json_schema_extra={"env": "SNOWFLAKE_ACCOUNT"}
    )

    # SSO Configuration
    SSO_CLIENT_ID: Optional[str] = Field(
        default=None, json_schema_extra={"env": "SSO_CLIENT_ID"}
    )
    SSO_CLIENT_SECRET: Optional[str] = Field(
        default=None, json_schema_extra={"env": "SSO_CLIENT_SECRET"}
    )
    SSO_CALLBACK_URL: str = Field(
        default="http://0.0.0.0:8081/callback/",
        json_schema_extra={"env": "SSO_CALLBACK_URL"},
    )

    # Session Configuration
    SESSION_SECRET: Optional[str] = Field(
        default=None, json_schema_extra={"env": "SESSION_SECRET"}
    )

    # Google Application Credentials
    GOOGLE_APPLICATION_CREDENTIALS_CONTENT: Optional[str] = Field(
        default=None,
        json_schema_extra={"env": "GOOGLE_APPLICATION_CREDENTIALS_CONTENT"},
    )

    @property
    def database_uri(self) -> str:
        """Generate database URI from individual components.

        Constructs a PostgreSQL connection URI using the configured
        database settings including user, password, host, port, and
        database name.

        Returns:
            The complete PostgreSQL database URI string.
        """
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"


def validate_config(settings: Settings) -> None:
    """Validate configuration settings.

    Performs comprehensive validation to ensure required settings are
    present and values are within acceptable ranges. This function
    validates port ranges, log levels, and transport protocols.

    Args:
        settings: Settings instance to validate.

    Raises:
        ValueError: If required configuration is missing or invalid.
    """
    # Validate port range
    if not (1024 <= settings.AGENT_PORT <= 65535):
        logger.error(
            f"AGENT_PORT must be between 1024 and 65535, got {settings.AGENT_PORT}"
        )
        raise AppException(
            f"AGENT_PORT must be between 1024 and 65535, got {settings.AGENT_PORT}",
            AppExceptionCode.CONFIGURATION_VALIDATION_ERROR,
        )

    # Validate log level
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if settings.PYTHON_LOG_LEVEL.upper() not in valid_log_levels:
        logger.error(
            f"PYTHON_LOG_LEVEL must be one of {valid_log_levels}, got {settings.PYTHON_LOG_LEVEL}"
        )
        raise AppException(
            f"PYTHON_LOG_LEVEL must be one of {valid_log_levels}, got {settings.PYTHON_LOG_LEVEL}",
            AppExceptionCode.CONFIGURATION_VALIDATION_ERROR,
        )


# Create settings instance without validation (validation happens in main.py)
settings = Settings()
