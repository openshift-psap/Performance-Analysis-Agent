"""Shared S3 client factory for the MCP server.

Provides a single ``get_s3_client`` helper that handles the authentication
chain used across tools:

1. Explicit credentials (AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY)
2. IAM role / default boto3 credential chain
3. Anonymous (unsigned) access for public buckets
"""

from __future__ import annotations

from psap_mcp_server.src.settings import settings
from psap_mcp_server.utils.pylogger import get_python_logger

logger = get_python_logger()


def get_s3_client(region: str | None = None, test_bucket: str | None = None, test_key: str | None = None):
    """Return a boto3 S3 client using the shared auth chain.

    Auth priority:
        1. Explicit credentials from settings (AWS_ACCESS_KEY_ID / SECRET)
        2. Default boto3 credential chain (IAM role, env vars, etc.)
        3. Anonymous / unsigned access (for public buckets)

    If *test_bucket* and *test_key* are supplied, step 2 verifies the
    default credentials by issuing a ``head_object`` call and falls back
    to anonymous access on failure.

    Args:
        region: AWS region override.  Defaults to ``settings.S3_REGION``.
        test_bucket: Bucket name used to probe default credentials.
        test_key: Object key used to probe default credentials.

    Returns:
        A ``boto3.client('s3', ...)`` instance.

    Raises:
        ImportError: If boto3 is not installed.
    """
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    region = region or getattr(settings, "S3_REGION", "us-east-1")

    # 1. Explicit credentials
    if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
        logger.debug("Using explicit AWS credentials for S3 access")
        return boto3.client(
            "s3",
            region_name=region,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )

    # 2. Default credential chain (IAM role, env, etc.)
    try:
        logger.debug("Attempting S3 access with IAM role/default credentials")
        client = boto3.client("s3", region_name=region)
        if test_bucket and test_key:
            client.head_object(Bucket=test_bucket, Key=test_key)
        return client
    except Exception:
        pass

    # 3. Anonymous access
    logger.debug("IAM credentials not available, using anonymous access")
    return boto3.client(
        "s3",
        region_name=region,
        config=Config(signature_version=UNSIGNED),
    )
