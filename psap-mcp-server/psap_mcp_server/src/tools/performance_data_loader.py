"""Performance data loader utility for RHAIIS, MLPerf, and LLM-D dashboards.

This module provides centralized data loading functionality for all performance dashboards.
Supports loading data from S3 bucket (when configured) with automatic fallback to local files.
Includes TTL-based caching to reduce S3 requests while keeping data fresh.
"""

import io
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from psap_mcp_server.src.settings import settings
from psap_mcp_server.utils.pylogger import get_python_logger

logger = get_python_logger()

# Data file paths
# Container path for the baked-in CSV fallback
_CONTAINER_DATA_DIR = Path("/app/data")
# Local development fallback — looks for consolidated_dashboard.csv in the MCP project root
_LOCAL_DATA_DIR = Path(__file__).parent.parent.parent.parent.parent

DATA_DIR = _CONTAINER_DATA_DIR if _CONTAINER_DATA_DIR.exists() else _LOCAL_DATA_DIR

RHAIIS_DATA_PATH = DATA_DIR / "consolidated_dashboard.csv"
MLPERF_50_DATA_PATH = DATA_DIR / "mlperf-data" / "mlperf-5.0.csv"
MLPERF_51_DATA_PATH = DATA_DIR / "mlperf-data" / "mlperf-5.1.csv"
LLMD_DATA_PATH = DATA_DIR / "llmd-dashboard.csv"


def validate_data_availability() -> None:
    """Validate that performance data is available at startup.

    Checks whether the RHAIIS dashboard CSV can be loaded from S3 or exists
    locally. If neither is available, logs a clear error message with setup
    instructions and exits the process.

    This should be called during server startup (from main.py) to fail fast
    rather than returning cryptic errors when tools are invoked later.

    Raises:
        SystemExit: If no data source is available.
    """
    # 1. Check if S3 is configured and reachable
    if settings.S3_BUCKET:
        try:
            _read_csv_from_s3(settings.S3_BUCKET, settings.S3_KEY, settings.S3_REGION)
            logger.info(
                f"Data validation passed: S3 source available at "
                f"s3://{settings.S3_BUCKET}/{settings.S3_KEY}"
            )
            return
        except Exception as e:
            logger.warning(f"S3 data source not reachable: {e}")

    # 2. Check if the local CSV fallback exists
    if RHAIIS_DATA_PATH.exists():
        logger.info(f"Data validation passed: local CSV found at {RHAIIS_DATA_PATH}")
        return

    # 3. Neither source available — fail fast with instructions
    logger.critical(
        "\n"
        "════════════════════════════════════════════════════════════════\n"
        "  FATAL: No performance data available!\n"
        "════════════════════════════════════════════════════════════════\n"
        "\n"
        "  The MCP server requires the RHAIIS consolidated dashboard CSV\n"
        "  but could not find it from any source.\n"
        "\n"
        "  To fix this, do ONE of the following:\n"
        "\n"
        "  Option 1 — Configure S3 (recommended for production):\n"
        "    Export these environment variables before starting the server:\n"
        "      export S3_BUCKET=psap-dashboard-data\n"
        "      export S3_KEY=main/rhaiis-dashboard/consolidated_dashboard.csv\n"
        "      export AWS_ACCESS_KEY_ID=<your-key>\n"
        "      export AWS_SECRET_ACCESS_KEY=<your-secret>\n"
        "\n"
        "  Option 2 — Copy the CSV locally:\n"
        f"    Download consolidated_dashboard.csv from the S3 bucket and\n"
        f"    place it at: {RHAIIS_DATA_PATH}\n"
        "\n"
        "    Example:\n"
        "      aws s3 cp s3://psap-dashboard-data/main/rhaiis-dashboard/"
        "consolidated_dashboard.csv \\\n"
        f"        {RHAIIS_DATA_PATH}\n"
        "\n"
        "════════════════════════════════════════════════════════════════"
    )
    sys.exit(1)


@dataclass
class CacheEntry:
    """Cache entry for storing DataFrame with timestamp."""
    data: pd.DataFrame
    timestamp: float
    source: str


class DataCache:
    """Thread-safe TTL cache for performance data.
    
    Caches DataFrames in memory with automatic expiration based on TTL.
    Thread-safe for concurrent access from multiple requests.
    """
    
    def __init__(self):
        self._cache: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
    
    def get(self, key: str, ttl_seconds: int) -> Optional[CacheEntry]:
        """Get cached entry if valid (not expired).
        
        Args:
            key: Cache key
            ttl_seconds: Time-to-live in seconds (0 = no caching)
            
        Returns:
            CacheEntry if valid, None if expired or not found
        """
        if ttl_seconds == 0:
            return None
            
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            
            age = time.time() - entry.timestamp
            if age > ttl_seconds:
                # Cache expired
                logger.debug(f"Cache expired for '{key}' (age: {age:.1f}s, TTL: {ttl_seconds}s)")
                del self._cache[key]
                return None
            
            logger.debug(f"Cache hit for '{key}' (age: {age:.1f}s, TTL: {ttl_seconds}s)")
            return entry
    
    def set(self, key: str, data: pd.DataFrame, source: str) -> None:
        """Store DataFrame in cache.
        
        Args:
            key: Cache key
            data: DataFrame to cache
            source: Description of data source (for logging)
        """
        with self._lock:
            self._cache[key] = CacheEntry(
                data=data.copy(),  # Store a copy to prevent mutations
                timestamp=time.time(),
                source=source
            )
            logger.debug(f"Cached data for '{key}' from {source}")
    
    def invalidate(self, key: Optional[str] = None) -> None:
        """Invalidate cache entry or all entries.
        
        Args:
            key: Specific key to invalidate, or None for all
        """
        with self._lock:
            if key is None:
                self._cache.clear()
                logger.info("Invalidated all cache entries")
            elif key in self._cache:
                del self._cache[key]
                logger.info(f"Invalidated cache entry: {key}")
    
    def get_stats(self) -> dict:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        with self._lock:
            stats = {}
            current_time = time.time()
            for key, entry in self._cache.items():
                age = current_time - entry.timestamp
                stats[key] = {
                    "source": entry.source,
                    "age_seconds": round(age, 1),
                    "rows": len(entry.data),
                }
            return stats


# Global cache instance
_data_cache = DataCache()


def _read_csv_from_s3(bucket: str, key: str, region: str = "us-east-1") -> pd.DataFrame:
    """Read a CSV file from S3 bucket.
    
    Uses the shared S3 client factory which handles authentication in
    this priority:
    1. Explicit credentials (AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY)
    2. IAM role (default boto3 credential chain)
    3. Anonymous access (for public buckets)
    
    Args:
        bucket: S3 bucket name
        key: S3 object key (path to file in bucket)
        region: AWS region (default: us-east-1)
        
    Returns:
        DataFrame with the CSV data
        
    Raises:
        ImportError: If boto3 is not installed
        Exception: If S3 read fails
    """
    try:
        from psap_mcp_server.src.tools.s3_utils import get_s3_client

        s3_client = get_s3_client(
            region=region, test_bucket=bucket, test_key=key
        )
        
        # Fetch the object
        response = s3_client.get_object(Bucket=bucket, Key=key)
        csv_content = response["Body"].read().decode("utf-8")
        
        return pd.read_csv(io.StringIO(csv_content))
        
    except ImportError:
        raise ImportError("boto3 is required for S3 access. Install with: pip install boto3")
    except Exception as e:
        raise Exception(f"Failed to read from S3 bucket '{bucket}', key '{key}': {str(e)}")


def _clean_rhaiis_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply standard cleaning to RHAIIS DataFrame.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    # Data cleaning
    df["run"] = df["run"].str.strip()
    df["accelerator"] = df["accelerator"].str.strip()
    df["model"] = df["model"].str.strip()
    df["version"] = df["version"].str.strip()
    df["TP"] = pd.to_numeric(df["TP"], errors="coerce")
    
    # Add derived columns
    if "output_tok/sec" in df.columns and "TP" in df.columns:
        df["efficiency_ratio"] = df["output_tok/sec"] / df["TP"]
    
    return df


def load_rhaiis_data(force_refresh: bool = False) -> Optional[pd.DataFrame]:
    """Load RHAIIS benchmark data with caching.
    
    Attempts to load from S3 if S3_BUCKET is configured, with automatic
    fallback to local file if S3 fails or is not configured.
    
    Data is cached in memory based on S3_CACHE_TTL_SECONDS setting.
    
    Args:
        force_refresh: If True, bypass cache and fetch fresh data

    Returns:
        DataFrame with RHAIIS benchmark data, or None if loading fails.
    """
    cache_key = "rhaiis_data"
    ttl = settings.S3_CACHE_TTL_SECONDS
    
    # Check cache first (unless force refresh)
    if not force_refresh:
        cached = _data_cache.get(cache_key, ttl)
        if cached is not None:
            logger.info(f"Using cached RHAIIS data ({len(cached.data)} rows, from {cached.source})")
            return cached.data.copy()  # Return copy to prevent mutations
    
    try:
        df = None
        source = "unknown"
        
        # Try S3 first if configured
        if settings.S3_BUCKET:
            try:
                df = _read_csv_from_s3(
                    settings.S3_BUCKET,
                    settings.S3_KEY,
                    settings.S3_REGION
                )
                source = f"S3: s3://{settings.S3_BUCKET}/{settings.S3_KEY}"
                logger.info(f"Fetched fresh RHAIIS data from {source}")
            except Exception as s3_error:
                logger.warning(f"S3 load failed ({s3_error}), falling back to local file")
                df = None
        
        # Fall back to local file
        if df is None:
            if not RHAIIS_DATA_PATH.exists():
                logger.error(f"RHAIIS data file not found: {RHAIIS_DATA_PATH}")
                return None
            
            df = pd.read_csv(RHAIIS_DATA_PATH)
            source = f"local: {RHAIIS_DATA_PATH}"
            logger.info(f"Loaded RHAIIS data from {source}")
        
        # Clean the data
        df = _clean_rhaiis_data(df)
        
        # Cache the cleaned data
        if ttl > 0:
            _data_cache.set(cache_key, df, source)
            logger.info(f"Cached RHAIIS data: {len(df)} rows, TTL: {ttl}s")
        
        logger.info(f"Loaded RHAIIS data: {len(df)} rows, {len(df.columns)} columns from {source}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading RHAIIS data: {e}")
        return None


def refresh_rhaiis_data() -> Optional[pd.DataFrame]:
    """Force refresh RHAIIS data from source, bypassing cache.
    
    Use this to immediately get the latest data from S3 without
    waiting for cache expiration.
    
    Returns:
        DataFrame with fresh RHAIIS benchmark data, or None if loading fails.
    """
    logger.info("Force refreshing RHAIIS data...")
    _data_cache.invalidate("rhaiis_data")
    return load_rhaiis_data(force_refresh=True)


def get_cache_stats() -> dict:
    """Get current cache statistics.
    
    Returns:
        Dictionary with cache information including age and source of cached data.
    """
    stats = _data_cache.get_stats()
    stats["ttl_seconds"] = settings.S3_CACHE_TTL_SECONDS
    stats["s3_configured"] = bool(settings.S3_BUCKET)
    if settings.S3_BUCKET:
        stats["s3_uri"] = f"s3://{settings.S3_BUCKET}/{settings.S3_KEY}"
    return stats


def load_mlperf_data(version: str = "5.1") -> Optional[pd.DataFrame]:
    """Load MLPerf benchmark data.

    Args:
        version: MLPerf version ("5.0" or "5.1")

    Returns:
        DataFrame with MLPerf benchmark data, or None if loading fails.
    """
    try:
        if version == "5.0":
            data_path = MLPERF_50_DATA_PATH
        elif version == "5.1":
            data_path = MLPERF_51_DATA_PATH
        else:
            logger.error(f"Invalid MLPerf version: {version}")
            return None

        if not data_path.exists():
            logger.error(f"MLPerf data file not found: {data_path}")
            return None

        # MLPerf files have complex multi-row headers, simplified loading here
        df = pd.read_csv(data_path, skiprows=5, header=None, low_memory=False)
        
        logger.info(f"Loaded MLPerf {version} data: {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error loading MLPerf data: {e}")
        return None


def load_llmd_data() -> Optional[pd.DataFrame]:
    """Load LLM-D disaggregated architecture benchmark data.

    Returns:
        DataFrame with LLM-D benchmark data, or None if loading fails.
    """
    try:
        if not LLMD_DATA_PATH.exists():
            logger.error(f"LLM-D data file not found: {LLMD_DATA_PATH}")
            return None

        df = pd.read_csv(LLMD_DATA_PATH)
        
        # Data cleaning
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].str.strip()
        
        # Convert numeric columns
        numeric_cols = ["TP", "DP", "EP", "replicas", "prefill_pod_count", "decode_pod_count"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Calculate efficiency ratio
        if "output_tok/sec" in df.columns and "TP" in df.columns:
            df["efficiency_ratio"] = df["output_tok/sec"] / df["TP"]
        
        logger.info(f"Loaded LLM-D data: {len(df)} rows, {len(df.columns)} columns")
        return df
    except Exception as e:
        logger.error(f"Error loading LLM-D data: {e}")
        return None


def get_available_models(dashboard: str = "rhaiis") -> list[str]:
    """Get list of available models in the specified dashboard.

    Args:
        dashboard: Dashboard name ("rhaiis", "mlperf", "llmd")

    Returns:
        List of unique model names.
    """
    try:
        if dashboard == "rhaiis":
            df = load_rhaiis_data()
        elif dashboard == "llmd":
            df = load_llmd_data()
        else:
            return []

        if df is None or "model" not in df.columns:
            return []

        return sorted(df["model"].unique().tolist())
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        return []


def get_base_accelerator(name: str) -> str:
    """Extract the base GPU type from a cluster-specific accelerator name.

    Cluster-specific names follow the pattern ``{GPU_TYPE}_{CLUSTER}``
    (e.g. ``H200_ZEUS2``, ``H200_HERA``).  This returns the base GPU type
    (``H200``) so tools can cross-reference data across clusters that use
    the same hardware.

    If the name has no underscore or is already a base type, returns it
    unchanged (e.g. ``H200`` → ``H200``, ``MI300X`` → ``MI300X``).
    """
    if not name:
        return name
    parts = name.split("_", 1)
    return parts[0]


def get_available_accelerators(dashboard: str = "rhaiis") -> list[str]:
    """Get list of available accelerators in the specified dashboard.

    Args:
        dashboard: Dashboard name ("rhaiis", "llmd")

    Returns:
        List of unique accelerator names.
    """
    try:
        if dashboard == "rhaiis":
            df = load_rhaiis_data()
        elif dashboard == "llmd":
            df = load_llmd_data()
        else:
            return []

        if df is None or "accelerator" not in df.columns:
            return []

        return sorted(df["accelerator"].unique().tolist())
    except Exception as e:
        logger.error(f"Error getting available accelerators: {e}")
        return []
