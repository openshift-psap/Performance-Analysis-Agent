"""Cache manager for Gemini API explicit context caching.

This module manages the creation, update, and retrieval of cached content
for the system prompt to reduce API costs and improve response times.
"""

import datetime
from typing import Optional

from google import genai
from google.genai import types

from psap_agent.src.core.prompt import get_system_prompt
from psap_agent.src.settings import settings
from psap_agent.utils.pylogger import get_python_logger

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)


class PromptCacheManager:
    """Manages cached content for the agent's system prompt."""
    
    # Model for caching - using gemini-2.5-flash
    # Gemini 2.5 Flash supports explicit context caching at $0.03/1M tokens (90% off)
    CACHE_MODEL = "models/gemini-2.5-flash"
    
    # Default TTL: 1 hour (can be extended up to 24 hours)
    DEFAULT_TTL_HOURS = 1
    
    # Cache display name for identification
    CACHE_DISPLAY_NAME = "rhaiis_performance_agent_system_prompt"
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the cache manager.
        
        Args:
            api_key: Google API key. If not provided, uses GOOGLE_API_KEY from settings.
        """
        self.client = genai.Client(api_key=api_key or settings.GOOGLE_API_KEY)
        self._cache: Optional[types.CachedContent] = None
        self._cache_name: Optional[str] = None
    
    def _get_ttl_seconds(self, hours: float = DEFAULT_TTL_HOURS) -> int:
        """Convert hours to seconds for TTL.
        
        Args:
            hours: Number of hours for TTL
            
        Returns:
            TTL in seconds
        """
        return int(hours * 3600)
    
    async def create_or_get_cache(
        self, 
        force_recreate: bool = False,
        ttl_hours: float = DEFAULT_TTL_HOURS
    ) -> types.CachedContent:
        """Create a new cache or retrieve existing one.
        
        Args:
            force_recreate: If True, delete existing cache and create new one
            ttl_hours: Time to live in hours (default: 1 hour)
            
        Returns:
            The cached content object
        """
        # Check if we already have a cache in memory
        if self._cache and not force_recreate:
            try:
                # Verify cache still exists and is valid
                cache = self.client.caches.get(name=self._cache.name)
                logger.info(f"Using existing cache: {cache.name}")
                return cache
            except Exception as e:
                logger.warning(f"Cached content no longer valid: {e}")
                self._cache = None
                self._cache_name = None
        
        # Try to find existing cache by display name
        if not force_recreate:
            try:
                for cache in self.client.caches.list():
                    if cache.display_name == self.CACHE_DISPLAY_NAME:
                        logger.info(f"Found existing cache: {cache.name}")
                        self._cache = cache
                        self._cache_name = cache.name
                        
                        # Update TTL to extend expiry
                        self._update_cache_ttl(ttl_hours)
                        return self._cache
            except Exception as e:
                logger.warning(f"Error listing caches: {e}")
        
        # Create new cache
        logger.info("Creating new cache for system prompt...")
        
        system_prompt = get_system_prompt()
        
        try:
            cache = self.client.caches.create(
                model=self.CACHE_MODEL,
                config=types.CreateCachedContentConfig(
                    display_name=self.CACHE_DISPLAY_NAME,
                    system_instruction=system_prompt,
                    ttl=f"{self._get_ttl_seconds(ttl_hours)}s",
                )
            )
            
            self._cache = cache
            self._cache_name = cache.name
            
            logger.info(
                f"Cache created successfully: {cache.name} "
                f"(expires in {ttl_hours} hours, "
                f"~{cache.usage_metadata.total_token_count} tokens)"
            )
            
            return cache
            
        except Exception as e:
            logger.error(f"Failed to create cache: {e}")
            raise
    
    def _update_cache_ttl(self, ttl_hours: float):
        """Update the TTL of the existing cache.
        
        Args:
            ttl_hours: New TTL in hours
        """
        if not self._cache:
            return
        
        try:
            self.client.caches.update(
                name=self._cache.name,
                config=types.UpdateCachedContentConfig(
                    ttl=f"{self._get_ttl_seconds(ttl_hours)}s"
                )
            )
            logger.info(f"Cache TTL updated to {ttl_hours} hours")
        except Exception as e:
            logger.warning(f"Failed to update cache TTL: {e}")
    
    def get_cache_name(self) -> Optional[str]:
        """Get the name of the current cache.
        
        Returns:
            Cache name if available, None otherwise
        """
        return self._cache_name
    
    def get_cache_stats(self) -> dict:
        """Get statistics about the current cache.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self._cache:
            return {
                "status": "no_cache",
                "message": "No cache available"
            }
        
        try:
            cache = self.client.caches.get(name=self._cache.name)
            
            # Get token count
            token_count = cache.usage_metadata.total_token_count if hasattr(cache.usage_metadata, 'total_token_count') else 0
            
            # Calculate time remaining
            expire_time = None
            time_remaining_seconds = 0
            time_remaining_human = "unknown"
            
            if hasattr(cache, 'expire_time') and cache.expire_time:
                try:
                    # Handle if expire_time is already a datetime object
                    if isinstance(cache.expire_time, datetime.datetime):
                        expire_time = cache.expire_time
                    else:
                        # Convert string to datetime
                        expire_time_str = str(cache.expire_time).replace('Z', '+00:00')
                        expire_time = datetime.datetime.fromisoformat(expire_time_str)
                    
                    now = datetime.datetime.now(datetime.timezone.utc)
                    time_remaining = expire_time - now
                    time_remaining_seconds = int(time_remaining.total_seconds())
                    
                    # Format as HH:MM:SS
                    hours, remainder = divmod(time_remaining_seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    time_remaining_human = f"{hours}:{minutes:02d}:{seconds:02d}"
                except Exception as e:
                    logger.warning(f"Could not parse expire_time: {e}")
            
            return {
                "status": "active",
                "cache_name": cache.name,
                "display_name": getattr(cache, 'display_name', 'N/A'),
                "model": getattr(cache, 'model', 'N/A'),
                "token_count": token_count,
                "created_at": getattr(cache, 'create_time', 'N/A'),
                "expires_at": str(cache.expire_time) if hasattr(cache, 'expire_time') else 'N/A',
                "time_remaining_seconds": time_remaining_seconds,
                "time_remaining_human": time_remaining_human,
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def delete_cache(self):
        """Delete the current cache."""
        if not self._cache:
            logger.warning("No cache to delete")
            return
        
        try:
            self.client.caches.delete(self._cache.name)
            logger.info(f"Cache deleted: {self._cache.name}")
            self._cache = None
            self._cache_name = None
        except Exception as e:
            logger.error(f"Failed to delete cache: {e}")
            raise
    
    def refresh_cache(self, ttl_hours: float = DEFAULT_TTL_HOURS):
        """Refresh the cache by recreating it with new TTL.
        
        Args:
            ttl_hours: New TTL in hours
        """
        logger.info("Refreshing cache...")
        return self.create_or_get_cache(force_recreate=True, ttl_hours=ttl_hours)


# Global cache manager instance
_cache_manager: Optional[PromptCacheManager] = None


def get_cache_manager() -> PromptCacheManager:
    """Get or create the global cache manager instance.
    
    Returns:
        The global PromptCacheManager instance
    """
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = PromptCacheManager()
    return _cache_manager

