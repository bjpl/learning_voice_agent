"""
Dashboard Cache
===============

In-memory caching layer for dashboard data.

PATTERN: LRU-style cache with TTL
WHY: Reduce computation for frequently accessed dashboard data
SPARC: Efficient data caching with automatic expiration
"""

from datetime import datetime
from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class CacheEntry:
    """
    Cache entry with TTL tracking.

    Attributes:
        data: Cached data
        created_at: When entry was created
        ttl_seconds: Time-to-live in seconds
    """

    data: Any
    created_at: datetime
    ttl_seconds: int

    def is_valid(self) -> bool:
        """
        Check if cache entry is still valid.

        Returns:
            True if entry has not expired
        """
        elapsed = (datetime.utcnow() - self.created_at).total_seconds()
        return elapsed < self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()

    @property
    def remaining_ttl(self) -> float:
        """Get remaining TTL in seconds (negative if expired)."""
        return self.ttl_seconds - self.age_seconds


class DashboardCache:
    """
    Simple in-memory cache for dashboard data.

    PATTERN: LRU-style cache with TTL
    WHY: Reduce computation for frequently accessed data

    Features:
    - TTL-based expiration
    - Key-based invalidation
    - Full cache clearing
    - Statistics tracking

    Usage:
        cache = DashboardCache(default_ttl=300)

        # Set with default TTL
        cache.set("key", data)

        # Set with custom TTL
        cache.set("key", data, ttl=60)

        # Get cached value
        value = cache.get("key")  # Returns None if expired or missing

        # Invalidate specific key
        cache.invalidate("key")

        # Clear all
        cache.clear()
    """

    def __init__(self, default_ttl: int = 300):
        """
        Initialize dashboard cache.

        Args:
            default_ttl: Default TTL in seconds (5 minutes)
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._default_ttl = default_ttl
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """
        Get cached value if valid.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        entry = self._cache.get(key)

        if entry and entry.is_valid():
            self._hits += 1
            return entry.data
        elif entry:
            # Expired - remove
            self._misses += 1
            del self._cache[key]

        self._misses += 1
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set cache value with TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL override in seconds
        """
        self._cache[key] = CacheEntry(
            data=value,
            created_at=datetime.utcnow(),
            ttl_seconds=ttl or self._default_ttl
        )

    def invalidate(self, key: str) -> bool:
        """
        Remove specific cache entry.

        Args:
            key: Cache key to invalidate

        Returns:
            True if key was present and removed
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def invalidate_pattern(self, pattern: str) -> int:
        """
        Remove cache entries matching a pattern.

        Args:
            pattern: Key prefix to match

        Returns:
            Number of entries removed
        """
        keys_to_remove = [k for k in self._cache.keys() if k.startswith(pattern)]
        for key in keys_to_remove:
            del self._cache[key]
        return len(keys_to_remove)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        expired_keys = [k for k, v in self._cache.items() if not v.is_valid()]
        for key in expired_keys:
            del self._cache[key]
        return len(expired_keys)

    @property
    def size(self) -> int:
        """Get number of entries in cache."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate as percentage."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return (self._hits / total) * 100

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        valid_entries = sum(1 for e in self._cache.values() if e.is_valid())

        return {
            "total_entries": len(self._cache),
            "valid_entries": valid_entries,
            "expired_entries": len(self._cache) - valid_entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "default_ttl": self._default_ttl
        }

    def reset_stats(self) -> None:
        """Reset hit/miss statistics."""
        self._hits = 0
        self._misses = 0
