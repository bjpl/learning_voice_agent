"""
Mock implementation of neo4j library for testing

Provides minimal mock classes needed for tests to import successfully.
"""

from typing import Any, Dict, List, Optional, Union
from unittest.mock import AsyncMock, MagicMock


# Type aliases for compatibility
AsyncDriver = AsyncMock
AsyncSession = AsyncMock
Driver = MagicMock
Session = MagicMock


class AsyncGraphDatabase:
    """Mock AsyncGraphDatabase class"""

    @staticmethod
    def driver(*args, **kwargs):
        """Create mock async driver"""
        driver = AsyncMock()
        driver.verify_connectivity = AsyncMock()
        driver.close = AsyncMock()

        # Mock session
        session = AsyncMock()
        session.run = AsyncMock()
        session.close = AsyncMock()

        # Mock session context manager
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock()

        driver.session = MagicMock(return_value=session)

        return driver


class GraphDatabase:
    """Mock GraphDatabase class"""

    @staticmethod
    def driver(*args, **kwargs):
        """Create mock driver"""
        driver = MagicMock()
        driver.verify_connectivity = MagicMock()
        driver.close = MagicMock()

        session = MagicMock()
        session.run = MagicMock()
        session.close = MagicMock()

        driver.session = MagicMock(return_value=session)

        return driver


class Record:
    """Mock Neo4j Record"""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def __getitem__(self, key):
        return self._data.get(key)

    def get(self, key, default=None):
        return self._data.get(key, default)

    def data(self):
        return self._data


class Result:
    """Mock Neo4j Result"""

    def __init__(self, records: List[Dict[str, Any]] = None):
        self._records = [Record(r) for r in (records or [])]

    def single(self):
        """Return single record"""
        if not self._records:
            return None
        return self._records[0]

    def values(self):
        """Return list of record values"""
        return [[r._data.get(k) for k in r._data] for r in self._records]

    def data(self):
        """Return list of record data"""
        return [r.data() for r in self._records]

    async def __aiter__(self):
        """Async iteration support"""
        for record in self._records:
            yield record


# Mock exceptions module
class exceptions:
    """Mock exceptions module"""

    class ServiceUnavailable(Exception):
        """Mock ServiceUnavailable exception"""
        pass

    class TransientError(Exception):
        """Mock TransientError exception"""
        pass

    class DatabaseError(Exception):
        """Mock DatabaseError exception"""
        pass

    class Neo4jError(Exception):
        """Mock Neo4jError exception"""
        pass


# Export common exceptions at module level
ServiceUnavailable = exceptions.ServiceUnavailable
TransientError = exceptions.TransientError
DatabaseError = exceptions.DatabaseError
Neo4jError = exceptions.Neo4jError
