"""
Base Store - Abstract Base Class for SQLite Persistence Layers
PATTERN: Template Method with async operations
WHY: Unified interface for all store implementations, reducing code duplication

Features:
- Async SQLite operations with aiosqlite
- Common CRUD operation patterns
- Centralized error handling
- Transaction support
- Consistent logging patterns
- Connection pooling via context manager

Usage:
    class MyStore(BaseStore[MyModel]):
        def __init__(self, db_path: str = "data/my_store.db"):
            super().__init__(db_path, "my_store")

        async def _create_schema(self, db) -> None:
            await db.execute('''CREATE TABLE IF NOT EXISTS my_table ...''')

        async def save(self, item: MyModel) -> str:
            ...
"""
import aiosqlite
import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from datetime import datetime
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    TypeVar,
    Type,
    Callable,
    AsyncGenerator,
)

from app.logger import get_logger

# Type variable for model types
T = TypeVar('T')


class StoreError(Exception):
    """Base exception for store operations."""

    def __init__(self, message: str, operation: str, original_error: Optional[Exception] = None):
        self.message = message
        self.operation = operation
        self.original_error = original_error
        super().__init__(f"{operation}: {message}")


class ConnectionError(StoreError):
    """Database connection error."""
    pass


class TransactionError(StoreError):
    """Transaction-related error."""
    pass


class ValidationError(StoreError):
    """Data validation error."""
    pass


class BaseStore(ABC, Generic[T]):
    """
    Abstract base class for SQLite-backed stores.

    PATTERN: Template Method with Repository pattern
    WHY: Provides consistent interface and reduces boilerplate across stores

    Subclasses must implement:
    - _create_schema(db): Define database tables and indexes
    - Model-specific CRUD methods

    Features:
    - Automatic initialization on first use
    - Thread-safe async operations
    - Consistent error handling and logging
    - Transaction support
    - JSON metadata column support
    """

    def __init__(self, db_path: str, store_name: str):
        """
        Initialize the base store.

        Args:
            db_path: Path to SQLite database file
            store_name: Name for logging purposes
        """
        self.db_path = db_path
        self._store_name = store_name
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._logger = get_logger(store_name)

    @property
    def is_initialized(self) -> bool:
        """Check if store has been initialized."""
        return self._initialized

    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        """
        Get a database connection with row factory.

        Yields:
            aiosqlite.Connection with Row factory set

        Raises:
            ConnectionError: If connection cannot be established
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                yield db
        except aiosqlite.Error as e:
            self._logger.error(
                f"{self._store_name}_connection_error",
                db_path=self.db_path,
                error=str(e),
                error_type=type(e).__name__
            )
            raise ConnectionError(
                f"Failed to connect to database: {e}",
                "get_connection",
                e
            )

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        """
        Get a database connection with transaction support.

        Automatically commits on success, rolls back on error.

        Yields:
            aiosqlite.Connection in transaction mode

        Raises:
            TransactionError: If transaction fails
        """
        async with self.get_connection() as db:
            try:
                yield db
                await db.commit()
            except Exception as e:
                await db.rollback()
                self._logger.error(
                    f"{self._store_name}_transaction_failed",
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise TransactionError(
                    f"Transaction failed: {e}",
                    "transaction",
                    e
                )

    async def initialize(self) -> None:
        """
        Initialize database schema.

        Thread-safe initialization that creates tables and indexes.
        Only runs once per store instance.

        Raises:
            StoreError: If initialization fails
        """
        async with self._init_lock:
            if self._initialized:
                return

            self._logger.info(
                f"{self._store_name}_initialization_started",
                db_path=self.db_path
            )

            try:
                # Ensure directory exists
                import os
                os.makedirs(os.path.dirname(self.db_path) or '.', exist_ok=True)

                async with aiosqlite.connect(self.db_path) as db:
                    db.row_factory = aiosqlite.Row

                    # Call subclass schema creation
                    await self._create_schema(db)

                    # Create any additional indexes
                    await self._create_indexes(db)

                    await db.commit()

                self._initialized = True
                self._logger.info(
                    f"{self._store_name}_initialization_complete",
                    db_path=self.db_path
                )

            except Exception as e:
                self._logger.error(
                    f"{self._store_name}_initialization_failed",
                    db_path=self.db_path,
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise StoreError(
                    f"Failed to initialize store: {e}",
                    "initialize",
                    e
                )

    async def ensure_initialized(self) -> None:
        """Ensure store is initialized before operations."""
        if not self._initialized:
            await self.initialize()

    @abstractmethod
    async def _create_schema(self, db: aiosqlite.Connection) -> None:
        """
        Create database schema.

        Subclasses must implement this to define their tables.

        Args:
            db: Database connection
        """
        pass

    async def _create_indexes(self, db: aiosqlite.Connection) -> None:
        """
        Create database indexes.

        Override in subclasses to add custom indexes.
        Called after _create_schema during initialization.

        Args:
            db: Database connection
        """
        pass

    async def close(self) -> None:
        """
        Close the store and cleanup resources.

        Resets initialization state.
        """
        self._initialized = False
        self._logger.info(
            f"{self._store_name}_closed",
            db_path=self.db_path
        )

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def _generate_id(self) -> str:
        """Generate a unique ID for new records."""
        return str(uuid.uuid4())

    def _now_iso(self) -> str:
        """Get current UTC timestamp in ISO format."""
        return datetime.utcnow().isoformat()

    def _to_json(self, data: Optional[Any]) -> Optional[str]:
        """Convert data to JSON string for storage."""
        if data is None:
            return None
        return json.dumps(data)

    def _from_json(self, data: Optional[str], default: Any = None) -> Any:
        """Parse JSON string from storage."""
        if data is None:
            return default
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return default

    async def _execute_query(
        self,
        query: str,
        params: tuple = (),
        operation: str = "query"
    ) -> List[aiosqlite.Row]:
        """
        Execute a query and return results.

        Args:
            query: SQL query string
            params: Query parameters
            operation: Operation name for logging

        Returns:
            List of row results

        Raises:
            StoreError: If query fails
        """
        await self.ensure_initialized()

        try:
            async with self.get_connection() as db:
                cursor = await db.execute(query, params)
                return await cursor.fetchall()
        except aiosqlite.Error as e:
            self._logger.error(
                f"{self._store_name}_{operation}_failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise StoreError(f"Query failed: {e}", operation, e)

    async def _execute_write(
        self,
        query: str,
        params: tuple = (),
        operation: str = "write"
    ) -> int:
        """
        Execute a write operation (INSERT, UPDATE, DELETE).

        Args:
            query: SQL query string
            params: Query parameters
            operation: Operation name for logging

        Returns:
            Number of affected rows

        Raises:
            StoreError: If write fails
        """
        await self.ensure_initialized()

        try:
            async with self.transaction() as db:
                cursor = await db.execute(query, params)
                return cursor.rowcount
        except aiosqlite.Error as e:
            self._logger.error(
                f"{self._store_name}_{operation}_failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise StoreError(f"Write failed: {e}", operation, e)

    async def count(self, table: str, where: str = "", params: tuple = ()) -> int:
        """
        Get count of records in a table.

        Args:
            table: Table name
            where: Optional WHERE clause (without 'WHERE' keyword)
            params: Query parameters for WHERE clause

        Returns:
            Count of matching records
        """
        await self.ensure_initialized()

        try:
            query = f"SELECT COUNT(*) as count FROM {table}"
            if where:
                query += f" WHERE {where}"

            async with self.get_connection() as db:
                cursor = await db.execute(query, params)
                row = await cursor.fetchone()
                return row['count'] if row else 0
        except aiosqlite.Error as e:
            self._logger.error(
                f"{self._store_name}_count_failed",
                table=table,
                error=str(e)
            )
            return 0

    async def exists(self, table: str, where: str, params: tuple = ()) -> bool:
        """
        Check if a record exists.

        Args:
            table: Table name
            where: WHERE clause (without 'WHERE' keyword)
            params: Query parameters

        Returns:
            True if record exists
        """
        count = await self.count(table, where, params)
        return count > 0

    async def delete_by_id(self, table: str, id_column: str, id_value: str) -> bool:
        """
        Delete a record by ID.

        Args:
            table: Table name
            id_column: Name of ID column
            id_value: ID value to delete

        Returns:
            True if record was deleted
        """
        try:
            rows = await self._execute_write(
                f"DELETE FROM {table} WHERE {id_column} = ?",
                (id_value,),
                "delete"
            )

            if rows > 0:
                self._logger.info(
                    f"{self._store_name}_record_deleted",
                    table=table,
                    id=id_value
                )

            return rows > 0
        except StoreError:
            return False


# Type alias for convenience
SQLiteStore = BaseStore
