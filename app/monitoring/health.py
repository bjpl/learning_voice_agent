"""
Health Check Endpoints

SPECIFICATION:
- GET /health - Basic health (returns 200 if running)
- GET /health/ready - Readiness (checks DB, vector store)
- GET /health/live - Liveness (simple ping)
- GET /health/detailed - Detailed status of all components

ARCHITECTURE:
- Async database checks for non-blocking operations
- Component-level health status tracking
- Proper HTTP status codes (200 healthy, 503 unhealthy)
- Structured response models

PATTERN: Strategy pattern for different health check types
WHY: Each component may have different health check requirements
"""

import os
import time
import shutil
import aiosqlite
from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pydantic import BaseModel

from app.config import settings
from app.logger import get_logger
from app.rag.config import rag_config

logger = get_logger(__name__)


class HealthStatus(str, Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status for a single component"""
    name: str
    status: HealthStatus
    latency_ms: Optional[float] = None
    message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    last_check: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "name": self.name,
            "status": self.status.value,
            "last_check": self.last_check,
        }
        if self.latency_ms is not None:
            result["latency_ms"] = round(self.latency_ms, 2)
        if self.message:
            result["message"] = self.message
        if self.details:
            result["details"] = self.details
        return result


class DetailedHealthResponse(BaseModel):
    """Response model for detailed health check"""
    status: str
    timestamp: str
    version: str
    uptime_seconds: float
    components: Dict[str, Any]
    summary: Dict[str, int]


class HealthChecker:
    """
    Comprehensive health checker for all system components

    PATTERN: Singleton with lazy initialization
    WHY: Avoid repeated initialization overhead, share state

    Features:
    - SQLite database health
    - ChromaDB vector store health
    - API key configuration validation
    - Disk space monitoring
    - Memory usage tracking
    """

    def __init__(self):
        self._startup_time: Optional[float] = None
        self._last_detailed_check: Optional[Dict[str, Any]] = None
        self._last_check_time: float = 0
        self._cache_ttl: float = 5.0  # Cache detailed checks for 5 seconds

    def set_startup_time(self, startup_time: float) -> None:
        """Set application startup time for uptime calculation"""
        self._startup_time = startup_time

    def get_uptime_seconds(self) -> float:
        """Get application uptime in seconds"""
        if self._startup_time is None:
            return 0.0
        return time.time() - self._startup_time

    async def check_basic(self) -> Dict[str, Any]:
        """
        Basic health check - just confirms the service is running

        PATTERN: Simple ping endpoint
        WHY: Fastest possible health check for load balancers

        Returns:
            Basic status with timestamp
        """
        return {
            "status": HealthStatus.HEALTHY.value,
            "timestamp": datetime.utcnow().isoformat(),
            "service": "learning-voice-agent",
        }

    async def check_liveness(self) -> Dict[str, Any]:
        """
        Liveness probe - confirms the process is alive

        PATTERN: Kubernetes liveness probe
        WHY: Detect if the process is stuck or deadlocked

        Returns:
            Simple alive status
        """
        return {
            "status": "alive",
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def check_readiness(self) -> tuple[Dict[str, Any], int]:
        """
        Readiness probe - checks if service can accept traffic

        PATTERN: Kubernetes readiness probe
        WHY: Only route traffic to instances that can handle it

        Checks:
        - Database connectivity
        - Vector store availability
        - Required API keys configured

        Returns:
            Tuple of (response dict, HTTP status code)
        """
        checks = []
        all_ready = True

        # Check database
        db_health = await self.check_database()
        checks.append(db_health)
        if db_health.status != HealthStatus.HEALTHY:
            all_ready = False

        # Check ChromaDB
        chroma_health = await self.check_chromadb()
        checks.append(chroma_health)
        if chroma_health.status != HealthStatus.HEALTHY:
            # ChromaDB being unhealthy is degraded, not critical
            pass

        # Check API keys configured
        api_health = await self.check_api_configuration()
        checks.append(api_health)
        if api_health.status == HealthStatus.UNHEALTHY:
            all_ready = False

        overall_status = HealthStatus.HEALTHY if all_ready else HealthStatus.UNHEALTHY
        status_code = 200 if all_ready else 503

        response = {
            "status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "ready": all_ready,
            "checks": [check.to_dict() for check in checks],
        }

        return response, status_code

    async def check_detailed(self) -> tuple[Dict[str, Any], int]:
        """
        Detailed health check - comprehensive status of all components

        PATTERN: Full system health dashboard
        WHY: Debugging and monitoring dashboards need detailed info

        Checks:
        - Database (SQLite)
        - Vector Store (ChromaDB)
        - API Configuration
        - Disk Space
        - System Resources

        Returns:
            Tuple of (detailed response dict, HTTP status code)
        """
        now = time.time()

        # Check cache
        if (
            self._last_detailed_check is not None
            and now - self._last_check_time < self._cache_ttl
        ):
            return self._last_detailed_check, 200

        components = {}
        healthy_count = 0
        degraded_count = 0
        unhealthy_count = 0

        # Database check
        db_health = await self.check_database()
        components["database"] = db_health.to_dict()
        self._update_counts(db_health.status, healthy_count, degraded_count, unhealthy_count)
        if db_health.status == HealthStatus.HEALTHY:
            healthy_count += 1
        elif db_health.status == HealthStatus.DEGRADED:
            degraded_count += 1
        else:
            unhealthy_count += 1

        # ChromaDB check
        chroma_health = await self.check_chromadb()
        components["vector_store"] = chroma_health.to_dict()
        if chroma_health.status == HealthStatus.HEALTHY:
            healthy_count += 1
        elif chroma_health.status == HealthStatus.DEGRADED:
            degraded_count += 1
        else:
            unhealthy_count += 1

        # API configuration check
        api_health = await self.check_api_configuration()
        components["api_configuration"] = api_health.to_dict()
        if api_health.status == HealthStatus.HEALTHY:
            healthy_count += 1
        elif api_health.status == HealthStatus.DEGRADED:
            degraded_count += 1
        else:
            unhealthy_count += 1

        # Disk space check
        disk_health = await self.check_disk_space()
        components["disk_space"] = disk_health.to_dict()
        if disk_health.status == HealthStatus.HEALTHY:
            healthy_count += 1
        elif disk_health.status == HealthStatus.DEGRADED:
            degraded_count += 1
        else:
            unhealthy_count += 1

        # Determine overall status
        if unhealthy_count > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        status_code = 200 if overall_status != HealthStatus.UNHEALTHY else 503

        response = {
            "status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "uptime_seconds": round(self.get_uptime_seconds(), 2),
            "components": components,
            "summary": {
                "healthy": healthy_count,
                "degraded": degraded_count,
                "unhealthy": unhealthy_count,
                "total": healthy_count + degraded_count + unhealthy_count,
            },
        }

        # Cache the result
        self._last_detailed_check = response
        self._last_check_time = now

        return response, status_code

    def _update_counts(
        self,
        status: HealthStatus,
        healthy: int,
        degraded: int,
        unhealthy: int
    ) -> tuple[int, int, int]:
        """Update health counts based on status"""
        if status == HealthStatus.HEALTHY:
            return healthy + 1, degraded, unhealthy
        elif status == HealthStatus.DEGRADED:
            return healthy, degraded + 1, unhealthy
        else:
            return healthy, degraded, unhealthy + 1

    async def check_database(self) -> ComponentHealth:
        """
        Check SQLite database health

        PATTERN: Connection test with query execution
        WHY: Verify both connectivity and query capability

        Checks:
        - Database file exists
        - Connection can be established
        - Simple query executes successfully
        - Query latency is acceptable
        """
        start_time = time.time()
        db_path = settings.database_url.replace("sqlite:///", "")

        # Handle relative path
        if db_path.startswith("./"):
            db_path = db_path[2:]

        try:
            # Check if database file exists (or can be created)
            db_dir = os.path.dirname(db_path) if os.path.dirname(db_path) else "."

            async with aiosqlite.connect(db_path) as conn:
                # Execute test query
                cursor = await conn.execute("SELECT 1")
                await cursor.fetchone()

                # Get database stats
                cursor = await conn.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
                )
                table_count = (await cursor.fetchone())[0]

            latency_ms = (time.time() - start_time) * 1000

            # Check latency threshold
            if latency_ms > 100:
                return ComponentHealth(
                    name="database",
                    status=HealthStatus.DEGRADED,
                    latency_ms=latency_ms,
                    message="High latency detected",
                    details={
                        "path": db_path,
                        "table_count": table_count,
                        "threshold_ms": 100,
                    },
                )

            return ComponentHealth(
                name="database",
                status=HealthStatus.HEALTHY,
                latency_ms=latency_ms,
                details={
                    "path": db_path,
                    "table_count": table_count,
                },
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(
                "database_health_check_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            return ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                message=f"Database check failed: {str(e)}",
                details={"path": db_path, "error": str(e)},
            )

    async def check_chromadb(self) -> ComponentHealth:
        """
        Check ChromaDB vector store health

        PATTERN: Collection existence and document count
        WHY: Verify vector store is operational

        Checks:
        - ChromaDB persist directory exists
        - Collection can be accessed
        - Document count is retrievable
        """
        start_time = time.time()
        persist_dir = rag_config.chroma_persist_directory
        collection_name = rag_config.chroma_collection_name

        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            # Check persist directory
            if not os.path.exists(persist_dir):
                return ComponentHealth(
                    name="vector_store",
                    status=HealthStatus.DEGRADED,
                    latency_ms=(time.time() - start_time) * 1000,
                    message="Persist directory does not exist (will be created on first use)",
                    details={
                        "persist_directory": persist_dir,
                        "collection_name": collection_name,
                    },
                )

            # Initialize client (this is synchronous but fast)
            client = chromadb.Client(
                ChromaSettings(
                    persist_directory=persist_dir,
                    anonymized_telemetry=False,
                )
            )

            # Try to get collection
            try:
                collection = client.get_collection(collection_name)
                doc_count = collection.count()
            except Exception:
                # Collection doesn't exist yet - this is okay
                doc_count = 0

            latency_ms = (time.time() - start_time) * 1000

            return ComponentHealth(
                name="vector_store",
                status=HealthStatus.HEALTHY,
                latency_ms=latency_ms,
                details={
                    "persist_directory": persist_dir,
                    "collection_name": collection_name,
                    "document_count": doc_count,
                },
            )

        except ImportError:
            return ComponentHealth(
                name="vector_store",
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start_time) * 1000,
                message="ChromaDB not installed",
                details={"error": "chromadb package not available"},
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(
                "chromadb_health_check_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            return ComponentHealth(
                name="vector_store",
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                message=f"ChromaDB check failed: {str(e)}",
                details={
                    "persist_directory": persist_dir,
                    "error": str(e),
                },
            )

    async def check_api_configuration(self) -> ComponentHealth:
        """
        Check API key configuration

        PATTERN: Configuration validation without key exposure
        WHY: Ensure required services are configured

        Checks:
        - Anthropic API key is set
        - OpenAI API key is set (for embeddings)
        - Keys are not placeholder values
        """
        start_time = time.time()
        details = {}
        issues = []

        # Check Anthropic API key
        anthropic_key = settings.anthropic_api_key
        if not anthropic_key:
            issues.append("Anthropic API key not configured")
            details["anthropic_api"] = "not_configured"
        elif anthropic_key.startswith("sk-") and len(anthropic_key) > 20:
            details["anthropic_api"] = "configured"
        else:
            issues.append("Anthropic API key appears invalid")
            details["anthropic_api"] = "invalid_format"

        # Check OpenAI API key (for embeddings)
        openai_key = settings.openai_api_key
        if not openai_key:
            issues.append("OpenAI API key not configured (embeddings disabled)")
            details["openai_api"] = "not_configured"
        elif openai_key.startswith("sk-") and len(openai_key) > 20:
            details["openai_api"] = "configured"
        else:
            issues.append("OpenAI API key appears invalid")
            details["openai_api"] = "invalid_format"

        latency_ms = (time.time() - start_time) * 1000

        # Determine status
        if details.get("anthropic_api") != "configured":
            return ComponentHealth(
                name="api_configuration",
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                message="; ".join(issues),
                details=details,
            )
        elif details.get("openai_api") != "configured":
            return ComponentHealth(
                name="api_configuration",
                status=HealthStatus.DEGRADED,
                latency_ms=latency_ms,
                message="; ".join(issues),
                details=details,
            )

        return ComponentHealth(
            name="api_configuration",
            status=HealthStatus.HEALTHY,
            latency_ms=latency_ms,
            details=details,
        )

    async def check_disk_space(self) -> ComponentHealth:
        """
        Check disk space for uploads and data storage

        PATTERN: Resource monitoring
        WHY: Prevent failures due to disk exhaustion

        Thresholds:
        - Healthy: > 1GB free
        - Degraded: 500MB - 1GB free
        - Unhealthy: < 500MB free
        """
        start_time = time.time()

        try:
            # Check the data directory disk space
            data_path = rag_config.chroma_persist_directory
            if not os.path.exists(data_path):
                data_path = "."

            disk_usage = shutil.disk_usage(data_path)
            free_gb = disk_usage.free / (1024 ** 3)
            total_gb = disk_usage.total / (1024 ** 3)
            used_percent = (disk_usage.used / disk_usage.total) * 100

            latency_ms = (time.time() - start_time) * 1000

            details = {
                "free_gb": round(free_gb, 2),
                "total_gb": round(total_gb, 2),
                "used_percent": round(used_percent, 1),
                "path": data_path,
            }

            # Determine status based on free space
            if free_gb < 0.5:  # Less than 500MB
                return ComponentHealth(
                    name="disk_space",
                    status=HealthStatus.UNHEALTHY,
                    latency_ms=latency_ms,
                    message=f"Critical: Only {free_gb:.2f}GB free",
                    details=details,
                )
            elif free_gb < 1.0:  # Less than 1GB
                return ComponentHealth(
                    name="disk_space",
                    status=HealthStatus.DEGRADED,
                    latency_ms=latency_ms,
                    message=f"Warning: Only {free_gb:.2f}GB free",
                    details=details,
                )

            return ComponentHealth(
                name="disk_space",
                status=HealthStatus.HEALTHY,
                latency_ms=latency_ms,
                details=details,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(
                "disk_space_check_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            return ComponentHealth(
                name="disk_space",
                status=HealthStatus.UNKNOWN,
                latency_ms=latency_ms,
                message=f"Could not check disk space: {str(e)}",
                details={"error": str(e)},
            )


# Global health checker instance
health_checker = HealthChecker()
