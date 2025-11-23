"""Admin Dashboard Module"""
from .dashboard import router as admin_router
from .metrics import MetricsCollector

__all__ = ["admin_router", "MetricsCollector"]
