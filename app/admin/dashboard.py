"""
Admin Dashboard Routes
Provides monitoring, metrics, and system management endpoints.
"""
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse
from typing import Dict, Optional
from datetime import datetime
import json

from .metrics import metrics_collector, MetricsCollector

router = APIRouter(prefix="/admin", tags=["admin"])


# Simple API key authentication for admin routes
async def verify_admin_key(request: Request):
    """Verify admin API key from header or query param."""
    api_key = request.headers.get("X-Admin-Key") or request.query_params.get("admin_key")

    # In production, this should be a secure environment variable
    # For development, allow access without key
    import os
    expected_key = os.getenv("ADMIN_API_KEY", "")

    if expected_key and api_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid admin key")

    return True


@router.get("/dashboard", response_class=HTMLResponse)
async def admin_dashboard(request: Request, _: bool = Depends(verify_admin_key)):
    """
    Render admin dashboard HTML.
    Displays real-time metrics and system status.
    """
    html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Learning Voice Agent - Admin Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            min-height: 100vh;
        }
        .header {
            background: #1e293b;
            padding: 1rem 2rem;
            border-bottom: 1px solid #334155;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 { font-size: 1.5rem; color: #38bdf8; }
        .status-badge {
            padding: 0.5rem 1rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 600;
        }
        .status-healthy { background: #166534; color: #86efac; }
        .status-warning { background: #854d0e; color: #fde047; }
        .status-error { background: #991b1b; color: #fca5a5; }

        .container { padding: 2rem; max-width: 1400px; margin: 0 auto; }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }

        .card {
            background: #1e293b;
            border-radius: 0.75rem;
            padding: 1.5rem;
            border: 1px solid #334155;
        }
        .card-title {
            font-size: 0.875rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 1rem;
        }
        .card-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: #f8fafc;
        }
        .card-subtitle {
            font-size: 0.875rem;
            color: #64748b;
            margin-top: 0.5rem;
        }

        .metric-row {
            display: flex;
            justify-content: space-between;
            padding: 0.75rem 0;
            border-bottom: 1px solid #334155;
        }
        .metric-row:last-child { border-bottom: none; }
        .metric-label { color: #94a3b8; }
        .metric-value { font-weight: 600; }

        .progress-bar {
            height: 8px;
            background: #334155;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 0.5rem;
        }
        .progress-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        .progress-green { background: #22c55e; }
        .progress-yellow { background: #eab308; }
        .progress-red { background: #ef4444; }

        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            text-align: left;
            padding: 0.75rem;
            border-bottom: 1px solid #334155;
        }
        th { color: #94a3b8; font-weight: 600; font-size: 0.875rem; }

        .refresh-btn {
            background: #3b82f6;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            cursor: pointer;
            font-size: 0.875rem;
        }
        .refresh-btn:hover { background: #2563eb; }

        .error-badge {
            background: #7f1d1d;
            color: #fca5a5;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            font-size: 0.75rem;
        }

        #last-updated { font-size: 0.75rem; color: #64748b; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Learning Voice Agent Dashboard</h1>
        <div style="display: flex; align-items: center; gap: 1rem;">
            <span id="last-updated">Loading...</span>
            <span id="status-badge" class="status-badge status-healthy">Healthy</span>
            <button class="refresh-btn" onclick="refreshData()">Refresh</button>
        </div>
    </div>

    <div class="container">
        <!-- Overview Cards -->
        <div class="grid">
            <div class="card">
                <div class="card-title">Total Requests</div>
                <div class="card-value" id="total-requests">-</div>
                <div class="card-subtitle" id="rps">- requests/sec</div>
            </div>
            <div class="card">
                <div class="card-title">Error Rate</div>
                <div class="card-value" id="error-rate">-</div>
                <div class="card-subtitle" id="total-errors">- total errors</div>
            </div>
            <div class="card">
                <div class="card-title">P95 Latency</div>
                <div class="card-value" id="p95-latency">-</div>
                <div class="card-subtitle" id="avg-latency">Avg: -ms</div>
            </div>
            <div class="card">
                <div class="card-title">Uptime</div>
                <div class="card-value" id="uptime">-</div>
                <div class="card-subtitle" id="start-time">Started: -</div>
            </div>
        </div>

        <!-- System Resources -->
        <div class="grid">
            <div class="card">
                <div class="card-title">CPU Usage</div>
                <div id="cpu-percent" style="font-size: 1.5rem; font-weight: 600;">-</div>
                <div class="progress-bar">
                    <div id="cpu-bar" class="progress-fill progress-green" style="width: 0%"></div>
                </div>
            </div>
            <div class="card">
                <div class="card-title">Memory Usage</div>
                <div id="memory-percent" style="font-size: 1.5rem; font-weight: 600;">-</div>
                <div class="progress-bar">
                    <div id="memory-bar" class="progress-fill progress-green" style="width: 0%"></div>
                </div>
                <div class="card-subtitle" id="memory-detail">-</div>
            </div>
            <div class="card">
                <div class="card-title">Active Connections</div>
                <div id="connections" style="font-size: 1.5rem; font-weight: 600;">-</div>
            </div>
        </div>

        <!-- Endpoint Stats -->
        <div class="card" style="margin-bottom: 2rem;">
            <div class="card-title">Top Endpoints (Last 5 Minutes)</div>
            <table>
                <thead>
                    <tr>
                        <th>Endpoint</th>
                        <th>Method</th>
                        <th>Requests</th>
                        <th>Errors</th>
                        <th>Avg (ms)</th>
                        <th>P95 (ms)</th>
                    </tr>
                </thead>
                <tbody id="endpoint-table">
                    <tr><td colspan="6" style="text-align: center;">Loading...</td></tr>
                </tbody>
            </table>
        </div>

        <!-- Recent Errors -->
        <div class="card">
            <div class="card-title">Recent Errors</div>
            <div id="errors-list">
                <p style="color: #64748b; text-align: center; padding: 2rem;">No recent errors</p>
            </div>
        </div>
    </div>

    <script>
        let autoRefresh = true;

        async function refreshData() {
            try {
                const response = await fetch('/admin/api/metrics');
                const data = await response.json();
                updateDashboard(data);
            } catch (error) {
                console.error('Failed to fetch metrics:', error);
                document.getElementById('status-badge').className = 'status-badge status-error';
                document.getElementById('status-badge').textContent = 'Error';
            }
        }

        function updateDashboard(data) {
            // Update timestamp
            document.getElementById('last-updated').textContent =
                'Updated: ' + new Date().toLocaleTimeString();

            // Overview
            const overview = data.overview;
            document.getElementById('total-requests').textContent =
                overview.total_requests.toLocaleString();
            document.getElementById('error-rate').textContent =
                overview.error_rate.toFixed(2) + '%';
            document.getElementById('total-errors').textContent =
                overview.total_errors + ' total errors';

            // Format uptime
            const hours = Math.floor(overview.uptime_seconds / 3600);
            const mins = Math.floor((overview.uptime_seconds % 3600) / 60);
            document.getElementById('uptime').textContent =
                hours + 'h ' + mins + 'm';

            // Request stats
            const reqStats = data.request_stats;
            document.getElementById('rps').textContent =
                reqStats.requests_per_second.toFixed(2) + ' requests/sec';
            document.getElementById('p95-latency').textContent =
                reqStats.p95_ms.toFixed(0) + 'ms';
            document.getElementById('avg-latency').textContent =
                'Avg: ' + reqStats.avg_response_ms.toFixed(0) + 'ms';

            // System stats
            const sysStats = data.system_stats.current;
            document.getElementById('cpu-percent').textContent =
                sysStats.cpu_percent.toFixed(1) + '%';
            document.getElementById('memory-percent').textContent =
                sysStats.memory_percent.toFixed(1) + '%';
            document.getElementById('memory-detail').textContent =
                'Used: ' + sysStats.memory_used_mb.toFixed(0) + 'MB / Available: ' +
                sysStats.memory_available_mb.toFixed(0) + 'MB';
            document.getElementById('connections').textContent =
                sysStats.active_connections;

            // Update progress bars
            updateProgressBar('cpu-bar', sysStats.cpu_percent);
            updateProgressBar('memory-bar', sysStats.memory_percent);

            // Update status badge
            const statusBadge = document.getElementById('status-badge');
            if (overview.error_rate > 1) {
                statusBadge.className = 'status-badge status-error';
                statusBadge.textContent = 'Degraded';
            } else if (overview.error_rate > 0.1 || reqStats.p95_ms > 2000) {
                statusBadge.className = 'status-badge status-warning';
                statusBadge.textContent = 'Warning';
            } else {
                statusBadge.className = 'status-badge status-healthy';
                statusBadge.textContent = 'Healthy';
            }

            // Endpoint table
            const tbody = document.getElementById('endpoint-table');
            if (data.top_endpoints.length > 0) {
                tbody.innerHTML = data.top_endpoints.map(ep => `
                    <tr>
                        <td>${ep.endpoint}</td>
                        <td>${ep.method}</td>
                        <td>${ep.requests.toLocaleString()}</td>
                        <td>${ep.errors > 0 ? '<span class="error-badge">' + ep.errors + '</span>' : '0'}</td>
                        <td>${ep.avg_ms.toFixed(0)}</td>
                        <td>${ep.p95_ms.toFixed(0)}</td>
                    </tr>
                `).join('');
            } else {
                tbody.innerHTML = '<tr><td colspan="6" style="text-align: center;">No data</td></tr>';
            }

            // Recent errors
            const errorsList = document.getElementById('errors-list');
            if (data.recent_errors.length > 0) {
                errorsList.innerHTML = data.recent_errors.map(err => `
                    <div class="metric-row">
                        <span>${err.timestamp} - ${err.method} ${err.endpoint}</span>
                        <span class="error-badge">${err.status_code}</span>
                    </div>
                `).join('');
            } else {
                errorsList.innerHTML = '<p style="color: #64748b; text-align: center; padding: 2rem;">No recent errors</p>';
            }
        }

        function updateProgressBar(id, percent) {
            const bar = document.getElementById(id);
            bar.style.width = percent + '%';

            if (percent > 80) {
                bar.className = 'progress-fill progress-red';
            } else if (percent > 60) {
                bar.className = 'progress-fill progress-yellow';
            } else {
                bar.className = 'progress-fill progress-green';
            }
        }

        // Initial load
        refreshData();

        // Auto-refresh every 5 seconds
        setInterval(() => {
            if (autoRefresh) refreshData();
        }, 5000);
    </script>
</body>
</html>
    '''
    return HTMLResponse(content=html_content)


@router.get("/api/metrics")
async def get_metrics(_: bool = Depends(verify_admin_key)) -> Dict:
    """Get all dashboard metrics as JSON."""
    return metrics_collector.get_dashboard_summary()


@router.get("/api/metrics/requests")
async def get_request_metrics(
    window_minutes: int = 5,
    _: bool = Depends(verify_admin_key)
) -> Dict:
    """Get request-specific metrics."""
    return metrics_collector.get_request_stats(window_minutes)


@router.get("/api/metrics/system")
async def get_system_metrics(_: bool = Depends(verify_admin_key)) -> Dict:
    """Get system resource metrics."""
    return metrics_collector.get_system_stats()


@router.get("/api/metrics/endpoints")
async def get_endpoint_metrics(_: bool = Depends(verify_admin_key)) -> Dict:
    """Get per-endpoint metrics."""
    return {"endpoints": metrics_collector.get_endpoint_stats()}


@router.get("/api/metrics/errors")
async def get_error_metrics(
    limit: int = 50,
    _: bool = Depends(verify_admin_key)
) -> Dict:
    """Get recent errors."""
    return {"errors": metrics_collector.get_recent_errors(limit)}


@router.get("/api/health/detailed")
async def detailed_health_check(_: bool = Depends(verify_admin_key)) -> Dict:
    """
    Detailed health check for monitoring systems.
    Returns component-level health status.
    """
    from app.database import db
    from app.state_manager import state_manager

    health = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {}
    }

    # Check database
    try:
        stats = await db.get_stats()
        health["components"]["database"] = {
            "status": "healthy",
            "total_exchanges": stats.get("total_exchanges", 0)
        }
    except Exception as e:
        health["components"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health["status"] = "degraded"

    # Check Redis
    try:
        sessions = await state_manager.get_active_sessions()
        health["components"]["redis"] = {
            "status": "healthy",
            "active_sessions": len(sessions)
        }
    except Exception as e:
        health["components"]["redis"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health["status"] = "degraded"

    # System resources
    system = metrics_collector.get_system_stats()
    health["components"]["system"] = {
        "status": "healthy" if system["current"]["cpu_percent"] < 90 else "warning",
        "cpu_percent": system["current"]["cpu_percent"],
        "memory_percent": system["current"]["memory_percent"]
    }

    return health
