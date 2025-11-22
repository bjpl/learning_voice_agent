#!/bin/bash
#
# healthcheck.sh - Health check script for Learning Voice Agent
#
# Usage: ./scripts/healthcheck.sh [--json] [--verbose]
#
# Options:
#   --json      Output in JSON format
#   --verbose   Show detailed information
#   --quiet     Only output on failure
#
# Exit codes:
#   0 - All checks passed (healthy)
#   1 - Critical failure (app not responding)
#   2 - Warning (degraded performance)
#   3 - Configuration error

set -uo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
APP_HOST="${HOST:-localhost}"
APP_PORT="${PORT:-8000}"
HEALTH_ENDPOINT="http://${APP_HOST}:${APP_PORT}/api/health"

# Thresholds
DISK_WARNING_THRESHOLD=80
DISK_CRITICAL_THRESHOLD=90
MEMORY_WARNING_THRESHOLD=80
MEMORY_CRITICAL_THRESHOLD=90
DB_SIZE_WARNING_MB=500

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
JSON_OUTPUT=false
VERBOSE=false
QUIET=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --json|-j)
            JSON_OUTPUT=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --quiet|-q)
            QUIET=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--json] [--verbose] [--quiet]"
            echo ""
            echo "Options:"
            echo "  --json, -j     Output in JSON format"
            echo "  --verbose, -v  Show detailed information"
            echo "  --quiet, -q    Only output on failure"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

# Global status tracking
OVERALL_STATUS="healthy"
WARNINGS=()
ERRORS=()
CHECKS=()

# Output functions
output() {
    if [ "$QUIET" = false ]; then
        echo -e "$1"
    fi
}

# Add check result
add_check() {
    local name="$1"
    local status="$2"
    local message="$3"
    local details="${4:-}"

    CHECKS+=("{\"name\":\"$name\",\"status\":\"$status\",\"message\":\"$message\",\"details\":\"$details\"}")

    if [ "$status" = "critical" ]; then
        OVERALL_STATUS="unhealthy"
        ERRORS+=("$name: $message")
    elif [ "$status" = "warning" ]; then
        if [ "$OVERALL_STATUS" != "unhealthy" ]; then
            OVERALL_STATUS="degraded"
        fi
        WARNINGS+=("$name: $message")
    fi
}

# Check if application is responding
check_app_health() {
    output "${BLUE}[CHECK]${NC} Application health..."

    if ! command -v curl &> /dev/null; then
        add_check "app_health" "warning" "curl not installed, cannot check HTTP endpoint" ""
        return
    fi

    local response
    local http_code
    local start_time
    local end_time
    local response_time

    start_time=$(date +%s%N)

    # Try to reach health endpoint with timeout
    response=$(curl -s -w "\n%{http_code}" --connect-timeout 5 --max-time 10 "$HEALTH_ENDPOINT" 2>/dev/null) || {
        add_check "app_health" "critical" "Application not responding at $HEALTH_ENDPOINT" ""
        output "${RED}[FAIL]${NC} Application not responding"
        return
    }

    end_time=$(date +%s%N)
    response_time=$(( (end_time - start_time) / 1000000 ))

    http_code=$(echo "$response" | tail -n1)
    local body
    body=$(echo "$response" | sed '$d')

    if [ "$http_code" = "200" ]; then
        add_check "app_health" "healthy" "Application responding" "response_time_ms: $response_time"
        output "${GREEN}[OK]${NC} Application responding (${response_time}ms)"

        # Check response content
        if command -v python3 &> /dev/null && [ -n "$body" ]; then
            local app_status
            app_status=$(echo "$body" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','unknown'))" 2>/dev/null || echo "unknown")

            if [ "$app_status" = "degraded" ]; then
                add_check "app_dependencies" "warning" "Some dependencies degraded" ""
                output "${YELLOW}[WARN]${NC} Some dependencies are degraded"
            fi

            if [ "$VERBOSE" = true ]; then
                echo "$body" | python3 -m json.tool 2>/dev/null || echo "$body"
            fi
        fi
    elif [ "$http_code" = "503" ]; then
        add_check "app_health" "warning" "Application in degraded state" "http_code: $http_code"
        output "${YELLOW}[WARN]${NC} Application in degraded state"
    else
        add_check "app_health" "critical" "Unexpected HTTP status" "http_code: $http_code"
        output "${RED}[FAIL]${NC} Unexpected HTTP status: $http_code"
    fi
}

# Check disk space
check_disk_space() {
    output "${BLUE}[CHECK]${NC} Disk space..."

    local disk_usage
    local mount_point

    # Get disk usage for project root
    disk_usage=$(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $5}' | tr -d '%')
    mount_point=$(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $6}')

    if [ -z "$disk_usage" ]; then
        add_check "disk_space" "warning" "Could not determine disk usage" ""
        return
    fi

    local available
    available=$(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $4}')

    if [ "$disk_usage" -ge "$DISK_CRITICAL_THRESHOLD" ]; then
        add_check "disk_space" "critical" "Disk usage critical: ${disk_usage}%" "mount: $mount_point, available: $available"
        output "${RED}[FAIL]${NC} Disk usage critical: ${disk_usage}% (${available} available)"
    elif [ "$disk_usage" -ge "$DISK_WARNING_THRESHOLD" ]; then
        add_check "disk_space" "warning" "Disk usage high: ${disk_usage}%" "mount: $mount_point, available: $available"
        output "${YELLOW}[WARN]${NC} Disk usage high: ${disk_usage}% (${available} available)"
    else
        add_check "disk_space" "healthy" "Disk usage OK: ${disk_usage}%" "mount: $mount_point, available: $available"
        output "${GREEN}[OK]${NC} Disk usage: ${disk_usage}% (${available} available)"
    fi
}

# Check memory usage
check_memory() {
    output "${BLUE}[CHECK]${NC} Memory usage..."

    local mem_info
    local mem_total
    local mem_available
    local mem_used_percent

    if [ -f /proc/meminfo ]; then
        mem_total=$(grep MemTotal /proc/meminfo | awk '{print $2}')
        mem_available=$(grep MemAvailable /proc/meminfo | awk '{print $2}')

        if [ -n "$mem_total" ] && [ -n "$mem_available" ] && [ "$mem_total" -gt 0 ]; then
            mem_used_percent=$(( (mem_total - mem_available) * 100 / mem_total ))

            local mem_total_gb
            local mem_available_gb
            mem_total_gb=$(echo "scale=1; $mem_total / 1024 / 1024" | bc 2>/dev/null || echo "N/A")
            mem_available_gb=$(echo "scale=1; $mem_available / 1024 / 1024" | bc 2>/dev/null || echo "N/A")

            if [ "$mem_used_percent" -ge "$MEMORY_CRITICAL_THRESHOLD" ]; then
                add_check "memory" "critical" "Memory usage critical: ${mem_used_percent}%" "total: ${mem_total_gb}GB, available: ${mem_available_gb}GB"
                output "${RED}[FAIL]${NC} Memory usage critical: ${mem_used_percent}%"
            elif [ "$mem_used_percent" -ge "$MEMORY_WARNING_THRESHOLD" ]; then
                add_check "memory" "warning" "Memory usage high: ${mem_used_percent}%" "total: ${mem_total_gb}GB, available: ${mem_available_gb}GB"
                output "${YELLOW}[WARN]${NC} Memory usage high: ${mem_used_percent}%"
            else
                add_check "memory" "healthy" "Memory usage OK: ${mem_used_percent}%" "total: ${mem_total_gb}GB, available: ${mem_available_gb}GB"
                output "${GREEN}[OK]${NC} Memory usage: ${mem_used_percent}% (${mem_available_gb}GB available)"
            fi
        else
            add_check "memory" "warning" "Could not parse memory info" ""
        fi
    elif command -v free &> /dev/null; then
        local mem_line
        mem_line=$(free | grep Mem)
        mem_total=$(echo "$mem_line" | awk '{print $2}')
        mem_available=$(echo "$mem_line" | awk '{print $7}')

        if [ -n "$mem_total" ] && [ "$mem_total" -gt 0 ]; then
            mem_used_percent=$(( (mem_total - mem_available) * 100 / mem_total ))
            add_check "memory" "healthy" "Memory usage: ${mem_used_percent}%" ""
            output "${GREEN}[OK]${NC} Memory usage: ${mem_used_percent}%"
        fi
    else
        add_check "memory" "warning" "Could not determine memory usage" ""
        output "${YELLOW}[WARN]${NC} Could not determine memory usage"
    fi
}

# Check database
check_database() {
    output "${BLUE}[CHECK]${NC} Database..."

    local db_file="${PROJECT_ROOT}/learning_captures.db"

    if [ ! -f "$db_file" ]; then
        add_check "database" "warning" "Database file not found" "path: $db_file"
        output "${YELLOW}[WARN]${NC} Database file not found"
        return
    fi

    # Check file size
    local db_size_bytes
    local db_size_mb
    db_size_bytes=$(stat -f%z "$db_file" 2>/dev/null || stat -c%s "$db_file" 2>/dev/null || echo "0")
    db_size_mb=$(( db_size_bytes / 1024 / 1024 ))

    # Check if database is readable
    if command -v sqlite3 &> /dev/null; then
        local integrity_check
        integrity_check=$(sqlite3 "$db_file" "PRAGMA integrity_check;" 2>&1)

        if [ "$integrity_check" = "ok" ]; then
            if [ "$db_size_mb" -gt "$DB_SIZE_WARNING_MB" ]; then
                add_check "database" "warning" "Database large: ${db_size_mb}MB" "integrity: ok"
                output "${YELLOW}[WARN]${NC} Database integrity OK but large (${db_size_mb}MB)"
            else
                add_check "database" "healthy" "Database OK: ${db_size_mb}MB" "integrity: ok"
                output "${GREEN}[OK]${NC} Database healthy (${db_size_mb}MB)"
            fi

            if [ "$VERBOSE" = true ]; then
                # Get table stats
                local table_count
                table_count=$(sqlite3 "$db_file" "SELECT COUNT(*) FROM sqlite_master WHERE type='table';" 2>/dev/null || echo "unknown")
                output "  Tables: $table_count"

                # Get record count
                local record_count
                record_count=$(sqlite3 "$db_file" "SELECT COUNT(*) FROM captures;" 2>/dev/null || echo "unknown")
                output "  Records in captures: $record_count"
            fi
        else
            add_check "database" "critical" "Database integrity check failed" "error: $integrity_check"
            output "${RED}[FAIL]${NC} Database integrity check failed"
        fi
    else
        # Just check if file exists and is readable
        if [ -r "$db_file" ]; then
            add_check "database" "healthy" "Database file exists: ${db_size_mb}MB" "sqlite3 not available for integrity check"
            output "${GREEN}[OK]${NC} Database file exists (${db_size_mb}MB)"
        else
            add_check "database" "critical" "Database file not readable" ""
            output "${RED}[FAIL]${NC} Database file not readable"
        fi
    fi
}

# Check process status
check_process() {
    output "${BLUE}[CHECK]${NC} Application process..."

    local pid_file="${PROJECT_ROOT}/app.pid"

    if [ -f "$pid_file" ]; then
        local pid
        pid=$(cat "$pid_file")

        if kill -0 "$pid" 2>/dev/null; then
            # Get process info
            local mem_usage
            local cpu_usage

            if command -v ps &> /dev/null; then
                mem_usage=$(ps -o rss= -p "$pid" 2>/dev/null | awk '{print int($1/1024)"MB"}' || echo "unknown")
                cpu_usage=$(ps -o %cpu= -p "$pid" 2>/dev/null | tr -d ' ' || echo "unknown")
            fi

            add_check "process" "healthy" "Application running" "pid: $pid, memory: $mem_usage, cpu: $cpu_usage%"
            output "${GREEN}[OK]${NC} Application running (PID: $pid, Memory: $mem_usage)"
        else
            add_check "process" "warning" "PID file exists but process not running" "stale pid: $pid"
            output "${YELLOW}[WARN]${NC} Stale PID file found (PID: $pid)"
        fi
    else
        # Check for uvicorn process anyway
        local uvicorn_pid
        uvicorn_pid=$(pgrep -f "uvicorn app.main:app" 2>/dev/null | head -1 || true)

        if [ -n "$uvicorn_pid" ]; then
            add_check "process" "healthy" "Application running (no PID file)" "pid: $uvicorn_pid"
            output "${GREEN}[OK]${NC} Application running (PID: $uvicorn_pid, no PID file)"
        else
            add_check "process" "warning" "No application process found" ""
            output "${YELLOW}[WARN]${NC} No application process found"
        fi
    fi
}

# Check log files
check_logs() {
    output "${BLUE}[CHECK]${NC} Log files..."

    local log_dir="${PROJECT_ROOT}/logs"

    if [ -d "$log_dir" ]; then
        local log_size
        log_size=$(du -sh "$log_dir" 2>/dev/null | cut -f1 || echo "unknown")

        # Check for recent errors in logs
        local recent_errors=0
        if [ -f "${log_dir}/app.log" ]; then
            recent_errors=$(tail -100 "${log_dir}/app.log" 2>/dev/null | grep -ci "error" || echo "0")
        fi

        if [ "$recent_errors" -gt 10 ]; then
            add_check "logs" "warning" "Multiple recent errors in logs" "errors: $recent_errors, size: $log_size"
            output "${YELLOW}[WARN]${NC} Multiple recent errors in logs ($recent_errors)"
        else
            add_check "logs" "healthy" "Logs OK" "size: $log_size, recent_errors: $recent_errors"
            output "${GREEN}[OK]${NC} Logs healthy ($log_size)"
        fi
    else
        add_check "logs" "healthy" "No log directory" ""
        output "${GREEN}[OK]${NC} No log directory (may use stdout)"
    fi
}

# Output JSON result
output_json() {
    local checks_json
    checks_json=$(printf '%s,' "${CHECKS[@]}")
    checks_json="[${checks_json%,}]"

    cat << EOF
{
    "status": "$OVERALL_STATUS",
    "timestamp": "$(date -Iseconds)",
    "host": "$(hostname)",
    "checks": $checks_json,
    "warnings_count": ${#WARNINGS[@]},
    "errors_count": ${#ERRORS[@]}
}
EOF
}

# Output summary
output_summary() {
    echo ""
    echo "=========================================="

    if [ "$OVERALL_STATUS" = "healthy" ]; then
        echo -e "${GREEN}OVERALL STATUS: HEALTHY${NC}"
    elif [ "$OVERALL_STATUS" = "degraded" ]; then
        echo -e "${YELLOW}OVERALL STATUS: DEGRADED${NC}"
    else
        echo -e "${RED}OVERALL STATUS: UNHEALTHY${NC}"
    fi

    if [ ${#WARNINGS[@]} -gt 0 ]; then
        echo ""
        echo "Warnings:"
        for warning in "${WARNINGS[@]}"; do
            echo "  - $warning"
        done
    fi

    if [ ${#ERRORS[@]} -gt 0 ]; then
        echo ""
        echo "Errors:"
        for error in "${ERRORS[@]}"; do
            echo "  - $error"
        done
    fi

    echo "=========================================="
}

# Main function
main() {
    if [ "$QUIET" = false ]; then
        echo "=========================================="
        echo "Health Check - Learning Voice Agent"
        echo "Time: $(date)"
        echo "=========================================="
        echo ""
    fi

    # Run all checks
    check_app_health
    check_process
    check_disk_space
    check_memory
    check_database
    check_logs

    # Output results
    if [ "$JSON_OUTPUT" = true ]; then
        output_json
    else
        output_summary
    fi

    # Return appropriate exit code
    case "$OVERALL_STATUS" in
        healthy)
            exit 0
            ;;
        degraded)
            exit 2
            ;;
        unhealthy)
            exit 1
            ;;
        *)
            exit 3
            ;;
    esac
}

# Run main
main "$@"
