#!/bin/bash
#
# Load Test Runner for Learning Voice Agent
#
# Usage:
#   ./scripts/run_load_tests.sh                    # Default: smoke test
#   ./scripts/run_load_tests.sh --users 100        # Custom user count
#   ./scripts/run_load_tests.sh --scenario stress  # Run stress test
#   ./scripts/run_load_tests.sh --help             # Show help
#
# Performance Targets:
#   - Response time: p95 < 500ms
#   - Success rate: > 99.5%
#   - No memory leaks over 10 min run
#   - Rate limiting works under load
#

set -e

# =============================================================================
# Configuration Defaults
# =============================================================================

DEFAULT_HOST="http://localhost:8000"
DEFAULT_USERS=10
DEFAULT_SPAWN_RATE=2
DEFAULT_DURATION="60s"
DEFAULT_SCENARIO="smoke"
LOCUST_FILE="tests/performance/locustfile.py"

# Output directories
OUTPUT_DIR="reports/load_tests"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# =============================================================================
# Color Output
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =============================================================================
# Help
# =============================================================================

show_help() {
    cat << EOF
Load Test Runner for Learning Voice Agent

Usage: $0 [OPTIONS]

Options:
    -h, --host HOST         Target host URL (default: $DEFAULT_HOST)
    -u, --users NUM         Number of concurrent users (default: $DEFAULT_USERS)
    -r, --spawn-rate NUM    Users spawned per second (default: $DEFAULT_SPAWN_RATE)
    -d, --duration TIME     Test duration (e.g., 60s, 5m, 1h) (default: $DEFAULT_DURATION)
    -s, --scenario NAME     Test scenario (default: $DEFAULT_SCENARIO)
    -t, --tags TAGS         Only run tasks with these tags (comma-separated)
    -w, --web               Run with web UI (default: headless)
    -o, --output DIR        Output directory (default: $OUTPUT_DIR)
    --help                  Show this help message

Scenarios:
    smoke       Quick validation (10 users, 1 min)
    load        Normal capacity (100 users, 10 min)
    stress      Target capacity (1000 users, 15 min)
    spike       Sudden traffic (500 users, instant spawn, 5 min)
    endurance   Long-running (200 users, 30 min)
    auth-only   Authentication endpoints only (50 users, 5 min)
    gdpr-only   GDPR endpoints only (20 users, 5 min)
    rate-limit  Rate limiting test (100 users, rapid requests, 2 min)
    custom      Use provided --users, --spawn-rate, --duration

Tag Options:
    health          Health check endpoints
    authenticated   Endpoints requiring auth
    unauthenticated Public endpoints
    conversation    Conversation API
    search          Search functionality
    gdpr            GDPR compliance endpoints
    rate-limit      Rate limiting tests
    websocket       WebSocket endpoints

Examples:
    # Smoke test (quick validation)
    $0 --scenario smoke

    # Load test with 100 users
    $0 --scenario load --host https://staging.example.com

    # Stress test (1000 concurrent users)
    $0 --scenario stress

    # Custom configuration
    $0 --users 200 --spawn-rate 20 --duration 10m

    # Test only authentication endpoints
    $0 --tags authenticated --users 50 --duration 5m

    # Run with web UI for debugging
    $0 --web --users 10

    # CI/CD integration (outputs JSON metrics)
    $0 --scenario load --output /tmp/load_results

EOF
}

# =============================================================================
# Parse Arguments
# =============================================================================

HOST="$DEFAULT_HOST"
USERS="$DEFAULT_USERS"
SPAWN_RATE="$DEFAULT_SPAWN_RATE"
DURATION="$DEFAULT_DURATION"
SCENARIO="$DEFAULT_SCENARIO"
TAGS=""
WEB_UI=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--host)
            HOST="$2"
            shift 2
            ;;
        -u|--users)
            USERS="$2"
            shift 2
            ;;
        -r|--spawn-rate)
            SPAWN_RATE="$2"
            shift 2
            ;;
        -d|--duration)
            DURATION="$2"
            shift 2
            ;;
        -s|--scenario)
            SCENARIO="$2"
            shift 2
            ;;
        -t|--tags)
            TAGS="$2"
            shift 2
            ;;
        -w|--web)
            WEB_UI=true
            shift
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# =============================================================================
# Apply Scenario Presets
# =============================================================================

apply_scenario() {
    case $SCENARIO in
        smoke)
            USERS=10
            SPAWN_RATE=2
            DURATION="60s"
            ;;
        load)
            USERS=100
            SPAWN_RATE=10
            DURATION="10m"
            ;;
        stress)
            USERS=1000
            SPAWN_RATE=50
            DURATION="15m"
            ;;
        spike)
            USERS=500
            SPAWN_RATE=500
            DURATION="5m"
            ;;
        endurance)
            USERS=200
            SPAWN_RATE=10
            DURATION="30m"
            ;;
        auth-only)
            USERS=50
            SPAWN_RATE=10
            DURATION="5m"
            TAGS="authenticated"
            ;;
        gdpr-only)
            USERS=20
            SPAWN_RATE=5
            DURATION="5m"
            TAGS="gdpr"
            ;;
        rate-limit)
            USERS=100
            SPAWN_RATE=100
            DURATION="2m"
            TAGS="rate-limit"
            ;;
        custom)
            # Use provided values
            ;;
        *)
            log_error "Unknown scenario: $SCENARIO"
            log_info "Available scenarios: smoke, load, stress, spike, endurance, auth-only, gdpr-only, rate-limit, custom"
            exit 1
            ;;
    esac
}

apply_scenario

# =============================================================================
# Pre-flight Checks
# =============================================================================

preflight_checks() {
    log_info "Running pre-flight checks..."

    # Check if locust is installed
    if ! command -v locust &> /dev/null; then
        log_error "Locust is not installed. Install with: pip install locust"
        exit 1
    fi

    # Check if locustfile exists
    if [ ! -f "$LOCUST_FILE" ]; then
        log_error "Locust file not found: $LOCUST_FILE"
        exit 1
    fi

    # Check if target host is reachable
    log_info "Checking target host: $HOST"
    if ! curl -s -o /dev/null -w "%{http_code}" "$HOST/health" | grep -q "200"; then
        log_warn "Target host may not be reachable or healthy: $HOST"
        log_warn "Continuing anyway..."
    else
        log_success "Target host is healthy"
    fi

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    log_success "Pre-flight checks passed"
}

# =============================================================================
# Run Load Test
# =============================================================================

run_load_test() {
    local report_file="$OUTPUT_DIR/load_test_${SCENARIO}_${TIMESTAMP}.html"
    local csv_prefix="$OUTPUT_DIR/load_test_${SCENARIO}_${TIMESTAMP}"
    local json_file="$OUTPUT_DIR/load_test_${SCENARIO}_${TIMESTAMP}_metrics.json"

    log_info "========================================"
    log_info "Starting Load Test"
    log_info "========================================"
    log_info "Scenario:    $SCENARIO"
    log_info "Host:        $HOST"
    log_info "Users:       $USERS"
    log_info "Spawn Rate:  $SPAWN_RATE/s"
    log_info "Duration:    $DURATION"
    log_info "Tags:        ${TAGS:-all}"
    log_info "Output:      $OUTPUT_DIR"
    log_info "========================================"

    # Build locust command
    local cmd="locust -f $LOCUST_FILE --host $HOST"

    if [ "$WEB_UI" = true ]; then
        log_info "Starting Locust with Web UI..."
        log_info "Open http://localhost:8089 in your browser"
        $cmd
    else
        cmd="$cmd --headless"
        cmd="$cmd --users $USERS"
        cmd="$cmd --spawn-rate $SPAWN_RATE"
        cmd="$cmd --run-time $DURATION"
        cmd="$cmd --html $report_file"
        cmd="$cmd --csv $csv_prefix"

        # Add tags if specified
        if [ -n "$TAGS" ]; then
            cmd="$cmd --tags $TAGS"
        fi

        # Add CSV stats interval
        cmd="$cmd --csv-full-history"

        log_info "Running: $cmd"
        echo ""

        # Run the test
        eval $cmd

        # Move the JSON metrics file if it was created
        if [ -f "load_test_metrics.json" ]; then
            mv load_test_metrics.json "$json_file"
            log_success "Metrics exported to: $json_file"
        fi
    fi
}

# =============================================================================
# Post-Test Analysis
# =============================================================================

analyze_results() {
    local json_file="$OUTPUT_DIR/load_test_${SCENARIO}_${TIMESTAMP}_metrics.json"

    if [ ! -f "$json_file" ]; then
        log_warn "Metrics file not found, skipping analysis"
        return
    fi

    log_info "========================================"
    log_info "Post-Test Analysis"
    log_info "========================================"

    # Parse JSON and check targets
    if command -v python3 &> /dev/null; then
        python3 << EOF
import json
import sys

try:
    with open("$json_file") as f:
        data = json.load(f)

    targets = data.get("targets", {})
    passed = True

    print("\nTarget Validation:")
    print("-" * 40)

    for target_name, target_data in targets.items():
        target = target_data.get("target")
        actual = target_data.get("actual")
        target_passed = target_data.get("passed", False)

        status = "PASS" if target_passed else "FAIL"
        symbol = "✓" if target_passed else "✗"

        print(f"  [{symbol}] {target_name}: {actual:.2f} (target: {target})")

        if not target_passed:
            passed = False

    print("-" * 40)

    if passed:
        print("\n[SUCCESS] All performance targets met!")
        sys.exit(0)
    else:
        print("\n[FAILURE] Some performance targets were not met")
        sys.exit(1)

except Exception as e:
    print(f"Error analyzing results: {e}")
    sys.exit(1)
EOF
        local exit_code=$?
        return $exit_code
    else
        log_warn "Python not found, skipping detailed analysis"
    fi
}

# =============================================================================
# Cleanup
# =============================================================================

cleanup() {
    log_info "Cleaning up temporary files..."
    # Remove any temporary files created during test
    rm -f locust_*.csv 2>/dev/null || true
}

# =============================================================================
# Main
# =============================================================================

main() {
    trap cleanup EXIT

    echo ""
    echo "========================================"
    echo "  Learning Voice Agent - Load Testing"
    echo "========================================"
    echo ""

    preflight_checks
    run_load_test

    if [ "$WEB_UI" = false ]; then
        analyze_results
        local result=$?

        echo ""
        log_info "========================================"
        log_info "Test Complete"
        log_info "========================================"
        log_info "HTML Report: $OUTPUT_DIR/load_test_${SCENARIO}_${TIMESTAMP}.html"
        log_info "CSV Stats:   $OUTPUT_DIR/load_test_${SCENARIO}_${TIMESTAMP}_stats.csv"
        log_info "Metrics:     $OUTPUT_DIR/load_test_${SCENARIO}_${TIMESTAMP}_metrics.json"
        log_info "========================================"

        exit $result
    fi
}

main
