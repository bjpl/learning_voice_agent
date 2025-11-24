#!/bin/bash
#
# Complete Production Verification Suite
# Runs all verification checks for a production deployment.
#
# Usage:
#   ./scripts/deployment/verify_all.sh https://yourdomain.com
#   ./scripts/deployment/verify_all.sh https://yourdomain.com --full
#   ./scripts/deployment/verify_all.sh https://yourdomain.com --output /tmp/reports
#

set -e

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TARGET_URL="${1:-}"
FULL_CHECK=false
OUTPUT_DIR=""
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Results
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNINGS=0

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED_CHECKS++))
    ((TOTAL_CHECKS++))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAILED_CHECKS++))
    ((TOTAL_CHECKS++))
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    ((WARNINGS++))
}

log_header() {
    echo ""
    echo -e "${CYAN}========================================"
    echo "  $1"
    echo -e "========================================${NC}"
    echo ""
}

show_help() {
    cat << EOF
Complete Production Verification Suite

Usage: $0 <target-url> [OPTIONS]

Arguments:
    target-url          Target URL (e.g., https://yourdomain.com)

Options:
    --full              Run full verification including load tests
    --output DIR        Save reports to directory
    --skip-security     Skip security checks
    --skip-performance  Skip performance checks
    -h, --help          Show this help message

Examples:
    $0 https://myapp.com
    $0 https://myapp.com --full
    $0 https://myapp.com --output /tmp/reports

Verification Categories:
    1. Functional Testing   - Health, Auth, API endpoints
    2. Security Validation  - Headers, CORS, Rate limiting, SSL
    3. Performance Check    - Response times, throughput
    4. Configuration        - Environment, dependencies

EOF
}

# =============================================================================
# Parse Arguments
# =============================================================================

SKIP_SECURITY=false
SKIP_PERFORMANCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --full)
            FULL_CHECK=true
            shift
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-security)
            SKIP_SECURITY=true
            shift
            ;;
        --skip-performance)
            SKIP_PERFORMANCE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            if [[ -z "$TARGET_URL" && "$1" == http* ]]; then
                TARGET_URL="$1"
            fi
            shift
            ;;
    esac
done

if [[ -z "$TARGET_URL" ]]; then
    echo "Error: Target URL required"
    show_help
    exit 1
fi

# Setup output directory
if [[ -n "$OUTPUT_DIR" ]]; then
    mkdir -p "$OUTPUT_DIR"
fi

# =============================================================================
# Verification Functions
# =============================================================================

run_functional_tests() {
    log_header "Functional Testing"

    # Health check
    log_info "Testing health endpoint..."
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$TARGET_URL/health" 2>/dev/null || echo "000")
    if [[ "$STATUS" == "200" ]]; then
        log_success "Health endpoint: OK"
    else
        log_fail "Health endpoint: returned $STATUS"
    fi

    # Root endpoint
    log_info "Testing root endpoint..."
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$TARGET_URL/" 2>/dev/null || echo "000")
    if [[ "$STATUS" == "200" ]]; then
        log_success "Root endpoint: OK"
    else
        log_fail "Root endpoint: returned $STATUS"
    fi

    # API docs
    log_info "Testing API documentation..."
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$TARGET_URL/docs" 2>/dev/null || echo "000")
    if [[ "$STATUS" == "200" ]]; then
        log_success "API docs: OK"
    else
        log_warn "API docs: returned $STATUS (may be disabled)"
    fi

    # OpenAPI schema
    log_info "Testing OpenAPI schema..."
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$TARGET_URL/openapi.json" 2>/dev/null || echo "000")
    if [[ "$STATUS" == "200" ]]; then
        log_success "OpenAPI schema: OK"
    else
        log_warn "OpenAPI schema: returned $STATUS"
    fi

    # Auth endpoint (should return 422 without body)
    log_info "Testing auth endpoint..."
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$TARGET_URL/api/auth/login" 2>/dev/null || echo "000")
    if [[ "$STATUS" =~ ^(200|401|422)$ ]]; then
        log_success "Auth endpoint: responding"
    else
        log_fail "Auth endpoint: unexpected status $STATUS"
    fi
}

run_security_validation() {
    if $SKIP_SECURITY; then
        log_info "Skipping security validation (--skip-security)"
        return
    fi

    log_header "Security Validation"

    # Run the security script
    if [[ -f "$SCRIPT_DIR/verify_security.sh" ]]; then
        log_info "Running security validation script..."
        if "$SCRIPT_DIR/verify_security.sh" "$TARGET_URL" 2>&1 | while read line; do
            echo "  $line"
        done; then
            log_success "Security validation: passed"
        else
            log_fail "Security validation: issues found"
        fi
    else
        # Basic security checks inline
        HEADERS=$(curl -sI "$TARGET_URL" 2>/dev/null)

        # X-Frame-Options
        if echo "$HEADERS" | grep -qi "X-Frame-Options"; then
            log_success "X-Frame-Options: present"
        else
            log_fail "X-Frame-Options: missing"
        fi

        # HSTS
        if echo "$HEADERS" | grep -qi "Strict-Transport-Security"; then
            log_success "HSTS: present"
        else
            if [[ "$TARGET_URL" == https://* ]]; then
                log_fail "HSTS: missing on HTTPS site"
            else
                log_warn "HSTS: not applicable (HTTP)"
            fi
        fi

        # X-Content-Type-Options
        if echo "$HEADERS" | grep -qi "X-Content-Type-Options"; then
            log_success "X-Content-Type-Options: present"
        else
            log_fail "X-Content-Type-Options: missing"
        fi
    fi
}

run_performance_check() {
    if $SKIP_PERFORMANCE; then
        log_info "Skipping performance check (--skip-performance)"
        return
    fi

    log_header "Performance Verification"

    # Response time check
    log_info "Measuring response times..."

    local times=()
    local success=0
    local failed=0

    for i in {1..10}; do
        RESPONSE=$(curl -s -o /dev/null -w "%{time_total}" "$TARGET_URL/health" 2>/dev/null)
        if [[ $? -eq 0 && -n "$RESPONSE" ]]; then
            TIME_MS=$(echo "$RESPONSE * 1000" | bc 2>/dev/null || echo "0")
            times+=("$TIME_MS")
            ((success++))
        else
            ((failed++))
        fi
    done

    if [[ ${#times[@]} -gt 0 ]]; then
        # Calculate average
        local sum=0
        for t in "${times[@]}"; do
            sum=$(echo "$sum + $t" | bc)
        done
        local avg=$(echo "$sum / ${#times[@]}" | bc)

        # Sort for percentiles
        IFS=$'\n' sorted=($(sort -n <<<"${times[*]}")); unset IFS
        local p95_idx=$(( ${#sorted[@]} * 95 / 100 ))
        local p95=${sorted[$p95_idx]:-${sorted[-1]}}

        log_info "Results: $success/10 successful requests"
        log_info "Average response time: ${avg}ms"
        log_info "P95 response time: ${p95}ms"

        if (( $(echo "$avg < 300" | bc -l) )); then
            log_success "Average response time: ${avg}ms (target: <300ms)"
        else
            log_fail "Average response time: ${avg}ms (target: <300ms)"
        fi

        if (( $(echo "$p95 < 500" | bc -l) )); then
            log_success "P95 response time: ${p95}ms (target: <500ms)"
        else
            log_fail "P95 response time: ${p95}ms (target: <500ms)"
        fi
    else
        log_fail "Could not measure response times"
    fi

    # Full performance test (optional)
    if $FULL_CHECK && [[ -f "$SCRIPT_DIR/verify_performance.py" ]]; then
        log_info "Running full performance verification..."
        local output_arg=""
        if [[ -n "$OUTPUT_DIR" ]]; then
            output_arg="--output $OUTPUT_DIR/performance_report_$TIMESTAMP.json"
        fi

        python3 "$SCRIPT_DIR/verify_performance.py" \
            --target "$TARGET_URL" \
            --samples 20 \
            --concurrent 5 \
            $output_arg 2>&1 | while read line; do
            echo "  $line"
        done || log_warn "Performance script returned non-zero"
    fi
}

run_configuration_check() {
    log_header "Configuration Check"

    # SSL check
    if [[ "$TARGET_URL" == https://* ]]; then
        log_info "Checking SSL certificate..."
        HOSTNAME=$(echo "$TARGET_URL" | sed 's|https://||' | cut -d/ -f1)

        if command -v openssl &> /dev/null; then
            EXPIRY=$(echo | openssl s_client -servername "$HOSTNAME" -connect "$HOSTNAME:443" 2>/dev/null | \
                     openssl x509 -noout -enddate 2>/dev/null | cut -d= -f2)

            if [[ -n "$EXPIRY" ]]; then
                log_success "SSL certificate valid (expires: $EXPIRY)"
            else
                log_fail "Could not verify SSL certificate"
            fi
        else
            log_warn "OpenSSL not available for SSL check"
        fi
    else
        log_warn "Not using HTTPS - SSL check skipped"
    fi

    # Check for exposed sensitive endpoints
    log_info "Checking for exposed sensitive endpoints..."
    SENSITIVE_ENDPOINTS=("/debug" "/.env" "/config" "/__debug__")
    local exposed=false

    for endpoint in "${SENSITIVE_ENDPOINTS[@]}"; do
        STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$TARGET_URL$endpoint" 2>/dev/null || echo "000")
        if [[ "$STATUS" == "200" ]]; then
            log_fail "Sensitive endpoint exposed: $endpoint"
            exposed=true
        fi
    done

    if ! $exposed; then
        log_success "No sensitive endpoints exposed"
    fi
}

# =============================================================================
# Generate Report
# =============================================================================

generate_summary() {
    log_header "Verification Summary"

    echo "Target:     $TARGET_URL"
    echo "Timestamp:  $(date -Iseconds)"
    echo ""
    echo "Results:"
    echo -e "  ${GREEN}Passed:${NC}   $PASSED_CHECKS"
    echo -e "  ${RED}Failed:${NC}   $FAILED_CHECKS"
    echo -e "  ${YELLOW}Warnings:${NC} $WARNINGS"
    echo "  Total:    $TOTAL_CHECKS"
    echo ""

    # Calculate pass rate
    if [[ $TOTAL_CHECKS -gt 0 ]]; then
        PASS_RATE=$(( PASSED_CHECKS * 100 / TOTAL_CHECKS ))
        echo "Pass Rate:  ${PASS_RATE}%"
    fi

    echo ""
    echo "========================================"

    if [[ $FAILED_CHECKS -eq 0 ]]; then
        echo -e "${GREEN}VERIFICATION PASSED${NC}"
        echo "System is ready for production"
        EXIT_CODE=0
    elif [[ $FAILED_CHECKS -le 2 ]]; then
        echo -e "${YELLOW}VERIFICATION PASSED WITH WARNINGS${NC}"
        echo "Review failed checks before proceeding"
        EXIT_CODE=0
    else
        echo -e "${RED}VERIFICATION FAILED${NC}"
        echo "Address issues before production deployment"
        EXIT_CODE=1
    fi

    echo "========================================"

    # Save report if output directory specified
    if [[ -n "$OUTPUT_DIR" ]]; then
        REPORT_FILE="$OUTPUT_DIR/verification_report_$TIMESTAMP.txt"
        {
            echo "Production Verification Report"
            echo "=============================="
            echo ""
            echo "Target:    $TARGET_URL"
            echo "Timestamp: $(date -Iseconds)"
            echo "Mode:      $(if $FULL_CHECK; then echo "Full"; else echo "Standard"; fi)"
            echo ""
            echo "Results:"
            echo "  Passed:   $PASSED_CHECKS"
            echo "  Failed:   $FAILED_CHECKS"
            echo "  Warnings: $WARNINGS"
            echo "  Total:    $TOTAL_CHECKS"
            echo ""
            echo "Status: $(if [[ $FAILED_CHECKS -eq 0 ]]; then echo "PASSED"; else echo "FAILED"; fi)"
        } > "$REPORT_FILE"

        echo ""
        log_info "Report saved to: $REPORT_FILE"
    fi
}

# =============================================================================
# Main
# =============================================================================

echo ""
echo "========================================"
echo "  Production Verification Suite"
echo "========================================"
echo "Target: $TARGET_URL"
echo "Mode:   $(if $FULL_CHECK; then echo "Full"; else echo "Standard"; fi)"
echo "========================================"

# Check if target is reachable
log_info "Checking target reachability..."
if ! curl -s -o /dev/null "$TARGET_URL" 2>/dev/null; then
    echo -e "${RED}ERROR: Target not reachable${NC}"
    exit 1
fi
log_info "Target is reachable"

# Run all verification checks
run_functional_tests
run_security_validation
run_performance_check
run_configuration_check

# Generate summary
generate_summary

exit ${EXIT_CODE:-1}
