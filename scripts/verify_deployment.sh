#!/bin/bash
# Deployment Verification Script for Learning Voice Agent
# Usage: ./scripts/verify_deployment.sh [URL]
# Example: ./scripts/verify_deployment.sh https://yourdomain.com

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
BASE_URL="${1:-http://localhost:8000}"
TIMEOUT=10
PASSED=0
FAILED=0
WARNINGS=0

echo "=============================================="
echo "Learning Voice Agent - Deployment Verification"
echo "=============================================="
echo "Target: $BASE_URL"
echo "Time: $(date)"
echo ""

# Helper functions
check_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED++))
}

check_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAILED++))
}

check_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    ((WARNINGS++))
}

# 1. Health Check
echo "--- Health Checks ---"

response=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout $TIMEOUT "$BASE_URL/health" 2>/dev/null || echo "000")
if [ "$response" = "200" ]; then
    check_pass "Health endpoint (/health) - HTTP $response"
else
    check_fail "Health endpoint (/health) - HTTP $response (expected 200)"
fi

response=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout $TIMEOUT "$BASE_URL/" 2>/dev/null || echo "000")
if [ "$response" = "200" ]; then
    check_pass "Root endpoint (/) - HTTP $response"
else
    check_fail "Root endpoint (/) - HTTP $response (expected 200)"
fi

response=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout $TIMEOUT "$BASE_URL/health/detailed" 2>/dev/null || echo "000")
if [ "$response" = "200" ]; then
    check_pass "Detailed health (/health/detailed) - HTTP $response"
else
    check_warn "Detailed health (/health/detailed) - HTTP $response"
fi

echo ""

# 2. Security Headers
echo "--- Security Headers ---"

headers=$(curl -sI --connect-timeout $TIMEOUT "$BASE_URL/" 2>/dev/null || echo "")

if echo "$headers" | grep -qi "content-security-policy"; then
    check_pass "Content-Security-Policy header present"
else
    check_fail "Content-Security-Policy header missing"
fi

if echo "$headers" | grep -qi "strict-transport-security"; then
    check_pass "Strict-Transport-Security (HSTS) header present"
else
    if [[ "$BASE_URL" == https://* ]]; then
        check_fail "Strict-Transport-Security header missing"
    else
        check_warn "Strict-Transport-Security header missing (expected for HTTPS)"
    fi
fi

if echo "$headers" | grep -qi "x-frame-options"; then
    check_pass "X-Frame-Options header present"
else
    check_fail "X-Frame-Options header missing"
fi

if echo "$headers" | grep -qi "x-content-type-options"; then
    check_pass "X-Content-Type-Options header present"
else
    check_fail "X-Content-Type-Options header missing"
fi

echo ""

# 3. API Endpoints
echo "--- API Endpoints ---"

response=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout $TIMEOUT "$BASE_URL/api/stats" 2>/dev/null || echo "000")
if [ "$response" = "200" ]; then
    check_pass "Stats endpoint (/api/stats) - HTTP $response"
else
    check_fail "Stats endpoint (/api/stats) - HTTP $response (expected 200)"
fi

response=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout $TIMEOUT "$BASE_URL/api/offline-manifest" 2>/dev/null || echo "000")
if [ "$response" = "200" ]; then
    check_pass "Offline manifest (/api/offline-manifest) - HTTP $response"
else
    check_warn "Offline manifest (/api/offline-manifest) - HTTP $response"
fi

echo ""

# 4. Authentication Endpoints
echo "--- Authentication Endpoints ---"

response=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout $TIMEOUT \
    -X POST "$BASE_URL/api/auth/register" \
    -H "Content-Type: application/json" \
    -d '{}' 2>/dev/null || echo "000")
if [ "$response" = "422" ] || [ "$response" = "400" ]; then
    check_pass "Auth register endpoint exists (returns validation error for empty body)"
else
    check_warn "Auth register endpoint - HTTP $response"
fi

response=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout $TIMEOUT \
    -X POST "$BASE_URL/api/auth/login" \
    -H "Content-Type: application/json" \
    -d '{}' 2>/dev/null || echo "000")
if [ "$response" = "422" ] || [ "$response" = "400" ] || [ "$response" = "401" ]; then
    check_pass "Auth login endpoint exists (returns validation/auth error for empty body)"
else
    check_warn "Auth login endpoint - HTTP $response"
fi

echo ""

# 5. CORS Verification
echo "--- CORS Verification ---"

cors_headers=$(curl -sI --connect-timeout $TIMEOUT \
    -H "Origin: https://unauthorized-origin.com" \
    -X OPTIONS "$BASE_URL/api/auth/login" 2>/dev/null || echo "")

if echo "$cors_headers" | grep -qi "access-control-allow-origin: https://unauthorized-origin.com"; then
    check_fail "CORS allows unauthorized origin (security issue)"
else
    check_pass "CORS blocks unauthorized origins"
fi

echo ""

# 6. Rate Limiting
echo "--- Rate Limiting ---"

echo "Sending 15 rapid requests to test rate limiting..."
rate_limit_triggered=false
for i in {1..15}; do
    response=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 "$BASE_URL/" 2>/dev/null || echo "000")
    if [ "$response" = "429" ]; then
        rate_limit_triggered=true
        break
    fi
done

if [ "$rate_limit_triggered" = true ]; then
    check_pass "Rate limiting is active (429 returned)"
else
    check_warn "Rate limiting may not be active (no 429 received in 15 requests)"
fi

echo ""

# 7. Response Times
echo "--- Performance ---"

total_time=$(curl -s -o /dev/null -w "%{time_total}" --connect-timeout $TIMEOUT "$BASE_URL/health" 2>/dev/null || echo "999")
time_ms=$(echo "$total_time * 1000" | bc 2>/dev/null || echo "N/A")

if [ "$time_ms" != "N/A" ]; then
    time_int=${time_ms%.*}
    if [ "$time_int" -lt 500 ]; then
        check_pass "Response time: ${time_ms}ms (< 500ms)"
    elif [ "$time_int" -lt 1000 ]; then
        check_warn "Response time: ${time_ms}ms (< 1000ms but > 500ms)"
    else
        check_fail "Response time: ${time_ms}ms (> 1000ms)"
    fi
else
    check_warn "Could not measure response time"
fi

echo ""

# 8. SSL/TLS Check (if HTTPS)
if [[ "$BASE_URL" == https://* ]]; then
    echo "--- SSL/TLS ---"

    ssl_info=$(curl -sI --connect-timeout $TIMEOUT "$BASE_URL/" 2>&1 | head -1 || echo "")
    if echo "$ssl_info" | grep -q "200\|301\|302"; then
        check_pass "SSL/TLS connection successful"
    else
        check_fail "SSL/TLS connection failed"
    fi
    echo ""
fi

# Summary
echo "=============================================="
echo "VERIFICATION SUMMARY"
echo "=============================================="
echo -e "${GREEN}Passed:${NC}   $PASSED"
echo -e "${RED}Failed:${NC}   $FAILED"
echo -e "${YELLOW}Warnings:${NC} $WARNINGS"
echo ""

if [ "$FAILED" -eq 0 ]; then
    echo -e "${GREEN}DEPLOYMENT VERIFICATION SUCCESSFUL${NC}"
    exit 0
else
    echo -e "${RED}DEPLOYMENT VERIFICATION FAILED${NC}"
    echo "Please review the failed checks above."
    exit 1
fi
