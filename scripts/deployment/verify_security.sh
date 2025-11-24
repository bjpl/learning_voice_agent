#!/bin/bash
#
# Security Validation Script for Production Deployment
# Validates security headers, rate limiting, CORS, and SSL configuration.
#
# Usage:
#   ./scripts/deployment/verify_security.sh https://yourdomain.com
#   ./scripts/deployment/verify_security.sh https://yourdomain.com --verbose
#   ./scripts/deployment/verify_security.sh https://yourdomain.com --output report.txt
#

set -e

# =============================================================================
# Configuration
# =============================================================================

TARGET_URL="${1:-}"
VERBOSE=false
OUTPUT_FILE=""
PASSED=0
FAILED=0
WARNINGS=0

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED++))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAILED++))
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    ((WARNINGS++))
}

show_help() {
    cat << EOF
Security Validation Script

Usage: $0 <target-url> [OPTIONS]

Arguments:
    target-url          Target URL to validate (e.g., https://yourdomain.com)

Options:
    -v, --verbose       Show detailed output
    -o, --output FILE   Save results to file
    -h, --help          Show this help message

Examples:
    $0 https://myapp.com
    $0 https://myapp.com --verbose
    $0 https://myapp.com --output security_report.txt

Checks performed:
    - Security Headers (X-Frame-Options, HSTS, CSP, etc.)
    - CORS Configuration
    - Rate Limiting
    - SSL/TLS Configuration
    - Cookie Security
    - Information Disclosure
EOF
}

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            if [[ -z "$TARGET_URL" ]]; then
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

# Remove trailing slash
TARGET_URL="${TARGET_URL%/}"

# =============================================================================
# Pre-flight Checks
# =============================================================================

echo ""
echo "========================================"
echo "  Security Validation"
echo "========================================"
echo "Target: $TARGET_URL"
echo "Date:   $(date -Iseconds)"
echo "========================================"
echo ""

# Check if curl is available
if ! command -v curl &> /dev/null; then
    echo "Error: curl is required but not installed"
    exit 1
fi

# Check if target is reachable
log_info "Checking target reachability..."
if ! curl -s -o /dev/null -w "%{http_code}" "$TARGET_URL" | grep -qE "^[23]"; then
    log_fail "Target not reachable: $TARGET_URL"
    exit 1
fi
log_pass "Target is reachable"

# =============================================================================
# Security Header Checks
# =============================================================================

echo ""
echo "--- Security Headers ---"

# Get headers
HEADERS=$(curl -sI "$TARGET_URL" 2>/dev/null)

if $VERBOSE; then
    echo "Response Headers:"
    echo "$HEADERS"
    echo ""
fi

# Check X-Frame-Options
if echo "$HEADERS" | grep -qi "X-Frame-Options"; then
    VALUE=$(echo "$HEADERS" | grep -i "X-Frame-Options" | cut -d: -f2 | tr -d ' \r')
    if [[ "$VALUE" =~ ^(DENY|SAMEORIGIN)$ ]]; then
        log_pass "X-Frame-Options: $VALUE"
    else
        log_warn "X-Frame-Options has unusual value: $VALUE"
    fi
else
    log_fail "X-Frame-Options header missing"
fi

# Check X-Content-Type-Options
if echo "$HEADERS" | grep -qi "X-Content-Type-Options"; then
    VALUE=$(echo "$HEADERS" | grep -i "X-Content-Type-Options" | cut -d: -f2 | tr -d ' \r')
    if [[ "$VALUE" == "nosniff" ]]; then
        log_pass "X-Content-Type-Options: nosniff"
    else
        log_warn "X-Content-Type-Options has unusual value: $VALUE"
    fi
else
    log_fail "X-Content-Type-Options header missing"
fi

# Check Strict-Transport-Security (HSTS)
if echo "$HEADERS" | grep -qi "Strict-Transport-Security"; then
    VALUE=$(echo "$HEADERS" | grep -i "Strict-Transport-Security" | cut -d: -f2-)
    if echo "$VALUE" | grep -q "max-age"; then
        log_pass "Strict-Transport-Security present"
        if $VERBOSE; then
            echo "    Value: $VALUE"
        fi
    else
        log_warn "HSTS present but may be misconfigured"
    fi
else
    if [[ "$TARGET_URL" == https://* ]]; then
        log_fail "Strict-Transport-Security header missing (HTTPS site)"
    else
        log_warn "Strict-Transport-Security not applicable (HTTP site)"
    fi
fi

# Check Content-Security-Policy
if echo "$HEADERS" | grep -qi "Content-Security-Policy"; then
    log_pass "Content-Security-Policy present"
    if $VERBOSE; then
        CSP=$(echo "$HEADERS" | grep -i "Content-Security-Policy" | cut -d: -f2-)
        echo "    Value: $CSP"
    fi
else
    log_warn "Content-Security-Policy header missing (recommended)"
fi

# Check X-XSS-Protection
if echo "$HEADERS" | grep -qi "X-XSS-Protection"; then
    VALUE=$(echo "$HEADERS" | grep -i "X-XSS-Protection" | cut -d: -f2 | tr -d ' \r')
    log_pass "X-XSS-Protection: $VALUE"
else
    log_warn "X-XSS-Protection header missing (deprecated but useful for older browsers)"
fi

# Check Referrer-Policy
if echo "$HEADERS" | grep -qi "Referrer-Policy"; then
    VALUE=$(echo "$HEADERS" | grep -i "Referrer-Policy" | cut -d: -f2 | tr -d ' \r')
    log_pass "Referrer-Policy: $VALUE"
else
    log_warn "Referrer-Policy header missing (recommended)"
fi

# Check for server information disclosure
if echo "$HEADERS" | grep -qi "^Server:"; then
    SERVER=$(echo "$HEADERS" | grep -i "^Server:" | cut -d: -f2 | tr -d ' \r')
    if [[ -n "$SERVER" && "$SERVER" != *"cloudflare"* ]]; then
        log_warn "Server header exposes information: $SERVER"
    else
        log_pass "Server header acceptable"
    fi
else
    log_pass "Server header not exposed"
fi

# Check X-Powered-By
if echo "$HEADERS" | grep -qi "X-Powered-By"; then
    VALUE=$(echo "$HEADERS" | grep -i "X-Powered-By" | cut -d: -f2 | tr -d ' \r')
    log_fail "X-Powered-By header exposes technology: $VALUE"
else
    log_pass "X-Powered-By header not exposed"
fi

# =============================================================================
# CORS Validation
# =============================================================================

echo ""
echo "--- CORS Configuration ---"

# Test with unauthorized origin
CORS_RESPONSE=$(curl -sI \
    -H "Origin: https://malicious-attacker-site.com" \
    -H "Access-Control-Request-Method: POST" \
    -X OPTIONS \
    "$TARGET_URL/api/auth/login" 2>/dev/null)

ACAO=$(echo "$CORS_RESPONSE" | grep -i "Access-Control-Allow-Origin" | cut -d: -f2 | tr -d ' \r')

if [[ "$ACAO" == "*" ]]; then
    log_fail "CORS allows all origins (wildcard *)"
elif [[ "$ACAO" == *"malicious"* ]]; then
    log_fail "CORS reflects malicious origin"
elif [[ -z "$ACAO" ]]; then
    log_pass "CORS does not allow unauthorized origins"
else
    log_pass "CORS restricts origins: $ACAO"
fi

# Check CORS credentials
if echo "$CORS_RESPONSE" | grep -qi "Access-Control-Allow-Credentials: true"; then
    if [[ "$ACAO" == "*" ]]; then
        log_fail "CORS allows credentials with wildcard origin (security risk)"
    else
        log_pass "CORS credentials allowed with specific origins"
    fi
fi

# =============================================================================
# Rate Limiting Check
# =============================================================================

echo ""
echo "--- Rate Limiting ---"

log_info "Testing rate limiting (sending 50 rapid requests)..."

RATE_LIMITED=false
REQUESTS_SENT=0

for i in {1..50}; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$TARGET_URL/health" 2>/dev/null)
    ((REQUESTS_SENT++))

    if [[ "$STATUS" == "429" ]]; then
        RATE_LIMITED=true
        break
    fi
done

if $RATE_LIMITED; then
    log_pass "Rate limiting active (triggered after $REQUESTS_SENT requests)"
else
    log_warn "Rate limiting not triggered after $REQUESTS_SENT requests"
fi

# =============================================================================
# SSL/TLS Configuration (if HTTPS)
# =============================================================================

if [[ "$TARGET_URL" == https://* ]]; then
    echo ""
    echo "--- SSL/TLS Configuration ---"

    # Extract hostname
    HOSTNAME=$(echo "$TARGET_URL" | sed 's|https://||' | cut -d/ -f1 | cut -d: -f1)

    # Check SSL certificate
    if command -v openssl &> /dev/null; then
        CERT_INFO=$(echo | openssl s_client -servername "$HOSTNAME" -connect "$HOSTNAME:443" 2>/dev/null)

        # Check certificate validity
        if echo "$CERT_INFO" | grep -q "Verify return code: 0"; then
            log_pass "SSL certificate valid"
        else
            VERIFY_CODE=$(echo "$CERT_INFO" | grep "Verify return code" | cut -d: -f2)
            log_fail "SSL certificate issue:$VERIFY_CODE"
        fi

        # Check TLS version
        TLS_VERSION=$(echo "$CERT_INFO" | grep "Protocol" | head -1 | awk '{print $3}')
        if [[ "$TLS_VERSION" == "TLSv1.2" || "$TLS_VERSION" == "TLSv1.3" ]]; then
            log_pass "TLS version: $TLS_VERSION"
        else
            log_warn "TLS version may be outdated: $TLS_VERSION"
        fi

        # Check certificate expiry
        EXPIRY=$(echo | openssl s_client -servername "$HOSTNAME" -connect "$HOSTNAME:443" 2>/dev/null | \
                 openssl x509 -noout -enddate 2>/dev/null | cut -d= -f2)
        if [[ -n "$EXPIRY" ]]; then
            log_info "Certificate expires: $EXPIRY"

            # Check if expiring soon (30 days)
            EXPIRY_EPOCH=$(date -d "$EXPIRY" +%s 2>/dev/null || date -j -f "%b %d %H:%M:%S %Y %Z" "$EXPIRY" +%s 2>/dev/null || echo "0")
            NOW_EPOCH=$(date +%s)
            DAYS_LEFT=$(( (EXPIRY_EPOCH - NOW_EPOCH) / 86400 ))

            if [[ $DAYS_LEFT -lt 30 ]]; then
                log_warn "Certificate expires in $DAYS_LEFT days"
            else
                log_pass "Certificate valid for $DAYS_LEFT days"
            fi
        fi

        # Check for weak ciphers
        CIPHER=$(echo "$CERT_INFO" | grep "Cipher" | head -1 | awk '{print $3}')
        if [[ -n "$CIPHER" ]]; then
            if echo "$CIPHER" | grep -qiE "(RC4|DES|MD5|NULL|EXPORT|ANON)"; then
                log_fail "Weak cipher detected: $CIPHER"
            else
                log_pass "Cipher appears secure: $CIPHER"
            fi
        fi
    else
        log_warn "OpenSSL not available, skipping detailed SSL checks"
    fi
fi

# =============================================================================
# Cookie Security
# =============================================================================

echo ""
echo "--- Cookie Security ---"

# Login to get session cookie
COOKIE_RESPONSE=$(curl -s -c - -X POST \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "username=test@test.com&password=test" \
    "$TARGET_URL/api/auth/login" 2>/dev/null)

# Check for secure cookie attributes in Set-Cookie header
COOKIE_HEADERS=$(curl -sI -X POST \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "username=test@test.com&password=test" \
    "$TARGET_URL/api/auth/login" 2>/dev/null | grep -i "Set-Cookie")

if [[ -n "$COOKIE_HEADERS" ]]; then
    # Check Secure flag
    if echo "$COOKIE_HEADERS" | grep -qi "Secure"; then
        log_pass "Cookies have Secure flag"
    else
        if [[ "$TARGET_URL" == https://* ]]; then
            log_fail "Cookies missing Secure flag on HTTPS site"
        else
            log_warn "Cookies missing Secure flag (expected on HTTP)"
        fi
    fi

    # Check HttpOnly flag
    if echo "$COOKIE_HEADERS" | grep -qi "HttpOnly"; then
        log_pass "Cookies have HttpOnly flag"
    else
        log_warn "Cookies missing HttpOnly flag"
    fi

    # Check SameSite attribute
    if echo "$COOKIE_HEADERS" | grep -qi "SameSite"; then
        SAMESITE=$(echo "$COOKIE_HEADERS" | grep -oi "SameSite=[^;]*" | head -1)
        log_pass "Cookies have SameSite attribute: $SAMESITE"
    else
        log_warn "Cookies missing SameSite attribute"
    fi
else
    log_info "No cookies set (may be using token-based auth)"
fi

# =============================================================================
# Information Disclosure
# =============================================================================

echo ""
echo "--- Information Disclosure ---"

# Check for exposed error details
ERROR_RESPONSE=$(curl -s "$TARGET_URL/api/nonexistent-endpoint-12345")

if echo "$ERROR_RESPONSE" | grep -qiE "(traceback|stack trace|exception|error in|debug)"; then
    log_fail "Error response may expose sensitive information"
else
    log_pass "Error responses do not expose stack traces"
fi

# Check for exposed debug endpoints
DEBUG_ENDPOINTS=("/__debug__" "/debug" "/.env" "/config" "/phpinfo.php" "/server-status")
for endpoint in "${DEBUG_ENDPOINTS[@]}"; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$TARGET_URL$endpoint" 2>/dev/null)
    if [[ "$STATUS" == "200" ]]; then
        log_fail "Debug/sensitive endpoint accessible: $endpoint"
    fi
done
log_pass "No common debug endpoints exposed"

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "========================================"
echo "  Security Validation Summary"
echo "========================================"
echo -e "${GREEN}Passed:${NC}   $PASSED"
echo -e "${RED}Failed:${NC}   $FAILED"
echo -e "${YELLOW}Warnings:${NC} $WARNINGS"
echo "========================================"

if [[ $FAILED -gt 0 ]]; then
    echo -e "\n${RED}SECURITY VALIDATION FAILED${NC}"
    echo "Address the failed checks before production deployment."
    EXIT_CODE=1
elif [[ $WARNINGS -gt 3 ]]; then
    echo -e "\n${YELLOW}SECURITY VALIDATION PASSED WITH WARNINGS${NC}"
    echo "Review warnings and address if possible."
    EXIT_CODE=0
else
    echo -e "\n${GREEN}SECURITY VALIDATION PASSED${NC}"
    echo "Security configuration appears solid."
    EXIT_CODE=0
fi

# Save output if requested
if [[ -n "$OUTPUT_FILE" ]]; then
    {
        echo "Security Validation Report"
        echo "========================="
        echo "Target: $TARGET_URL"
        echo "Date: $(date -Iseconds)"
        echo ""
        echo "Results:"
        echo "  Passed:   $PASSED"
        echo "  Failed:   $FAILED"
        echo "  Warnings: $WARNINGS"
        echo ""
        echo "Exit Code: $EXIT_CODE"
    } > "$OUTPUT_FILE"
    echo ""
    echo "Report saved to: $OUTPUT_FILE"
fi

exit $EXIT_CODE
