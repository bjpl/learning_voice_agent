# Plan A: Security-First Architecture Design

## SPARC Methodology - Security Implementation

**Document Version:** 1.0.0
**Created:** 2024-11-23
**Status:** Implementation Ready

---

## Executive Summary

This document defines the security architecture for the Learning Voice Agent application following SPARC methodology. It addresses 8 critical security features identified in the GMS audit:

1. JWT Authentication System
2. Rate Limiting with slowapi
3. CORS Configuration Fix
4. WebSocket Authentication
5. Twilio Validation Fix
6. Dependency Updates
7. Privacy Policy & Terms
8. GDPR Data Export/Deletion

---

## 1. JWT Authentication System

### Specification
- Token-based authentication for all API endpoints
- User registration and login system
- Access and refresh token mechanism
- Role-based access control (RBAC)
- Token blacklisting for logout

### Pseudocode
```
FUNCTION authenticate_user(credentials):
    user = find_user_by_email(credentials.email)
    IF user AND verify_password(credentials.password, user.hashed_password):
        access_token = create_access_token(user.id, expiry=15min)
        refresh_token = create_refresh_token(user.id, expiry=7days)
        RETURN {access_token, refresh_token}
    ELSE:
        RAISE AuthenticationError

FUNCTION verify_token(token):
    TRY:
        payload = decode_jwt(token, SECRET_KEY)
        IF token_is_blacklisted(payload.jti):
            RAISE InvalidTokenError
        RETURN payload
    CATCH ExpiredTokenError:
        RAISE TokenExpiredError
    CATCH InvalidTokenError:
        RAISE UnauthorizedError
```

### Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    Client Application                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Security Layer                     │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│  │ JWT Verify  │  │ Rate Limit   │  │ CORS Check      │    │
│  └─────────────┘  └──────────────┘  └─────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Authentication Service                    │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│  │ User Mgmt   │  │ Token Mgmt   │  │ Session Mgmt    │    │
│  └─────────────┘  └──────────────┘  └─────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       Data Layer                             │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│  │ SQLite      │  │ Redis        │  │ Token Blacklist │    │
│  │ (Users)     │  │ (Sessions)   │  │ (Redis)         │    │
│  └─────────────┘  └──────────────┘  └─────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Security Decisions
- **Algorithm:** RS256 for production, HS256 for development
- **Access Token TTL:** 15 minutes
- **Refresh Token TTL:** 7 days
- **Password Hashing:** bcrypt with work factor 12
- **Token Storage:** Redis with TTL-based expiration

---

## 2. Rate Limiting with slowapi

### Specification
- 100 requests/minute per IP for general endpoints
- 10 requests/minute for auth endpoints (brute-force protection)
- 1000 requests/minute for health checks
- Redis-backed for distributed deployments
- Configurable limits via environment variables

### Pseudocode
```
FUNCTION rate_limit_check(request, limit, window):
    key = f"rate_limit:{request.client.host}:{request.path}"
    current_count = redis.get(key) OR 0

    IF current_count >= limit:
        remaining_time = redis.ttl(key)
        RAISE RateLimitExceeded(retry_after=remaining_time)

    redis.incr(key)
    IF current_count == 0:
        redis.expire(key, window)

    RETURN {remaining: limit - current_count - 1, reset: redis.ttl(key)}
```

### Rate Limit Tiers
| Endpoint Category | Limit | Window | Key |
|------------------|-------|--------|-----|
| General API | 100 | 60s | IP |
| Auth endpoints | 10 | 60s | IP |
| Health checks | 1000 | 60s | IP |
| WebSocket | 30 | 60s | IP |
| Admin | 50 | 60s | User |

---

## 3. CORS Configuration Fix

### Specification
- Replace wildcard ["*"] with explicit origins
- Environment-based configuration
- Proper credentials handling
- Preflight caching for performance

### Configuration
```python
# Development
CORS_ORIGINS = ["http://localhost:3000", "http://localhost:8000"]

# Staging
CORS_ORIGINS = ["https://staging.app.example.com"]

# Production
CORS_ORIGINS = ["https://app.example.com", "https://www.example.com"]
```

### Security Headers
- Access-Control-Allow-Credentials: true (only for allowed origins)
- Access-Control-Max-Age: 600 (cache preflight for 10 minutes)
- Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
- Access-Control-Allow-Headers: Authorization, Content-Type, X-Request-ID

---

## 4. WebSocket Authentication

### Specification
- Token validation before handshake acceptance
- Session ownership verification
- Automatic disconnection on token expiry
- Heartbeat mechanism for session validation

### Pseudocode
```
FUNCTION websocket_connect(websocket, session_id, token):
    TRY:
        # Validate token first
        payload = verify_token(token)
        user_id = payload.sub

        # Verify session ownership
        session = get_session(session_id)
        IF session.user_id != user_id:
            RAISE UnauthorizedError("Session does not belong to user")

        # Accept connection
        await websocket.accept()

        # Start heartbeat
        start_heartbeat_task(websocket, payload.exp)

    CATCH AuthenticationError:
        await websocket.close(code=4001, reason="Authentication failed")
```

---

## 5. Twilio Validation Fix

### Specification
- FAIL CLOSED when auth token not configured
- Proper X-Twilio-Signature validation
- Request replay protection
- Audit logging for all Twilio requests

### Current Issue
```python
# VULNERABLE: Returns True when validator not configured
if not self.validator:
    return True  # Skip validation in dev
```

### Fixed Implementation
```python
# SECURE: Fails closed
if not self.validator:
    logger.warning("Twilio validator not configured - rejecting request")
    return False  # Fail closed
```

---

## 6. Dependency Updates

### Critical Updates Required
| Package | Current | Target | Risk Level |
|---------|---------|--------|------------|
| cryptography | 41.0.7 | 46.0.3 | HIGH |
| anthropic | 0.18.1 | 0.74.1 | MEDIUM |

### Breaking Changes - Anthropic
```python
# Old (0.18.x)
response = client.completion(prompt=prompt, max_tokens=100)

# New (0.74.x)
response = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=100,
    messages=[{"role": "user", "content": prompt}]
)
```

---

## 7. Privacy Policy & Terms

### Required Documents
1. **Privacy Policy** - GDPR/CCPA compliant
2. **Terms of Service** - Legal protection
3. **Cookie Policy** - Consent management
4. **Data Processing Agreement** - B2B compliance

### Key Sections
- Data Collection practices
- Data Usage and sharing
- User rights (access, rectification, erasure)
- Data retention periods
- Contact information for DPO

---

## 8. GDPR Data Export/Deletion

### Specification
- User data export in machine-readable format (JSON)
- Right to be forgotten implementation
- Audit trail for compliance
- Soft delete with configurable retention

### Pseudocode
```
FUNCTION export_user_data(user_id):
    data = {
        "user": get_user_profile(user_id),
        "conversations": get_all_conversations(user_id),
        "sessions": get_all_sessions(user_id),
        "preferences": get_user_preferences(user_id),
        "export_date": datetime.utcnow(),
        "export_format": "GDPR-compliant-v1"
    }
    RETURN json.dumps(data)

FUNCTION delete_user_data(user_id, reason):
    # Audit log
    log_deletion_request(user_id, reason)

    # Soft delete user
    mark_user_deleted(user_id)

    # Anonymize conversations (keep for analytics)
    anonymize_conversations(user_id)

    # Delete PII
    delete_personal_data(user_id)

    # Schedule hard delete after retention period
    schedule_hard_delete(user_id, retention_days=30)
```

---

## Implementation Timeline

| Feature | Priority | Estimated Days | Dependencies |
|---------|----------|----------------|--------------|
| JWT Auth | P0 | 2-3 | None |
| Rate Limiting | P0 | 1 | Redis |
| CORS Fix | P0 | 0.5 | None |
| WebSocket Auth | P1 | 1 | JWT Auth |
| Twilio Fix | P0 | 0.5 | None |
| Dependency Updates | P1 | 1 | Testing |
| Privacy Policy | P2 | 1-2 | None |
| GDPR Endpoints | P1 | 1-2 | JWT Auth |

**Total Estimated: 8-12 days**

---

## Security Verification Checklist

- [ ] All endpoints require authentication (except health/docs)
- [ ] Rate limiting active on all endpoints
- [ ] CORS restricted to known origins
- [ ] WebSocket connections authenticated
- [ ] Twilio validation fails closed
- [ ] No known vulnerable dependencies
- [ ] Privacy policy accessible
- [ ] Data export/deletion functional
- [ ] All security tests passing

---

## Files Created

- `app/security/__init__.py` - Security module exports
- `app/security/auth.py` - JWT authentication
- `app/security/models.py` - User and token models
- `app/security/rate_limit.py` - Rate limiting
- `app/security/dependencies.py` - FastAPI dependencies
- `tests/security/` - Security test suite
