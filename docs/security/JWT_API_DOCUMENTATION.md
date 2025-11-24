# JWT Authentication API Documentation

**Version**: 1.0.0
**Last Updated**: 2025-11-23
**Base URL**: `/api/auth`, `/api/user`, `/api/gdpr`

## Overview

The Learning Voice Agent uses JWT (JSON Web Tokens) for stateless authentication. The system provides:
- User registration and login
- Access tokens (short-lived, 15 minutes)
- Refresh tokens (long-lived, 7 days)
- Token blacklisting for logout
- Account security features (lockout, password requirements)
- GDPR compliance (data export/deletion)

## Table of Contents
- [Authentication Flow](#authentication-flow)
- [API Endpoints](#api-endpoints)
- [Error Responses](#error-responses)
- [Security Considerations](#security-considerations)
- [Code Examples](#code-examples)

---

## Authentication Flow

```
┌─────────┐                 ┌─────────┐                 ┌─────────┐
│ Client  │                 │   API   │                 │ Database│
└────┬────┘                 └────┬────┘                 └────┬────┘
     │                           │                           │
     │ 1. POST /auth/register    │                           │
     ├───────────────────────────>│ Create user              │
     │                           ├──────────────────────────>│
     │                           │                           │
     │ 2. POST /auth/login       │                           │
     ├───────────────────────────>│ Verify credentials       │
     │                           ├──────────────────────────>│
     │ 3. Return tokens          │                           │
     │<───────────────────────────┤                           │
     │ {access_token, refresh_token}                         │
     │                           │                           │
     │ 4. API Request            │                           │
     │ Authorization: Bearer xxx │                           │
     ├───────────────────────────>│ Verify token             │
     │                           │                           │
     │ 5. Response               │                           │
     │<───────────────────────────┤                           │
     │                           │                           │
     │ 6. Token Expired          │                           │
     ├───────────────────────────>│                           │
     │ 7. 401 Unauthorized       │                           │
     │<───────────────────────────┤                           │
     │                           │                           │
     │ 8. POST /auth/refresh     │                           │
     │ {refresh_token}           │                           │
     ├───────────────────────────>│ Issue new access token   │
     │                           │                           │
     │ 9. New access token       │                           │
     │<───────────────────────────┤                           │
     │                           │                           │
```

---

## API Endpoints

### Authentication Endpoints

#### 1. Register User

Create a new user account.

**Endpoint**: `POST /api/auth/register`
**Rate Limit**: 10 requests/minute
**Authentication**: None (public endpoint)

**Request Body**:
```json
{
  "email": "user@example.com",
  "password": "SecurePass123",
  "full_name": "John Doe"
}
```

**Request Fields**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `email` | string | Yes | Valid email address (unique) |
| `password` | string | Yes | Min 8 chars, must include uppercase, lowercase, and digit |
| `full_name` | string | No | User's display name |

**Response** (201 Created):
```json
{
  "id": "usr_1234567890abcdef",
  "email": "user@example.com",
  "full_name": "John Doe",
  "role": "user",
  "status": "active",
  "created_at": "2025-11-23T12:00:00Z",
  "last_login": null
}
```

**Errors**:
- `400 Bad Request`: Invalid email or password format
- `409 Conflict`: Email already registered
- `429 Too Many Requests`: Rate limit exceeded

---

#### 2. Login (Form-based)

Authenticate user and receive tokens. Uses OAuth2 password flow for OpenAPI compatibility.

**Endpoint**: `POST /api/auth/login`
**Rate Limit**: 10 requests/minute
**Authentication**: None (public endpoint)
**Content-Type**: `application/x-www-form-urlencoded`

**Request Body** (form data):
```
username=user@example.com&password=SecurePass123
```

**Response** (200 OK):
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 900
}
```

**Response Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `access_token` | string | JWT for API access (15 min validity) |
| `refresh_token` | string | JWT for token refresh (7 days validity) |
| `token_type` | string | Always "bearer" |
| `expires_in` | integer | Access token lifetime in seconds (900 = 15 min) |

**Errors**:
- `401 Unauthorized`: Invalid credentials
- `423 Locked`: Account locked (5 failed attempts, 15 min lockout)
- `429 Too Many Requests`: Rate limit exceeded

---

#### 3. Login (JSON-based)

Alternative login endpoint accepting JSON body.

**Endpoint**: `POST /api/auth/login/json`
**Rate Limit**: 10 requests/minute
**Authentication**: None (public endpoint)
**Content-Type**: `application/json`

**Request Body**:
```json
{
  "email": "user@example.com",
  "password": "SecurePass123"
}
```

**Response**: Same as form-based login

---

#### 4. Refresh Token

Get a new access token using refresh token.

**Endpoint**: `POST /api/auth/refresh`
**Rate Limit**: 10 requests/minute
**Authentication**: None (public endpoint)

**Request Body**:
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response** (200 OK):
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 900
}
```

**Errors**:
- `401 Unauthorized`: Invalid or expired refresh token
- `429 Too Many Requests`: Rate limit exceeded

---

#### 5. Logout

Invalidate access and refresh tokens.

**Endpoint**: `POST /api/auth/logout`
**Rate Limit**: 100 requests/minute
**Authentication**: Required (Bearer token)

**Headers**:
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Request Body** (optional):
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response** (204 No Content): Empty body

**Errors**:
- `401 Unauthorized`: Invalid or missing token
- `429 Too Many Requests`: Rate limit exceeded

---

### User Profile Endpoints

#### 6. Get Current User

Get the authenticated user's profile.

**Endpoint**: `GET /api/user/me`
**Rate Limit**: 100 requests/minute
**Authentication**: Required (Bearer token)

**Response** (200 OK):
```json
{
  "id": "usr_1234567890abcdef",
  "email": "user@example.com",
  "full_name": "John Doe",
  "role": "user",
  "status": "active",
  "created_at": "2025-11-23T12:00:00Z",
  "last_login": "2025-11-23T14:30:00Z"
}
```

---

#### 7. Update Profile

Update the current user's profile.

**Endpoint**: `PATCH /api/user/me`
**Rate Limit**: 100 requests/minute
**Authentication**: Required (Bearer token)

**Request Body**:
```json
{
  "full_name": "Jane Doe",
  "email": "newemail@example.com"
}
```

**Response** (200 OK): Updated user object

**Errors**:
- `400 Bad Request`: Email already in use
- `401 Unauthorized`: Invalid token

---

#### 8. Change Password

Change the current user's password.

**Endpoint**: `POST /api/user/me/password`
**Rate Limit**: 10 requests/minute
**Authentication**: Required (Bearer token)

**Request Body**:
```json
{
  "current_password": "OldPass123",
  "new_password": "NewSecurePass456"
}
```

**Response** (204 No Content): Empty body

**Errors**:
- `400 Bad Request`: Current password incorrect or new password invalid
- `401 Unauthorized`: Invalid token

---

### GDPR Compliance Endpoints

#### 9. Request Data Export

Request export of all user data (GDPR Article 20).

**Endpoint**: `POST /api/gdpr/export`
**Rate Limit**: 10 requests/minute
**Authentication**: Required (Bearer token)

**Request Body**:
```json
{
  "format": "json",
  "include": ["profile", "conversations", "sessions", "preferences"]
}
```

**Response** (200 OK):
```json
{
  "export_id": "exp_1234567890abcdef",
  "status": "processing",
  "created_at": "2025-11-23T15:00:00Z",
  "download_url": null,
  "expires_at": "2025-11-24T15:00:00Z"
}
```

---

#### 10. Get Export Status

Check the status of a data export request.

**Endpoint**: `GET /api/gdpr/export/{export_id}`
**Rate Limit**: 100 requests/minute
**Authentication**: Required (Bearer token)

**Response** (200 OK):
```json
{
  "export_id": "exp_1234567890abcdef",
  "status": "completed",
  "created_at": "2025-11-23T15:00:00Z",
  "download_url": "/api/gdpr/export/exp_1234567890abcdef/download",
  "expires_at": "2025-11-24T15:00:00Z"
}
```

**Status Values**:
- `processing`: Export in progress
- `completed`: Export ready for download
- `failed`: Export failed (error in response)
- `expired`: Download link expired

---

#### 11. Download Export

Download completed data export.

**Endpoint**: `GET /api/gdpr/export/{export_id}/download`
**Rate Limit**: 100 requests/minute
**Authentication**: Required (Bearer token)

**Response** (200 OK):
```json
{
  "export_id": "exp_1234567890abcdef",
  "format": "json",
  "data": {
    "export_metadata": {
      "format_version": "1.0.0",
      "generated_at": "2025-11-23T15:05:00Z",
      "user_id": "usr_1234567890abcdef",
      "gdpr_compliant": true
    },
    "user_profile": { ... },
    "conversations": [ ... ],
    "sessions": [ ... ],
    "preferences": { ... }
  }
}
```

---

#### 12. Request Account Deletion

Request deletion of all user data (GDPR Article 17 - Right to be Forgotten).

**Endpoint**: `POST /api/gdpr/delete`
**Rate Limit**: 10 requests/minute
**Authentication**: Required (Bearer token)

**Request Body**:
```json
{
  "confirm": true,
  "reason": "No longer using the service"
}
```

**Response** (200 OK):
```json
{
  "status": "scheduled",
  "scheduled_at": "2025-11-23T15:00:00Z",
  "completion_date": "2025-12-23T15:00:00Z",
  "items_to_delete": [
    "user_profile",
    "conversations",
    "sessions",
    "preferences",
    "metadata"
  ]
}
```

**Important**:
- **30-day grace period** before permanent deletion
- During grace period, contact support to cancel deletion
- After grace period, deletion is **irreversible**

---

#### 13. Cancel Account Deletion

Cancel a pending account deletion request (only during 30-day grace period).

**Endpoint**: `POST /api/gdpr/delete/cancel`
**Rate Limit**: 10 requests/minute
**Authentication**: Required (Bearer token)

**Response** (204 No Content): Empty body

---

## Error Responses

All endpoints return standardized error responses:

```json
{
  "detail": "Error message description"
}
```

### HTTP Status Codes

| Code | Meaning | Common Causes |
|------|---------|---------------|
| `400` | Bad Request | Invalid input, validation failed |
| `401` | Unauthorized | Invalid/expired token, wrong credentials |
| `403` | Forbidden | Insufficient permissions |
| `404` | Not Found | Resource doesn't exist |
| `409` | Conflict | Email already registered |
| `423` | Locked | Account locked (too many failed attempts) |
| `429` | Too Many Requests | Rate limit exceeded |
| `500` | Internal Server Error | Server-side error |

### Rate Limit Headers

When rate limited, responses include:
```
HTTP/1.1 429 Too Many Requests
Retry-After: 45
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1700751600
```

---

## Security Considerations

### Token Security

1. **Access Tokens**:
   - Short-lived (15 minutes)
   - Use for API authentication
   - Never store in localStorage (use memory or secure httpOnly cookie)
   - Always sent via `Authorization: Bearer` header

2. **Refresh Tokens**:
   - Long-lived (7 days)
   - Use only for token refresh
   - Store securely (httpOnly cookie recommended)
   - Invalidated on logout

3. **Token Blacklisting**:
   - Logged out tokens are blacklisted
   - Blacklisted tokens rejected even if not expired
   - Logout from all devices: blacklist both access and refresh tokens

### Password Requirements

- Minimum 8 characters
- At least one uppercase letter
- At least one lowercase letter
- At least one digit
- No common passwords (e.g., "Password123")

### Account Security

- **Lockout**: 5 failed login attempts = 15 minute lockout
- **Password Hashing**: bcrypt with automatic salt generation
- **Session Management**: Stateless JWT (no server-side sessions)

### Rate Limiting

- **Authentication endpoints**: 10 req/min (brute-force protection)
- **General API**: 100 req/min
- **Health checks**: 1000 req/min
- **Enforcement**: Per-IP basis (X-Forwarded-For header supported)

---

## Code Examples

### Python (requests)

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Register
response = requests.post(
    f"{BASE_URL}/api/auth/register",
    json={
        "email": "test@example.com",
        "password": "SecurePass123",
        "full_name": "Test User"
    }
)
user = response.json()
print(f"User created: {user['id']}")

# 2. Login
response = requests.post(
    f"{BASE_URL}/api/auth/login",
    data={
        "username": "test@example.com",
        "password": "SecurePass123"
    }
)
tokens = response.json()
access_token = tokens["access_token"]
refresh_token = tokens["refresh_token"]

# 3. Authenticated API call
headers = {"Authorization": f"Bearer {access_token}"}
response = requests.get(f"{BASE_URL}/api/user/me", headers=headers)
profile = response.json()
print(f"Logged in as: {profile['email']}")

# 4. Refresh token
response = requests.post(
    f"{BASE_URL}/api/auth/refresh",
    json={"refresh_token": refresh_token}
)
new_tokens = response.json()
access_token = new_tokens["access_token"]

# 5. Logout
headers = {"Authorization": f"Bearer {access_token}"}
response = requests.post(
    f"{BASE_URL}/api/auth/logout",
    headers=headers,
    json={"refresh_token": refresh_token}
)
print("Logged out successfully")
```

### JavaScript (fetch)

```javascript
const BASE_URL = "http://localhost:8000";

// 1. Register
const registerResponse = await fetch(`${BASE_URL}/api/auth/register`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    email: "test@example.com",
    password: "SecurePass123",
    full_name: "Test User"
  })
});
const user = await registerResponse.json();
console.log(`User created: ${user.id}`);

// 2. Login (JSON-based)
const loginResponse = await fetch(`${BASE_URL}/api/auth/login/json`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    email: "test@example.com",
    password: "SecurePass123"
  })
});
const tokens = await loginResponse.json();
let accessToken = tokens.access_token;
let refreshToken = tokens.refresh_token;

// 3. Authenticated API call
const profileResponse = await fetch(`${BASE_URL}/api/user/me`, {
  headers: { "Authorization": `Bearer ${accessToken}` }
});
const profile = await profileResponse.json();
console.log(`Logged in as: ${profile.email}`);

// 4. Auto-refresh helper
async function fetchWithAuth(url, options = {}) {
  let response = await fetch(url, {
    ...options,
    headers: {
      ...options.headers,
      "Authorization": `Bearer ${accessToken}`
    }
  });

  // If token expired, refresh and retry
  if (response.status === 401) {
    const refreshResponse = await fetch(`${BASE_URL}/api/auth/refresh`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: refreshToken })
    });
    const newTokens = await refreshResponse.json();
    accessToken = newTokens.access_token;
    refreshToken = newTokens.refresh_token;

    // Retry original request
    response = await fetch(url, {
      ...options,
      headers: {
        ...options.headers,
        "Authorization": `Bearer ${accessToken}`
      }
    });
  }

  return response;
}

// 5. Logout
await fetch(`${BASE_URL}/api/auth/logout`, {
  method: "POST",
  headers: {
    "Authorization": `Bearer ${accessToken}`,
    "Content-Type": "application/json"
  },
  body: JSON.stringify({ refresh_token: refreshToken })
});
console.log("Logged out successfully");
```

### cURL

```bash
# 1. Register
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"SecurePass123","full_name":"Test User"}'

# 2. Login (form-based)
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d 'username=test@example.com&password=SecurePass123'

# 3. Get profile (replace TOKEN with actual access token)
curl http://localhost:8000/api/user/me \
  -H "Authorization: Bearer TOKEN"

# 4. Refresh token (replace REFRESH_TOKEN)
curl -X POST http://localhost:8000/api/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{"refresh_token":"REFRESH_TOKEN"}'

# 5. Logout
curl -X POST http://localhost:8000/api/auth/logout \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"refresh_token":"REFRESH_TOKEN"}'
```

---

## WebSocket Authentication

WebSocket connections also require JWT authentication.

### Pass Token as Query Parameter

```javascript
const token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...";
const sessionId = "session-123";
const ws = new WebSocket(`ws://localhost:8000/ws/${sessionId}?token=${token}`);
```

### Pass Token in Authorization Header

```javascript
const token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...";
const sessionId = "session-123";
const ws = new WebSocket(`ws://localhost:8000/ws/${sessionId}`, {
  headers: {
    "Authorization": `Bearer ${token}`
  }
});
```

### Authentication Failure

If authentication fails, the WebSocket connection is closed with:
- **Close Code**: 4001
- **Reason**: "Authentication failed"

---

## Environment Variables

Configure JWT settings via environment variables:

```bash
# Required (change in production!)
JWT_SECRET_KEY=your-production-secret-key-change-this

# Optional (has defaults)
JWT_ALGORITHM=HS256
JWT_ACCESS_EXPIRE_MINUTES=15
JWT_REFRESH_EXPIRE_DAYS=7
```

---

## Migration from Unauthenticated API

If upgrading from a previous version without authentication:

1. **Backend Changes**:
   - Add `from app.security.dependencies import get_current_user` to route files
   - Add `user: User = Depends(get_current_user)` to protected endpoints
   - Public endpoints: health checks, legal docs, auth endpoints

2. **Frontend Changes**:
   - Implement login flow
   - Store tokens securely
   - Add `Authorization: Bearer <token>` header to all API requests
   - Handle 401 responses (token expired → refresh)
   - Implement logout

3. **Testing**:
   - Register test users
   - Verify protected endpoints reject unauthenticated requests
   - Verify authenticated requests work correctly
   - Test token refresh flow
   - Test rate limiting

---

## Support

For issues or questions:
- GitHub Issues: https://github.com/bjpl/learning_voice_agent/issues
- Documentation: https://github.com/bjpl/learning_voice_agent/tree/main/docs
- Security Contact: security@example.com (for vulnerability reports)

---

**Version History**:
- v1.0.0 (2025-11-23): Initial JWT authentication implementation
