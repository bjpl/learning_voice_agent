"""
Security Dependencies - FastAPI Dependency Injection

SPARC Implementation:
- Specification: Reusable security dependencies for routes
- Architecture: Composable auth and permission checks
"""

import logging
from typing import Optional
from fastapi import Depends, HTTPException, status, WebSocket, Query
from fastapi.security import OAuth2PasswordBearer

from app.security.auth import (
    auth_service,
    verify_token,
    get_current_user,
    get_current_active_user,
)
from app.security.models import User, UserRole, TokenData

logger = logging.getLogger(__name__)


def require_auth(func):
    """
    Decorator to require authentication on an endpoint.

    Usage:
        @app.get("/protected")
        @require_auth
        async def protected_endpoint(user: User = Depends(get_current_active_user)):
            return {"user": user.email}
    """
    return func


def require_admin(user: User = Depends(get_current_active_user)) -> User:
    """
    Dependency that requires admin role.

    Usage:
        @app.get("/admin/users")
        async def list_users(user: User = Depends(require_admin)):
            ...
    """
    if user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user


def require_role(*roles: UserRole):
    """
    Factory for role-based access control dependency.

    Usage:
        @app.get("/api/reports")
        async def get_reports(user: User = Depends(require_role(UserRole.ADMIN, UserRole.API_CLIENT))):
            ...
    """
    async def role_checker(user: User = Depends(get_current_active_user)) -> User:
        if user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of: {[r.value for r in roles]}",
            )
        return user

    return role_checker


async def websocket_auth(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
) -> TokenData:
    """
    WebSocket authentication dependency.

    Validates token before accepting WebSocket connection.
    Token can be passed as query parameter: /ws/{session_id}?token=xxx

    Usage:
        @app.websocket("/ws/{session_id}")
        async def ws_endpoint(
            websocket: WebSocket,
            session_id: str,
            token_data: TokenData = Depends(websocket_auth),
        ):
            ...
    """
    if not token:
        # Check for token in headers (for browsers that support it)
        token = websocket.headers.get("Authorization", "").replace("Bearer ", "")

    if not token:
        logger.warning("WebSocket connection attempted without token")
        await websocket.close(code=4001, reason="Authentication required")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    try:
        token_data = await auth_service.verify_token(token)
        return token_data
    except HTTPException as e:
        logger.warning(f"WebSocket authentication failed: {e.detail}")
        await websocket.close(code=4001, reason=str(e.detail))
        raise


async def websocket_auth_optional(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
) -> Optional[TokenData]:
    """
    Optional WebSocket authentication.

    Returns None if no token provided, allowing anonymous connections
    while still validating tokens when present.
    """
    if not token:
        token = websocket.headers.get("Authorization", "").replace("Bearer ", "")

    if not token:
        return None

    try:
        return await auth_service.verify_token(token)
    except HTTPException:
        return None


class SessionOwnershipChecker:
    """
    Verify that a user owns a specific session.

    Usage:
        ownership = SessionOwnershipChecker()

        @app.get("/api/session/{session_id}")
        async def get_session(
            session_id: str,
            user: User = Depends(ownership.verify),
        ):
            ...
    """

    def __init__(self, session_store=None):
        self._session_store = session_store

    async def verify(
        self,
        session_id: str,
        user: User = Depends(get_current_active_user),
    ) -> User:
        """Verify user owns the session."""
        # Get session from store
        if self._session_store:
            session = await self._session_store.get_session(session_id)
            if session and session.get("user_id") != user.id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You don't have access to this session",
                )

        return user


class APIKeyAuth:
    """
    API Key authentication for service-to-service communication.

    Usage:
        api_key_auth = APIKeyAuth()

        @app.post("/api/webhook")
        async def webhook(api_key: str = Depends(api_key_auth)):
            ...
    """

    def __init__(self, header_name: str = "X-API-Key"):
        self.header_name = header_name
        self._valid_keys: set = set()

    def add_key(self, key: str) -> None:
        """Add a valid API key."""
        self._valid_keys.add(key)

    def remove_key(self, key: str) -> None:
        """Remove an API key."""
        self._valid_keys.discard(key)

    async def __call__(
        self,
        request,
    ) -> str:
        """Validate API key from request header."""
        api_key = request.headers.get(self.header_name)

        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Missing {self.header_name} header",
            )

        if api_key not in self._valid_keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )

        return api_key


class CombinedAuth:
    """
    Combined authentication supporting both JWT and API key.

    Useful for endpoints that need to support both user and service access.
    """

    def __init__(self, api_key_auth: Optional[APIKeyAuth] = None):
        self.api_key_auth = api_key_auth or APIKeyAuth()

    async def __call__(
        self,
        request,
        token: Optional[str] = Depends(OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)),
    ) -> dict:
        """
        Authenticate via JWT or API key.

        Returns:
            Dict with auth_type ("jwt" or "api_key") and identity info
        """
        # Try JWT first
        if token:
            try:
                token_data = await auth_service.verify_token(token)
                return {
                    "auth_type": "jwt",
                    "user_id": token_data.sub,
                    "email": token_data.email,
                    "role": token_data.role,
                }
            except HTTPException:
                pass  # Fall through to API key

        # Try API key
        api_key = request.headers.get(self.api_key_auth.header_name)
        if api_key and api_key in self.api_key_auth._valid_keys:
            return {
                "auth_type": "api_key",
                "api_key": api_key,
            }

        # Neither worked
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required (JWT or API key)",
            headers={"WWW-Authenticate": "Bearer"},
        )
