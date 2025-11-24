"""
Security Routes - Authentication and GDPR Endpoints

SPARC Implementation:
- Authentication: Register, login, logout, refresh
- GDPR: Data export, deletion, user rights
"""

import json
import uuid
import logging
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm

from app.security.auth import (
    auth_service,
    get_current_user,
    get_current_active_user,
    SecurityConfig,
)
from app.security.models import (
    User,
    UserCreate,
    UserLogin,
    UserResponse,
    UserUpdate,
    PasswordChange,
    Token,
    TokenRefresh,
    GDPRExportRequest,
    GDPRExportResponse,
    GDPRDeleteRequest,
    GDPRDeleteResponse,
    SessionInfo,
)
from app.security.rate_limit import rate_limit, get_rate_limiter

logger = logging.getLogger(__name__)

# Create routers
auth_router = APIRouter(prefix="/api/auth", tags=["authentication"])
gdpr_router = APIRouter(prefix="/api/gdpr", tags=["gdpr"])
user_router = APIRouter(prefix="/api/user", tags=["user"])


# ============================================================================
# Authentication Routes
# ============================================================================

@auth_router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    request: Request,
):
    """
    Register a new user account.

    - **email**: Valid email address (required)
    - **password**: Min 8 chars with uppercase, lowercase, and digit (required)
    - **full_name**: Display name (optional)

    Returns the created user profile (without sensitive data).
    """
    # Rate limit check
    limiter = get_rate_limiter()
    await limiter.check_rate_limit(request)

    user = await auth_service.create_user(user_data)

    logger.info(f"New user registered: {user.id} from {request.client.host}")

    return user


@auth_router.post("/login", response_model=Token)
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
):
    """
    Authenticate user and return access/refresh tokens.

    Uses OAuth2 password flow for compatibility with OpenAPI.
    Submit as form data with `username` (email) and `password` fields.

    Returns:
    - **access_token**: Short-lived JWT for API access
    - **refresh_token**: Long-lived JWT for token refresh
    - **token_type**: Always "bearer"
    - **expires_in**: Access token validity in seconds
    """
    # Rate limit check (stricter for auth)
    limiter = get_rate_limiter()
    await limiter.check_rate_limit(request)

    credentials = UserLogin(email=form_data.username, password=form_data.password)
    user, token = await auth_service.authenticate_user(credentials)

    logger.info(f"User logged in: {user.id} from {request.client.host}")

    return token


@auth_router.post("/login/json", response_model=Token)
async def login_json(
    request: Request,
    credentials: UserLogin,
):
    """
    JSON-based login endpoint (alternative to form-based).

    Accepts JSON body instead of form data.
    """
    limiter = get_rate_limiter()
    await limiter.check_rate_limit(request)

    user, token = await auth_service.authenticate_user(credentials)

    logger.info(f"User logged in: {user.id} from {request.client.host}")

    return token


@auth_router.post("/refresh", response_model=Token)
async def refresh_token(
    request: Request,
    token_data: TokenRefresh,
):
    """
    Refresh access token using refresh token.

    Use this endpoint when your access token expires to get a new one
    without requiring the user to log in again.
    """
    limiter = get_rate_limiter()
    await limiter.check_rate_limit(request)

    return await auth_service.refresh_access_token(token_data.refresh_token)


@auth_router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    request: Request,
    user: User = Depends(get_current_active_user),
):
    """
    Logout user and invalidate tokens.

    Blacklists the current access token. Optionally include
    refresh_token in body to invalidate that as well.
    """
    # Get tokens from request
    auth_header = request.headers.get("Authorization", "")
    access_token = auth_header.replace("Bearer ", "") if auth_header else None

    # Try to get refresh token from body
    refresh_token = None
    try:
        body = await request.json()
        refresh_token = body.get("refresh_token")
    except Exception:
        pass

    if access_token:
        await auth_service.logout(access_token, refresh_token)

    logger.info(f"User logged out: {user.id}")


# ============================================================================
# User Profile Routes
# ============================================================================

@user_router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    user: User = Depends(get_current_active_user),
):
    """Get the current user's profile."""
    return UserResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        status=user.status,
        created_at=user.created_at,
        last_login=user.last_login,
    )


@user_router.patch("/me", response_model=UserResponse)
async def update_profile(
    updates: UserUpdate,
    user: User = Depends(get_current_active_user),
):
    """Update the current user's profile."""
    if updates.full_name is not None:
        user.full_name = updates.full_name
    if updates.email is not None:
        # Check email not already in use
        existing = await auth_service.get_user_by_email(updates.email)
        if existing and existing.id != user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already in use",
            )
        user.email = updates.email

    user.updated_at = datetime.utcnow()

    return UserResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        status=user.status,
        created_at=user.created_at,
        last_login=user.last_login,
    )


@user_router.post("/me/password", status_code=status.HTTP_204_NO_CONTENT)
async def change_password(
    password_data: PasswordChange,
    user: User = Depends(get_current_active_user),
):
    """Change the current user's password."""
    # Verify current password
    if not auth_service.verify_password(password_data.current_password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )

    # Update password
    user.hashed_password = auth_service.hash_password(password_data.new_password)
    user.updated_at = datetime.utcnow()

    logger.info(f"Password changed for user: {user.id}")


# ============================================================================
# GDPR Compliance Routes
# ============================================================================

@gdpr_router.post("/export", response_model=GDPRExportResponse)
async def request_data_export(
    request: GDPRExportRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_active_user),
):
    """
    Request export of all user data (GDPR Article 20).

    This endpoint initiates an asynchronous export of all data
    associated with your account. You will receive a download
    link when the export is ready.

    Supported formats: json, csv
    """
    export_id = str(uuid.uuid4())

    # Queue export task
    background_tasks.add_task(
        process_data_export,
        export_id,
        user.id,
        request,
    )

    logger.info(f"Data export requested: {export_id} for user {user.id}")

    return GDPRExportResponse(
        export_id=export_id,
        status="processing",
        created_at=datetime.utcnow(),
        download_url=None,
        expires_at=datetime.utcnow() + timedelta(hours=24),
    )


@gdpr_router.get("/export/{export_id}", response_model=GDPRExportResponse)
async def get_export_status(
    export_id: str,
    user: User = Depends(get_current_active_user),
):
    """
    Check the status of a data export request.
    """
    # In production, check database for export status
    # For now, return mock completed status
    return GDPRExportResponse(
        export_id=export_id,
        status="completed",
        created_at=datetime.utcnow() - timedelta(minutes=5),
        download_url=f"/api/gdpr/export/{export_id}/download",
        expires_at=datetime.utcnow() + timedelta(hours=24),
    )


@gdpr_router.get("/export/{export_id}/download")
async def download_export(
    export_id: str,
    user: User = Depends(get_current_active_user),
):
    """
    Download completed data export.
    """
    # Generate export data
    export_data = await generate_user_export(user.id)

    return {
        "export_id": export_id,
        "format": "json",
        "data": export_data,
    }


@gdpr_router.post("/delete", response_model=GDPRDeleteResponse)
async def request_account_deletion(
    request: GDPRDeleteRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_active_user),
):
    """
    Request deletion of all user data (GDPR Article 17).

    This endpoint initiates the deletion of your account and
    all associated data. This action is irreversible.

    A 30-day grace period applies before permanent deletion.
    During this time, you can contact support to cancel.

    **Warning**: This will delete:
    - Your user profile
    - All conversations and sessions
    - All preferences and settings
    - All associated metadata
    """
    if not request.confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You must confirm the deletion request",
        )

    # Schedule deletion
    background_tasks.add_task(
        process_account_deletion,
        user.id,
        request.reason,
    )

    scheduled_at = datetime.utcnow()
    completion_date = scheduled_at + timedelta(days=30)

    logger.info(f"Account deletion requested: {user.id}, reason: {request.reason}")

    return GDPRDeleteResponse(
        status="scheduled",
        scheduled_at=scheduled_at,
        completion_date=completion_date,
        items_to_delete=[
            "user_profile",
            "conversations",
            "sessions",
            "preferences",
            "metadata",
        ],
    )


@gdpr_router.post("/delete/cancel", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_deletion(
    user: User = Depends(get_current_active_user),
):
    """
    Cancel a pending account deletion request.

    Only works during the 30-day grace period before permanent deletion.
    """
    # In production, check for pending deletion and cancel
    logger.info(f"Deletion cancelled for user: {user.id}")


# ============================================================================
# Helper Functions
# ============================================================================

async def process_data_export(
    export_id: str,
    user_id: str,
    request: GDPRExportRequest,
) -> None:
    """Background task to process data export."""
    try:
        logger.info(f"Processing export {export_id} for user {user_id}")

        # Generate export
        export_data = await generate_user_export(user_id)

        # In production: save to storage and update database
        logger.info(f"Export {export_id} completed")

    except Exception as e:
        logger.error(f"Export {export_id} failed: {e}")


async def generate_user_export(user_id: str) -> dict:
    """Generate complete user data export."""
    user = await auth_service.get_user(user_id)

    if not user:
        return {}

    return {
        "export_metadata": {
            "format_version": "1.0.0",
            "generated_at": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "gdpr_compliant": True,
        },
        "user_profile": {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "created_at": user.created_at.isoformat(),
            "last_login": user.last_login.isoformat() if user.last_login else None,
        },
        "conversations": [],  # Would be populated from database
        "sessions": [],  # Would be populated from state manager
        "preferences": {},  # Would be populated from preferences store
    }


async def process_account_deletion(
    user_id: str,
    reason: Optional[str],
) -> None:
    """Background task to process account deletion."""
    try:
        logger.info(f"Processing deletion for user {user_id}, reason: {reason}")

        # Create audit log entry
        # In production: log to compliance database

        # Mark user as deleted (soft delete)
        user = await auth_service.get_user(user_id)
        if user:
            from app.security.models import UserStatus
            user.status = UserStatus.DELETED
            user.updated_at = datetime.utcnow()

        # Schedule hard delete after retention period
        # In production: use task queue like Celery

        logger.info(f"User {user_id} marked for deletion")

    except Exception as e:
        logger.error(f"Deletion processing failed for {user_id}: {e}")


def setup_security_routes(app):
    """Setup all security routes on the FastAPI app."""
    app.include_router(auth_router)
    app.include_router(user_router)
    app.include_router(gdpr_router)
