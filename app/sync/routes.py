"""
Sync API Routes - FastAPI Router
================================

RESTful API endpoints for data synchronization and backup.

PATTERN: FastAPI router with dependency injection
WHY: Modular routing with proper request/response handling
"""

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Query
from fastapi.responses import Response, StreamingResponse
from typing import Optional
from datetime import datetime
import io

try:
    from app.logger import api_logger as logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from app.sync.models import (
    SyncStatusResponse,
    ExportRequest,
    ExportResponse,
    ImportRequest,
    ImportResponse,
    DeviceRegistrationRequest,
    DeviceInfo,
    DeviceListResponse,
    ValidationResponse,
    ConflictListResponse,
    ConflictResolutionRequest,
    ConflictResolutionResponse,
    ScheduleBackupRequest,
    ScheduleBackupResponse,
    MergeStrategy,
    ConflictResolution,
    ExtendedSyncStatusResponse,
)
from app.sync.conflict_resolver import ResolutionStrategy
from app.sync.service import sync_service
from app.sync.scheduler import (
    backup_scheduler,
    schedule_auto_backup,
    get_next_backup_time,
    cancel_scheduled_backup,
)

# Create router with prefix and tags
router = APIRouter(
    prefix="/api/sync",
    tags=["sync"],
    responses={
        500: {"description": "Internal server error"},
        429: {"description": "Rate limit exceeded"}
    }
)


# =============================================================================
# Status Endpoints
# =============================================================================

@router.get(
    "/status",
    response_model=SyncStatusResponse,
    summary="Get sync status",
    description="Get current sync status, last backup time, and device count."
)
async def get_sync_status() -> SyncStatusResponse:
    """
    Get sync status and last backup time.

    PATTERN: Status endpoint for health monitoring
    WHY: Allow clients to check sync state before operations

    Returns:
        SyncStatusResponse with current sync state
    """
    try:
        logger.info("sync_status_requested")

        status = await sync_service.get_status()

        logger.debug(
            "sync_status_retrieved",
            status=status.status,
            device_count=status.device_count
        )

        return status

    except Exception as e:
        logger.error("sync_status_error", error=str(e))
        raise HTTPException(500, f"Failed to get sync status: {str(e)}")


# =============================================================================
# Export Endpoints
# =============================================================================

@router.post(
    "/export",
    response_model=ExportResponse,
    summary="Export data as backup",
    description="Export all data as a downloadable backup file."
)
async def export_data(
    request: Optional[ExportRequest] = None
) -> ExportResponse:
    """
    Export data as downloadable backup file.

    PATTERN: Async export with compression
    WHY: Enable data portability and backup

    Args:
        request: Export options (optional)

    Returns:
        ExportResponse with download URL
    """
    try:
        logger.info("data_export_requested")

        # Use defaults if no request provided
        if request is None:
            request = ExportRequest()

        response, _ = await sync_service.export_data(
            include_sessions=request.include_sessions,
            include_feedback=request.include_feedback,
            include_analytics=request.include_analytics,
            include_goals=request.include_goals,
            include_files=request.include_files,
            compress=request.compression
        )

        logger.info(
            "data_export_completed",
            export_id=response.export_id,
            record_count=response.record_count
        )

        return response

    except Exception as e:
        logger.error("data_export_error", error=str(e))
        raise HTTPException(500, f"Failed to export data: {str(e)}")


@router.get(
    "/download/{export_id}",
    summary="Download backup file",
    description="Download a previously exported backup file."
)
async def download_backup(export_id: str) -> Response:
    """
    Download a backup file by export ID.

    Args:
        export_id: Export ID from export response

    Returns:
        File download response
    """
    try:
        logger.info("backup_download_requested", export_id=export_id)

        result = await sync_service.get_backup_file(export_id)

        if result is None:
            raise HTTPException(404, f"Backup not found: {export_id}")

        file_bytes, filename = result

        # Determine media type
        if filename.endswith('.gz'):
            media_type = "application/gzip"
        else:
            media_type = "application/json"

        logger.info(
            "backup_download_served",
            export_id=export_id,
            filename=filename
        )

        return Response(
            content=file_bytes,
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("backup_download_error", export_id=export_id, error=str(e))
        raise HTTPException(500, f"Failed to download backup: {str(e)}")


# =============================================================================
# Import Endpoints
# =============================================================================

@router.post(
    "/import",
    response_model=ImportResponse,
    summary="Import/restore from backup",
    description="Import data from a backup file. Supports merge strategies for conflicts."
)
async def import_data(
    file: UploadFile = File(..., description="Backup file to import"),
    merge_strategy: str = Query("keep_remote", regex="^(keep_local|keep_remote|merge|manual)$"),
    dry_run: bool = Query(False, description="Validate without importing"),
    skip_conflicts: bool = Query(False, description="Skip conflicting records")
) -> ImportResponse:
    """
    Import/restore from backup file.

    PATTERN: Transactional import with conflict handling
    WHY: Safe data restoration with rollback capability

    Args:
        file: Backup file upload
        merge_strategy: How to handle existing data
        dry_run: Validate without importing
        skip_conflicts: Skip conflicting records

    Returns:
        ImportResponse with import results
    """
    try:
        logger.info(
            "data_import_requested",
            filename=file.filename,
            merge_strategy=merge_strategy,
            dry_run=dry_run
        )

        # Read file data
        backup_data = await file.read()

        # Convert strategy string to enum
        strategy = ResolutionStrategy(merge_strategy)

        response = await sync_service.import_data(
            backup_data=backup_data,
            merge_strategy=strategy,
            dry_run=dry_run,
            skip_conflicts=skip_conflicts
        )

        logger.info(
            "data_import_completed",
            success=response.success,
            imported=response.imported_count,
            conflicts=response.conflicts_count
        )

        return response

    except ValueError as e:
        logger.warning("data_import_validation_error", error=str(e))
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error("data_import_error", error=str(e))
        raise HTTPException(500, f"Failed to import data: {str(e)}")


# =============================================================================
# Validation Endpoint
# =============================================================================

@router.post(
    "/validate",
    response_model=ValidationResponse,
    summary="Validate backup file",
    description="Validate a backup file without importing it."
)
async def validate_backup(
    file: UploadFile = File(..., description="Backup file to validate")
) -> ValidationResponse:
    """
    Validate backup file without importing.

    PATTERN: Pre-import validation
    WHY: Prevent data corruption from invalid backups

    Args:
        file: Backup file to validate

    Returns:
        ValidationResponse with validation results
    """
    try:
        logger.info("backup_validation_requested", filename=file.filename)

        backup_data = await file.read()

        response = await sync_service.validate_backup(backup_data)

        logger.info(
            "backup_validation_completed",
            valid=response.valid,
            record_count=response.record_count
        )

        return response

    except Exception as e:
        logger.error("backup_validation_error", error=str(e))
        raise HTTPException(500, f"Failed to validate backup: {str(e)}")


# =============================================================================
# Device Management Endpoints
# =============================================================================

@router.get(
    "/devices",
    response_model=DeviceListResponse,
    summary="List registered devices",
    description="Get list of all devices registered for sync."
)
async def list_devices() -> DeviceListResponse:
    """
    List registered devices.

    Returns:
        DeviceListResponse with all devices
    """
    try:
        logger.info("devices_list_requested")

        response = await sync_service.get_devices()

        logger.debug("devices_listed", count=response.total_count)

        return response

    except Exception as e:
        logger.error("devices_list_error", error=str(e))
        raise HTTPException(500, f"Failed to list devices: {str(e)}")


@router.post(
    "/devices/register",
    response_model=DeviceInfo,
    summary="Register current device",
    description="Register the current device for sync."
)
async def register_device(
    request: DeviceRegistrationRequest
) -> DeviceInfo:
    """
    Register current device.

    Args:
        request: Device registration details

    Returns:
        DeviceInfo for the registered device
    """
    try:
        logger.info(
            "device_registration_requested",
            device_name=request.device_name,
            device_type=request.device_type
        )

        device = await sync_service.register_device(
            device_name=request.device_name,
            device_type=request.device_type,
            platform=request.platform,
            app_version=request.app_version
        )

        logger.info(
            "device_registered",
            device_id=device.device_id,
            device_name=device.device_name
        )

        return device

    except ValueError as e:
        logger.warning("device_registration_validation_error", error=str(e))
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error("device_registration_error", error=str(e))
        raise HTTPException(500, f"Failed to register device: {str(e)}")


@router.delete(
    "/devices/{device_id}",
    summary="Remove device",
    description="Remove a registered device from sync."
)
async def remove_device(device_id: str) -> dict:
    """
    Remove a registered device.

    Args:
        device_id: Device ID to remove

    Returns:
        Success status
    """
    try:
        logger.info("device_removal_requested", device_id=device_id)

        removed = await sync_service.remove_device(device_id)

        if not removed:
            raise HTTPException(404, f"Device not found: {device_id}")

        logger.info("device_removed", device_id=device_id)

        return {
            "success": True,
            "message": "Device removed",
            "device_id": device_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("device_removal_error", device_id=device_id, error=str(e))
        raise HTTPException(500, f"Failed to remove device: {str(e)}")


# =============================================================================
# Conflict Management Endpoints
# =============================================================================

@router.get(
    "/conflicts",
    response_model=ConflictListResponse,
    summary="Get pending conflicts",
    description="Get list of pending data conflicts that need resolution."
)
async def get_conflicts() -> ConflictListResponse:
    """
    Get pending conflicts.

    Returns:
        ConflictListResponse with all pending conflicts
    """
    try:
        logger.info("conflicts_requested")

        response = await sync_service.get_conflicts()

        logger.debug("conflicts_retrieved", count=response.total_count)

        return response

    except Exception as e:
        logger.error("conflicts_get_error", error=str(e))
        raise HTTPException(500, f"Failed to get conflicts: {str(e)}")


@router.post(
    "/conflicts/resolve",
    response_model=ConflictResolutionResponse,
    summary="Resolve conflicts",
    description="Resolve one or more data conflicts."
)
async def resolve_conflicts(
    request: ConflictResolutionRequest
) -> ConflictResolutionResponse:
    """
    Resolve conflicts.

    Args:
        request: Conflict resolution details

    Returns:
        ConflictResolutionResponse with results
    """
    try:
        logger.info(
            "conflict_resolution_requested",
            conflict_id=request.conflict_id,
            strategy=request.strategy
        )

        response = await sync_service.resolve_conflicts(
            conflict_id=request.conflict_id,
            conflict_ids=request.conflict_ids,
            strategy=ResolutionStrategy(request.strategy) if isinstance(request.strategy, str) else request.strategy,
            custom_value=request.custom_value
        )

        logger.info(
            "conflicts_resolved",
            resolved=response.resolved_count,
            remaining=response.remaining_conflicts
        )

        return response

    except ValueError as e:
        logger.warning("conflict_resolution_validation_error", error=str(e))
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error("conflict_resolution_error", error=str(e))
        raise HTTPException(500, f"Failed to resolve conflicts: {str(e)}")


# =============================================================================
# Scheduler Endpoints
# =============================================================================

@router.post(
    "/schedule",
    response_model=ScheduleBackupResponse,
    summary="Schedule auto-backup",
    description="Configure automatic backup scheduling."
)
async def configure_backup_schedule(
    request: ScheduleBackupRequest
) -> ScheduleBackupResponse:
    """
    Schedule automatic backups.

    Args:
        request: Scheduling configuration

    Returns:
        ScheduleBackupResponse with scheduling status
    """
    try:
        logger.info(
            "backup_schedule_requested",
            interval_hours=request.interval_hours,
            enabled=request.enabled
        )

        # Set backup callback
        async def backup_callback() -> bool:
            try:
                _, _ = await sync_service.export_data()
                return True
            except Exception as e:
                logger.error("scheduled_backup_failed", error=str(e))
                return False

        backup_scheduler.set_backup_callback(backup_callback)

        result = await schedule_auto_backup(
            interval_hours=request.interval_hours,
            enabled=request.enabled
        )

        logger.info(
            "backup_schedule_configured",
            enabled=result['enabled'],
            next_backup=result.get('next_backup')
        )

        return ScheduleBackupResponse(**result)

    except ValueError as e:
        logger.warning("backup_schedule_validation_error", error=str(e))
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error("backup_schedule_error", error=str(e))
        raise HTTPException(500, f"Failed to configure backup schedule: {str(e)}")


@router.delete(
    "/schedule",
    summary="Cancel scheduled backup",
    description="Cancel automatic backup scheduling."
)
async def cancel_backup_schedule() -> dict:
    """
    Cancel scheduled auto-backup.

    Returns:
        Success status
    """
    try:
        logger.info("backup_schedule_cancellation_requested")

        result = await cancel_scheduled_backup()

        logger.info("backup_schedule_cancelled")

        return result

    except Exception as e:
        logger.error("backup_schedule_cancel_error", error=str(e))
        raise HTTPException(500, f"Failed to cancel backup schedule: {str(e)}")


@router.get(
    "/schedule/next",
    summary="Get next backup time",
    description="Get the next scheduled backup time."
)
async def get_next_backup() -> dict:
    """
    Get next scheduled backup time.

    Returns:
        Next backup time info
    """
    try:
        next_time = get_next_backup_time()

        return {
            "next_backup": next_time.isoformat() if next_time else None,
            "enabled": backup_scheduler.is_enabled,
            "interval_hours": backup_scheduler.interval_hours
        }

    except Exception as e:
        logger.error("next_backup_time_error", error=str(e))
        raise HTTPException(500, f"Failed to get next backup time: {str(e)}")


# =============================================================================
# Router Setup Function
# =============================================================================

def setup_sync_routes(app):
    """
    Setup sync routes on the FastAPI app.

    Args:
        app: FastAPI application instance
    """
    app.include_router(router)
    logger.info("sync_routes_registered")
