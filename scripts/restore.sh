#!/bin/bash
#
# restore.sh - Restore script for Learning Voice Agent
#
# Usage: ./scripts/restore.sh [--list] [--backup FILE] [--verify-only]
#
# Options:
#   --list         List available backups
#   --backup FILE  Restore from specific backup file
#   --verify-only  Only verify backup integrity, don't restore
#   --no-confirm   Skip confirmation prompt
#   --db-only      Only restore database
#
# Exit codes:
#   0 - Success
#   1 - Configuration error
#   2 - Backup not found
#   3 - Integrity check failed
#   4 - Restore failed

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="${PROJECT_ROOT}/backups"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Parse arguments
LIST_BACKUPS=false
BACKUP_FILE=""
VERIFY_ONLY=false
NO_CONFIRM=false
DB_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --list|-l)
            LIST_BACKUPS=true
            shift
            ;;
        --backup|-b)
            BACKUP_FILE="$2"
            shift 2
            ;;
        --verify-only|-v)
            VERIFY_ONLY=true
            shift
            ;;
        --no-confirm|-y)
            NO_CONFIRM=true
            shift
            ;;
        --db-only)
            DB_ONLY=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--list] [--backup FILE] [--verify-only] [--no-confirm] [--db-only]"
            echo ""
            echo "Options:"
            echo "  --list, -l       List available backups"
            echo "  --backup, -b     Restore from specific backup file"
            echo "  --verify-only    Only verify backup integrity"
            echo "  --no-confirm     Skip confirmation prompt"
            echo "  --db-only        Only restore database"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }

# List available backups
list_backups() {
    echo -e "${CYAN}=========================================="
    echo "Available Backups"
    echo -e "==========================================${NC}"

    if [ ! -d "$BACKUP_DIR" ]; then
        log_error "Backup directory not found: $BACKUP_DIR"
        exit 2
    fi

    local backup_files
    backup_files=$(ls -1t "${BACKUP_DIR}"/backup_*.tar* 2>/dev/null || true)

    if [ -z "$backup_files" ]; then
        log_warning "No backups found in $BACKUP_DIR"
        exit 0
    fi

    local index=1
    echo ""
    echo "  #   Date                   Size       File"
    echo "  --- ---------------------- ---------- ----------------------------------------"

    while IFS= read -r backup; do
        local filename
        filename=$(basename "$backup")

        # Extract timestamp from filename
        local timestamp
        timestamp=$(echo "$filename" | sed 's/backup_\([0-9]\{8\}_[0-9]\{6\}\).*/\1/')

        # Format date
        local formatted_date
        if [[ "$timestamp" =~ ^([0-9]{4})([0-9]{2})([0-9]{2})_([0-9]{2})([0-9]{2})([0-9]{2})$ ]]; then
            formatted_date="${BASH_REMATCH[1]}-${BASH_REMATCH[2]}-${BASH_REMATCH[3]} ${BASH_REMATCH[4]}:${BASH_REMATCH[5]}:${BASH_REMATCH[6]}"
        else
            formatted_date="Unknown"
        fi

        # Get file size
        local size
        size=$(du -h "$backup" | cut -f1)

        # Check if checksum exists
        local checksum_status=""
        if [ -f "${backup}.sha256" ]; then
            checksum_status=" [checksum]"
        fi

        printf "  %-3d %-22s %-10s %s%s\n" "$index" "$formatted_date" "$size" "$filename" "$checksum_status"
        index=$((index + 1))
    done <<< "$backup_files"

    echo ""
    echo "Usage: $0 --backup <filename>"
    echo ""
}

# Verify backup integrity
verify_backup() {
    local backup_file="$1"

    log_info "Verifying backup integrity..."

    # Check if file exists
    if [ ! -f "$backup_file" ]; then
        log_error "Backup file not found: $backup_file"
        return 1
    fi

    # Verify checksum if available
    local checksum_file="${backup_file}.sha256"
    if [ -f "$checksum_file" ]; then
        log_info "Verifying checksum..."

        local expected_checksum
        expected_checksum=$(cat "$checksum_file" | awk '{print $1}')

        local actual_checksum
        if command -v sha256sum &> /dev/null; then
            actual_checksum=$(sha256sum "$backup_file" | awk '{print $1}')
        elif command -v shasum &> /dev/null; then
            actual_checksum=$(shasum -a 256 "$backup_file" | awk '{print $1}')
        else
            log_warning "No checksum tool available, skipping verification"
            actual_checksum="$expected_checksum"
        fi

        if [ "$expected_checksum" != "$actual_checksum" ]; then
            log_error "Checksum mismatch!"
            log_error "Expected: $expected_checksum"
            log_error "Actual:   $actual_checksum"
            return 1
        fi

        log_success "Checksum verified"
    else
        log_warning "No checksum file found, skipping verification"
    fi

    # Verify archive integrity
    log_info "Verifying archive integrity..."

    if [[ "$backup_file" == *.tar.gz ]]; then
        if ! tar -tzf "$backup_file" > /dev/null 2>&1; then
            log_error "Archive is corrupted"
            return 1
        fi
    elif [[ "$backup_file" == *.tar ]]; then
        if ! tar -tf "$backup_file" > /dev/null 2>&1; then
            log_error "Archive is corrupted"
            return 1
        fi
    else
        log_error "Unknown archive format"
        return 1
    fi

    log_success "Archive integrity verified"

    # List contents
    log_info "Archive contents:"
    if [[ "$backup_file" == *.tar.gz ]]; then
        tar -tzf "$backup_file" | head -20
    else
        tar -tf "$backup_file" | head -20
    fi

    # Check for manifest
    local temp_dir
    temp_dir=$(mktemp -d)

    if [[ "$backup_file" == *.tar.gz ]]; then
        tar -xzf "$backup_file" -C "$temp_dir" --wildcards "*/manifest.json" 2>/dev/null || true
    else
        tar -xf "$backup_file" -C "$temp_dir" --wildcards "*/manifest.json" 2>/dev/null || true
    fi

    local manifest
    manifest=$(find "$temp_dir" -name "manifest.json" 2>/dev/null | head -1)

    if [ -n "$manifest" ] && [ -f "$manifest" ]; then
        log_info "Backup manifest:"
        cat "$manifest" | python3 -m json.tool 2>/dev/null || cat "$manifest"
    fi

    rm -rf "$temp_dir"

    return 0
}

# Get confirmation from user
confirm_restore() {
    if [ "$NO_CONFIRM" = true ]; then
        return 0
    fi

    echo ""
    echo -e "${YELLOW}WARNING: This will overwrite existing data!${NC}"
    echo ""
    read -p "Are you sure you want to restore from this backup? (yes/no): " response

    if [ "$response" != "yes" ]; then
        log_info "Restore cancelled"
        exit 0
    fi
}

# Create pre-restore backup
create_pre_restore_backup() {
    log_info "Creating pre-restore backup of current state..."

    local pre_restore_dir="${BACKUP_DIR}/pre_restore_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$pre_restore_dir"

    # Backup current database
    if [ -f "${PROJECT_ROOT}/learning_captures.db" ]; then
        cp "${PROJECT_ROOT}/learning_captures.db" "$pre_restore_dir/"
    fi

    if [ -f "${PROJECT_ROOT}/feedback.db" ]; then
        cp "${PROJECT_ROOT}/feedback.db" "$pre_restore_dir/"
    fi

    if [ -f "${PROJECT_ROOT}/goals.db" ]; then
        cp "${PROJECT_ROOT}/goals.db" "$pre_restore_dir/"
    fi

    log_success "Pre-restore backup created at: $pre_restore_dir"
}

# Restore database
restore_database() {
    local extract_dir="$1"

    log_info "Restoring database..."

    local db_source="${extract_dir}/database"

    if [ -d "$db_source" ]; then
        # Stop application if running
        if [ -f "${PROJECT_ROOT}/app.pid" ]; then
            local pid
            pid=$(cat "${PROJECT_ROOT}/app.pid")
            if kill -0 "$pid" 2>/dev/null; then
                log_warning "Stopping running application..."
                kill "$pid" 2>/dev/null || true
                sleep 2
            fi
        fi

        # Restore main database
        if [ -f "${db_source}/learning_captures.db" ]; then
            cp "${db_source}/learning_captures.db" "${PROJECT_ROOT}/"
            log_success "Main database restored"
        fi

        # Restore WAL files if present
        if [ -f "${db_source}/learning_captures.db-wal" ]; then
            cp "${db_source}/learning_captures.db-wal" "${PROJECT_ROOT}/"
        fi
        if [ -f "${db_source}/learning_captures.db-shm" ]; then
            cp "${db_source}/learning_captures.db-shm" "${PROJECT_ROOT}/"
        fi

        # Restore feedback database
        if [ -f "${db_source}/feedback.db" ]; then
            cp "${db_source}/feedback.db" "${PROJECT_ROOT}/"
            log_success "Feedback database restored"
        fi

        # Restore goals database
        if [ -f "${db_source}/goals.db" ]; then
            cp "${db_source}/goals.db" "${PROJECT_ROOT}/"
            log_success "Goals database restored"
        fi
    else
        log_warning "No database found in backup"
    fi
}

# Restore ChromaDB
restore_chromadb() {
    local extract_dir="$1"

    log_info "Restoring ChromaDB..."

    local chroma_source="${extract_dir}/chromadb"
    local chroma_target="${PROJECT_ROOT}/chroma_data"

    if [ -d "$chroma_source" ]; then
        # Remove existing ChromaDB data
        if [ -d "$chroma_target" ]; then
            rm -rf "$chroma_target"
        fi

        mkdir -p "$chroma_target"
        cp -r "$chroma_source"/* "$chroma_target/"
        log_success "ChromaDB data restored"
    else
        log_warning "No ChromaDB data found in backup"
    fi

    # Restore vector store if present
    local vector_source="${extract_dir}/vector_store"
    local vector_target="${PROJECT_ROOT}/vector_store"

    if [ -d "$vector_source" ]; then
        if [ -d "$vector_target" ]; then
            rm -rf "$vector_target"
        fi

        mkdir -p "$vector_target"
        cp -r "$vector_source"/* "$vector_target/"
        log_success "Vector store data restored"
    fi
}

# Restore uploads
restore_uploads() {
    local extract_dir="$1"

    log_info "Restoring uploads..."

    local uploads_source="${extract_dir}/uploads"
    local uploads_target="${PROJECT_ROOT}/uploads"

    if [ -d "$uploads_source" ]; then
        mkdir -p "$uploads_target"
        cp -r "$uploads_source"/* "$uploads_target/" 2>/dev/null || true
        log_success "Uploads restored"
    else
        log_warning "No uploads found in backup"
    fi

    # Restore static uploads
    local static_source="${extract_dir}/static_uploads"
    local static_target="${PROJECT_ROOT}/static/uploads"

    if [ -d "$static_source" ]; then
        mkdir -p "$static_target"
        cp -r "$static_source"/* "$static_target/" 2>/dev/null || true
        log_success "Static uploads restored"
    fi
}

# Restore configuration
restore_config() {
    local extract_dir="$1"

    log_info "Restoring configuration..."

    local config_source="${extract_dir}/config"

    if [ -d "$config_source" ]; then
        # Only restore non-sensitive config files
        local safe_configs=(
            "pyproject.toml"
            "requirements.txt"
            "railway.toml"
            "railway.json"
        )

        for config in "${safe_configs[@]}"; do
            if [ -f "${config_source}/${config}" ]; then
                cp "${config_source}/${config}" "${PROJECT_ROOT}/"
                log_info "Restored: $config"
            fi
        done

        # Warn about .env files
        if [ -f "${config_source}/.env" ]; then
            log_warning ".env file found in backup but NOT restored for security"
            log_warning "Please manually review and restore: ${config_source}/.env"
        fi

        log_success "Configuration restored (excluding sensitive files)"
    else
        log_warning "No configuration found in backup"
    fi
}

# Main restore function
do_restore() {
    local backup_file="$1"

    log_info "Starting restore from: $backup_file"

    # Create temporary extraction directory
    local extract_dir
    extract_dir=$(mktemp -d)

    # Cleanup on exit
    cleanup_extract() {
        rm -rf "$extract_dir"
    }
    trap cleanup_extract EXIT

    # Extract archive
    log_info "Extracting backup..."

    if [[ "$backup_file" == *.tar.gz ]]; then
        tar -xzf "$backup_file" -C "$extract_dir"
    else
        tar -xf "$backup_file" -C "$extract_dir"
    fi

    # Find the temp_ directory inside
    local backup_content
    backup_content=$(find "$extract_dir" -maxdepth 1 -type d -name "temp_*" | head -1)

    if [ -z "$backup_content" ] || [ ! -d "$backup_content" ]; then
        log_error "Invalid backup structure"
        exit 4
    fi

    # Restore components
    create_pre_restore_backup
    restore_database "$backup_content"

    if [ "$DB_ONLY" = false ]; then
        restore_chromadb "$backup_content"
        restore_uploads "$backup_content"
        restore_config "$backup_content"
    fi

    log_success "Restore completed!"
}

# Main function
main() {
    # Handle list command
    if [ "$LIST_BACKUPS" = true ]; then
        list_backups
        exit 0
    fi

    # Check if backup file is specified
    if [ -z "$BACKUP_FILE" ]; then
        log_error "No backup file specified"
        echo ""
        echo "Usage: $0 --backup <filename>"
        echo "       $0 --list"
        exit 1
    fi

    # Resolve backup file path
    if [[ ! "$BACKUP_FILE" = /* ]]; then
        # Check in backup directory first
        if [ -f "${BACKUP_DIR}/${BACKUP_FILE}" ]; then
            BACKUP_FILE="${BACKUP_DIR}/${BACKUP_FILE}"
        elif [ -f "${PROJECT_ROOT}/${BACKUP_FILE}" ]; then
            BACKUP_FILE="${PROJECT_ROOT}/${BACKUP_FILE}"
        fi
    fi

    # Verify backup exists
    if [ ! -f "$BACKUP_FILE" ]; then
        log_error "Backup file not found: $BACKUP_FILE"
        exit 2
    fi

    log_info "=========================================="
    log_info "Restore Script"
    log_info "Backup file: $BACKUP_FILE"
    log_info "=========================================="

    # Verify backup
    if ! verify_backup "$BACKUP_FILE"; then
        log_error "Backup verification failed"
        exit 3
    fi

    # Exit if verify only
    if [ "$VERIFY_ONLY" = true ]; then
        log_success "Backup verification complete"
        exit 0
    fi

    # Confirm and restore
    confirm_restore
    do_restore "$BACKUP_FILE"

    log_info "=========================================="
    log_success "Restore completed successfully!"
    log_info "Please restart the application to apply changes"
    log_info "=========================================="
}

# Run main
main "$@"
