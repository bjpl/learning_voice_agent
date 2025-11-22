#!/bin/bash
#
# backup.sh - Backup script for Learning Voice Agent
#
# Usage: ./scripts/backup.sh [--keep N] [--output DIR]
#
# Options:
#   --keep N       Keep last N backups (default: 7)
#   --output DIR   Backup output directory (default: ./backups)
#   --db-only      Only backup database
#   --no-compress  Don't compress the backup
#
# Exit codes:
#   0 - Success
#   1 - Configuration error
#   2 - Backup failed
#   3 - Cleanup failed

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEFAULT_BACKUP_DIR="${PROJECT_ROOT}/backups"
DEFAULT_KEEP_COUNT=7

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
BACKUP_DIR="$DEFAULT_BACKUP_DIR"
KEEP_COUNT=$DEFAULT_KEEP_COUNT
DB_ONLY=false
COMPRESS=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --keep)
            KEEP_COUNT="$2"
            shift 2
            ;;
        --output)
            BACKUP_DIR="$2"
            shift 2
            ;;
        --db-only)
            DB_ONLY=true
            shift
            ;;
        --no-compress)
            COMPRESS=false
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--keep N] [--output DIR] [--db-only] [--no-compress]"
            echo ""
            echo "Options:"
            echo "  --keep N       Keep last N backups (default: 7)"
            echo "  --output DIR   Backup output directory (default: ./backups)"
            echo "  --db-only      Only backup database"
            echo "  --no-compress  Don't compress the backup"
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

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Temp directory for this backup
BACKUP_TEMP="${BACKUP_DIR}/temp_${TIMESTAMP}"
mkdir -p "$BACKUP_TEMP"

# Cleanup function
cleanup() {
    if [ -d "$BACKUP_TEMP" ]; then
        rm -rf "$BACKUP_TEMP"
    fi
}
trap cleanup EXIT

# Backup SQLite database
backup_sqlite() {
    log_info "Backing up SQLite database..."

    local db_file="${PROJECT_ROOT}/learning_captures.db"
    local backup_target="${BACKUP_TEMP}/database"
    mkdir -p "$backup_target"

    if [ -f "$db_file" ]; then
        # Use sqlite3 backup command if available for safe backup
        if command -v sqlite3 &> /dev/null; then
            sqlite3 "$db_file" ".backup '${backup_target}/learning_captures.db'" 2>/dev/null || {
                # Fallback to cp if sqlite3 backup fails
                cp "$db_file" "${backup_target}/learning_captures.db"
            }
        else
            cp "$db_file" "${backup_target}/learning_captures.db"
        fi
        log_success "SQLite database backed up"

        # Get database stats
        local db_size
        db_size=$(du -h "$db_file" | cut -f1)
        log_info "Database size: $db_size"
    else
        log_warning "SQLite database not found at $db_file"
    fi

    # Also backup any .db-wal and .db-shm files
    if [ -f "${db_file}-wal" ]; then
        cp "${db_file}-wal" "${backup_target}/"
    fi
    if [ -f "${db_file}-shm" ]; then
        cp "${db_file}-shm" "${backup_target}/"
    fi

    # Backup feedback database if exists
    local feedback_db="${PROJECT_ROOT}/feedback.db"
    if [ -f "$feedback_db" ]; then
        if command -v sqlite3 &> /dev/null; then
            sqlite3 "$feedback_db" ".backup '${backup_target}/feedback.db'" 2>/dev/null || {
                cp "$feedback_db" "${backup_target}/feedback.db"
            }
        else
            cp "$feedback_db" "${backup_target}/feedback.db"
        fi
        log_success "Feedback database backed up"
    fi

    # Backup goals database if exists
    local goals_db="${PROJECT_ROOT}/goals.db"
    if [ -f "$goals_db" ]; then
        if command -v sqlite3 &> /dev/null; then
            sqlite3 "$goals_db" ".backup '${backup_target}/goals.db'" 2>/dev/null || {
                cp "$goals_db" "${backup_target}/goals.db"
            }
        else
            cp "$goals_db" "${backup_target}/goals.db"
        fi
        log_success "Goals database backed up"
    fi
}

# Backup ChromaDB data
backup_chromadb() {
    log_info "Backing up ChromaDB data..."

    local chroma_dir="${PROJECT_ROOT}/chroma_data"
    local backup_target="${BACKUP_TEMP}/chromadb"

    if [ -d "$chroma_dir" ]; then
        mkdir -p "$backup_target"
        cp -r "$chroma_dir"/* "$backup_target/"
        log_success "ChromaDB data backed up"

        # Get directory size
        local chroma_size
        chroma_size=$(du -sh "$chroma_dir" | cut -f1)
        log_info "ChromaDB size: $chroma_size"
    else
        log_warning "ChromaDB directory not found at $chroma_dir"
    fi

    # Also check for vector store directory
    local vector_dir="${PROJECT_ROOT}/vector_store"
    if [ -d "$vector_dir" ]; then
        mkdir -p "${BACKUP_TEMP}/vector_store"
        cp -r "$vector_dir"/* "${BACKUP_TEMP}/vector_store/"
        log_success "Vector store data backed up"
    fi
}

# Backup uploads directory
backup_uploads() {
    log_info "Backing up uploads directory..."

    local uploads_dir="${PROJECT_ROOT}/uploads"
    local backup_target="${BACKUP_TEMP}/uploads"

    if [ -d "$uploads_dir" ]; then
        mkdir -p "$backup_target"

        # Count files
        local file_count
        file_count=$(find "$uploads_dir" -type f | wc -l)

        if [ "$file_count" -gt 0 ]; then
            cp -r "$uploads_dir"/* "$backup_target/"
            log_success "Uploads backed up ($file_count files)"

            # Get directory size
            local uploads_size
            uploads_size=$(du -sh "$uploads_dir" | cut -f1)
            log_info "Uploads size: $uploads_size"
        else
            log_info "No files in uploads directory"
        fi
    else
        log_warning "Uploads directory not found at $uploads_dir"
    fi

    # Also backup static/uploads if exists
    local static_uploads="${PROJECT_ROOT}/static/uploads"
    if [ -d "$static_uploads" ]; then
        mkdir -p "${BACKUP_TEMP}/static_uploads"
        cp -r "$static_uploads"/* "${BACKUP_TEMP}/static_uploads/" 2>/dev/null || true
        log_success "Static uploads backed up"
    fi
}

# Backup configuration files
backup_config() {
    log_info "Backing up configuration files..."

    local config_target="${BACKUP_TEMP}/config"
    mkdir -p "$config_target"

    # List of config files to backup
    local config_files=(
        ".env"
        ".env.local"
        "pyproject.toml"
        "requirements.txt"
        "railway.toml"
        "railway.json"
    )

    local backed_up=0
    for config_file in "${config_files[@]}"; do
        if [ -f "${PROJECT_ROOT}/${config_file}" ]; then
            cp "${PROJECT_ROOT}/${config_file}" "$config_target/"
            backed_up=$((backed_up + 1))
        fi
    done

    log_success "Configuration files backed up ($backed_up files)"
}

# Create backup manifest
create_manifest() {
    log_info "Creating backup manifest..."

    local manifest_file="${BACKUP_TEMP}/manifest.json"

    cat > "$manifest_file" << EOF
{
    "backup_timestamp": "$TIMESTAMP",
    "backup_date": "$(date -Iseconds)",
    "project": "learning_voice_agent",
    "version": "1.0.0",
    "contents": {
        "database": $([ -d "${BACKUP_TEMP}/database" ] && echo "true" || echo "false"),
        "chromadb": $([ -d "${BACKUP_TEMP}/chromadb" ] && echo "true" || echo "false"),
        "vector_store": $([ -d "${BACKUP_TEMP}/vector_store" ] && echo "true" || echo "false"),
        "uploads": $([ -d "${BACKUP_TEMP}/uploads" ] && echo "true" || echo "false"),
        "static_uploads": $([ -d "${BACKUP_TEMP}/static_uploads" ] && echo "true" || echo "false"),
        "config": $([ -d "${BACKUP_TEMP}/config" ] && echo "true" || echo "false")
    },
    "compressed": $COMPRESS,
    "hostname": "$(hostname)",
    "user": "$(whoami)"
}
EOF

    log_success "Manifest created"
}

# Create final archive
create_archive() {
    local archive_name="backup_${TIMESTAMP}"
    local final_archive

    if [ "$COMPRESS" = true ]; then
        log_info "Creating compressed archive..."
        final_archive="${BACKUP_DIR}/${archive_name}.tar.gz"
        tar -czf "$final_archive" -C "$BACKUP_DIR" "temp_${TIMESTAMP}"
    else
        log_info "Creating archive (uncompressed)..."
        final_archive="${BACKUP_DIR}/${archive_name}.tar"
        tar -cf "$final_archive" -C "$BACKUP_DIR" "temp_${TIMESTAMP}"
    fi

    # Get archive size
    local archive_size
    archive_size=$(du -h "$final_archive" | cut -f1)

    log_success "Archive created: $final_archive ($archive_size)"
    echo "$final_archive"
}

# Cleanup old backups
cleanup_old_backups() {
    log_info "Cleaning up old backups (keeping last $KEEP_COUNT)..."

    # Find and sort backup files
    local backup_pattern="${BACKUP_DIR}/backup_*.tar*"
    local backup_count
    backup_count=$(ls -1 $backup_pattern 2>/dev/null | wc -l)

    if [ "$backup_count" -gt "$KEEP_COUNT" ]; then
        local to_delete=$((backup_count - KEEP_COUNT))

        # Delete oldest backups
        ls -1t $backup_pattern 2>/dev/null | tail -n "$to_delete" | while read -r old_backup; do
            rm -f "$old_backup"
            log_info "Deleted old backup: $(basename "$old_backup")"
        done

        log_success "Cleaned up $to_delete old backup(s)"
    else
        log_info "No cleanup needed ($backup_count backups exist)"
    fi
}

# Calculate backup checksum
calculate_checksum() {
    local archive_file="$1"

    if command -v sha256sum &> /dev/null; then
        sha256sum "$archive_file" > "${archive_file}.sha256"
        log_success "Checksum created: ${archive_file}.sha256"
    elif command -v shasum &> /dev/null; then
        shasum -a 256 "$archive_file" > "${archive_file}.sha256"
        log_success "Checksum created: ${archive_file}.sha256"
    else
        log_warning "No checksum tool available"
    fi
}

# Main function
main() {
    log_info "=========================================="
    log_info "Starting backup at $(date)"
    log_info "Backup directory: $BACKUP_DIR"
    log_info "Keep count: $KEEP_COUNT"
    log_info "=========================================="

    # Always backup database
    backup_sqlite

    if [ "$DB_ONLY" = false ]; then
        backup_chromadb
        backup_uploads
        backup_config
    fi

    create_manifest

    # Create archive
    local archive_file
    archive_file=$(create_archive)

    # Calculate checksum
    calculate_checksum "$archive_file"

    # Cleanup old backups
    cleanup_old_backups

    log_info "=========================================="
    log_success "Backup completed successfully!"
    log_info "Archive: $archive_file"
    log_info "=========================================="
}

# Run main
main "$@"
