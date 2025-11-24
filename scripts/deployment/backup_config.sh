#!/bin/bash
#
# Backup Configuration Script for Learning Voice Agent
# Configures and manages backups for database, files, and configurations.
#
# Usage:
#   ./scripts/deployment/backup_config.sh setup      # Initial setup
#   ./scripts/deployment/backup_config.sh backup     # Run backup now
#   ./scripts/deployment/backup_config.sh restore    # Restore from backup
#   ./scripts/deployment/backup_config.sh list       # List available backups
#   ./scripts/deployment/backup_config.sh cleanup    # Remove old backups
#

set -e

# =============================================================================
# Configuration
# =============================================================================

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BACKUP_ROOT="${BACKUP_DIR:-$PROJECT_ROOT/backups}"
CONFIG_BACKUP_DIR="$BACKUP_ROOT/config"
DB_BACKUP_DIR="$BACKUP_ROOT/database"
DATA_BACKUP_DIR="$BACKUP_ROOT/data"

# Retention
DAILY_RETENTION=30      # Keep daily backups for 30 days
WEEKLY_RETENTION=12     # Keep weekly backups for 12 weeks
MONTHLY_RETENTION=12    # Keep monthly backups for 12 months

# Database (from environment or defaults)
DB_HOST="${DATABASE_HOST:-localhost}"
DB_PORT="${DATABASE_PORT:-5432}"
DB_NAME="${DATABASE_NAME:-learning_voice_agent}"
DB_USER="${DATABASE_USER:-postgres}"

# Timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATE_ONLY=$(date +%Y%m%d)
DAY_OF_WEEK=$(date +%u)  # 1=Monday, 7=Sunday
DAY_OF_MONTH=$(date +%d)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    cat << EOF
Backup Configuration Script for Learning Voice Agent

Usage: $0 <command> [options]

Commands:
    setup       Initialize backup directories and cron jobs
    backup      Run backup now (all components)
    db          Backup database only
    config      Backup configuration only
    data        Backup data files only
    restore     Restore from backup
    list        List available backups
    cleanup     Remove old backups (apply retention policy)
    verify      Verify backup integrity

Options:
    -h, --help          Show this help message
    -v, --verbose       Verbose output
    --dry-run           Show what would be done without doing it
    --backup-id ID      Specific backup ID for restore

Environment Variables:
    BACKUP_DIR          Base backup directory (default: ./backups)
    DATABASE_HOST       Database host (default: localhost)
    DATABASE_PORT       Database port (default: 5432)
    DATABASE_NAME       Database name (default: learning_voice_agent)
    DATABASE_USER       Database user (default: postgres)
    DATABASE_PASSWORD   Database password (required for db backup)

Examples:
    $0 setup                    # Initialize backup system
    $0 backup                   # Full backup
    $0 db                       # Database backup only
    $0 restore --backup-id 20231123_120000
    $0 list                     # Show available backups
    $0 cleanup                  # Apply retention policy

EOF
}

# =============================================================================
# Setup
# =============================================================================

setup_backup_system() {
    log_info "Setting up backup system..."

    # Create directories
    mkdir -p "$CONFIG_BACKUP_DIR"
    mkdir -p "$DB_BACKUP_DIR"
    mkdir -p "$DATA_BACKUP_DIR"

    log_success "Created backup directories:"
    echo "  - $CONFIG_BACKUP_DIR"
    echo "  - $DB_BACKUP_DIR"
    echo "  - $DATA_BACKUP_DIR"

    # Create backup script for cron
    CRON_SCRIPT="$BACKUP_ROOT/run_backup.sh"
    cat > "$CRON_SCRIPT" << 'SCRIPT'
#!/bin/bash
# Auto-generated backup script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
./scripts/deployment/backup_config.sh backup >> "$SCRIPT_DIR/backup.log" 2>&1
./scripts/deployment/backup_config.sh cleanup >> "$SCRIPT_DIR/backup.log" 2>&1
SCRIPT
    chmod +x "$CRON_SCRIPT"

    # Show cron setup instructions
    echo ""
    log_info "To enable automated backups, add to crontab:"
    echo ""
    echo "# Daily backup at 2 AM"
    echo "0 2 * * * $CRON_SCRIPT"
    echo ""
    echo "# Or run: crontab -e"
    echo ""

    log_success "Backup system setup complete"
}

# =============================================================================
# Backup Functions
# =============================================================================

backup_database() {
    log_info "Backing up database..."

    if [[ -z "$DATABASE_PASSWORD" ]]; then
        log_warn "DATABASE_PASSWORD not set. Using .pgpass or prompting."
    fi

    local backup_file="$DB_BACKUP_DIR/db_${TIMESTAMP}.sql.gz"

    # Check if pg_dump is available
    if ! command -v pg_dump &> /dev/null; then
        log_error "pg_dump not found. Install PostgreSQL client tools."
        return 1
    fi

    # Perform backup
    PGPASSWORD="$DATABASE_PASSWORD" pg_dump \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        --no-owner \
        --no-acl \
        --clean \
        --if-exists \
        2>/dev/null | gzip > "$backup_file"

    if [[ $? -eq 0 && -s "$backup_file" ]]; then
        local size=$(du -h "$backup_file" | cut -f1)
        log_success "Database backup created: $backup_file ($size)"

        # Create weekly/monthly copies if applicable
        if [[ "$DAY_OF_WEEK" == "7" ]]; then
            cp "$backup_file" "$DB_BACKUP_DIR/weekly_${DATE_ONLY}.sql.gz"
            log_info "Created weekly backup copy"
        fi

        if [[ "$DAY_OF_MONTH" == "01" ]]; then
            cp "$backup_file" "$DB_BACKUP_DIR/monthly_$(date +%Y%m).sql.gz"
            log_info "Created monthly backup copy"
        fi
    else
        log_error "Database backup failed or empty"
        rm -f "$backup_file"
        return 1
    fi
}

backup_configuration() {
    log_info "Backing up configuration..."

    local backup_file="$CONFIG_BACKUP_DIR/config_${TIMESTAMP}.tar.gz"
    local temp_dir=$(mktemp -d)

    # Files to backup
    local files_to_backup=(
        ".env.production"
        ".env.staging"
        "pyproject.toml"
        "requirements.txt"
        "docker-compose.yml"
        "docker-compose.production.yml"
        "Dockerfile"
        "Dockerfile.optimized"
        ".github/workflows/"
        "config/"
    )

    # Copy existing files
    for file in "${files_to_backup[@]}"; do
        local src="$PROJECT_ROOT/$file"
        if [[ -e "$src" ]]; then
            local dest_dir="$temp_dir/$(dirname "$file")"
            mkdir -p "$dest_dir"
            cp -r "$src" "$dest_dir/" 2>/dev/null || true
        fi
    done

    # Create tarball
    tar -czf "$backup_file" -C "$temp_dir" . 2>/dev/null

    # Cleanup
    rm -rf "$temp_dir"

    if [[ -s "$backup_file" ]]; then
        local size=$(du -h "$backup_file" | cut -f1)
        log_success "Configuration backup created: $backup_file ($size)"
    else
        log_error "Configuration backup failed"
        rm -f "$backup_file"
        return 1
    fi
}

backup_data() {
    log_info "Backing up data files..."

    local backup_file="$DATA_BACKUP_DIR/data_${TIMESTAMP}.tar.gz"

    # Directories to backup
    local dirs_to_backup=(
        "data/"
        "uploads/"
        "static/"
    )

    local include_args=""
    for dir in "${dirs_to_backup[@]}"; do
        if [[ -d "$PROJECT_ROOT/$dir" ]]; then
            include_args="$include_args $dir"
        fi
    done

    if [[ -z "$include_args" ]]; then
        log_warn "No data directories found to backup"
        return 0
    fi

    # Create tarball
    cd "$PROJECT_ROOT"
    tar -czf "$backup_file" $include_args 2>/dev/null

    if [[ -s "$backup_file" ]]; then
        local size=$(du -h "$backup_file" | cut -f1)
        log_success "Data backup created: $backup_file ($size)"
    else
        log_warn "Data backup empty or failed"
        rm -f "$backup_file"
    fi
}

run_full_backup() {
    log_info "Starting full backup..."
    echo ""

    local start_time=$(date +%s)
    local errors=0

    backup_database || ((errors++))
    echo ""

    backup_configuration || ((errors++))
    echo ""

    backup_data || ((errors++))
    echo ""

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    echo "========================================"
    if [[ $errors -eq 0 ]]; then
        log_success "Full backup completed in ${duration}s"
    else
        log_warn "Backup completed with $errors error(s) in ${duration}s"
    fi
    echo "========================================"

    return $errors
}

# =============================================================================
# Restore Functions
# =============================================================================

restore_database() {
    local backup_file="$1"

    if [[ ! -f "$backup_file" ]]; then
        log_error "Backup file not found: $backup_file"
        return 1
    fi

    log_warn "This will restore the database from: $backup_file"
    log_warn "All current data will be replaced!"
    read -p "Are you sure? (yes/no): " confirm

    if [[ "$confirm" != "yes" ]]; then
        log_info "Restore cancelled"
        return 0
    fi

    log_info "Restoring database..."

    gunzip -c "$backup_file" | PGPASSWORD="$DATABASE_PASSWORD" psql \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d "$DB_NAME"

    if [[ $? -eq 0 ]]; then
        log_success "Database restored successfully"
    else
        log_error "Database restore failed"
        return 1
    fi
}

restore_configuration() {
    local backup_file="$1"

    if [[ ! -f "$backup_file" ]]; then
        log_error "Backup file not found: $backup_file"
        return 1
    fi

    log_warn "This will restore configuration from: $backup_file"
    log_warn "Existing config files may be overwritten!"
    read -p "Are you sure? (yes/no): " confirm

    if [[ "$confirm" != "yes" ]]; then
        log_info "Restore cancelled"
        return 0
    fi

    log_info "Restoring configuration..."

    tar -xzf "$backup_file" -C "$PROJECT_ROOT"

    if [[ $? -eq 0 ]]; then
        log_success "Configuration restored successfully"
    else
        log_error "Configuration restore failed"
        return 1
    fi
}

# =============================================================================
# List and Cleanup
# =============================================================================

list_backups() {
    echo ""
    echo "========================================"
    echo "  Available Backups"
    echo "========================================"

    echo ""
    echo "--- Database Backups ---"
    if ls "$DB_BACKUP_DIR"/*.sql.gz 1>/dev/null 2>&1; then
        ls -lh "$DB_BACKUP_DIR"/*.sql.gz | awk '{print "  " $NF " (" $5 ")"}'
    else
        echo "  No database backups found"
    fi

    echo ""
    echo "--- Configuration Backups ---"
    if ls "$CONFIG_BACKUP_DIR"/*.tar.gz 1>/dev/null 2>&1; then
        ls -lh "$CONFIG_BACKUP_DIR"/*.tar.gz | awk '{print "  " $NF " (" $5 ")"}'
    else
        echo "  No configuration backups found"
    fi

    echo ""
    echo "--- Data Backups ---"
    if ls "$DATA_BACKUP_DIR"/*.tar.gz 1>/dev/null 2>&1; then
        ls -lh "$DATA_BACKUP_DIR"/*.tar.gz | awk '{print "  " $NF " (" $5 ")"}'
    else
        echo "  No data backups found"
    fi

    echo ""
    echo "========================================"

    # Show total size
    local total_size=$(du -sh "$BACKUP_ROOT" 2>/dev/null | cut -f1)
    echo "Total backup size: $total_size"
    echo "========================================"
}

cleanup_old_backups() {
    log_info "Applying retention policy..."

    local deleted_count=0

    # Daily backups - keep for DAILY_RETENTION days
    log_info "Cleaning daily backups (retention: $DAILY_RETENTION days)..."
    find "$DB_BACKUP_DIR" -name "db_*.sql.gz" -mtime +$DAILY_RETENTION -delete 2>/dev/null
    find "$CONFIG_BACKUP_DIR" -name "config_*.tar.gz" -mtime +$DAILY_RETENTION -delete 2>/dev/null
    find "$DATA_BACKUP_DIR" -name "data_*.tar.gz" -mtime +$DAILY_RETENTION -delete 2>/dev/null

    # Weekly backups - keep for WEEKLY_RETENTION weeks
    log_info "Cleaning weekly backups (retention: $WEEKLY_RETENTION weeks)..."
    find "$DB_BACKUP_DIR" -name "weekly_*.sql.gz" -mtime +$((WEEKLY_RETENTION * 7)) -delete 2>/dev/null

    # Monthly backups - keep for MONTHLY_RETENTION months
    log_info "Cleaning monthly backups (retention: $MONTHLY_RETENTION months)..."
    find "$DB_BACKUP_DIR" -name "monthly_*.sql.gz" -mtime +$((MONTHLY_RETENTION * 30)) -delete 2>/dev/null

    log_success "Cleanup complete"
}

verify_backups() {
    log_info "Verifying backup integrity..."

    local errors=0

    # Check database backups
    echo ""
    echo "--- Database Backups ---"
    for f in "$DB_BACKUP_DIR"/*.sql.gz; do
        if [[ -f "$f" ]]; then
            if gzip -t "$f" 2>/dev/null; then
                echo -e "  ${GREEN}[OK]${NC} $(basename "$f")"
            else
                echo -e "  ${RED}[CORRUPT]${NC} $(basename "$f")"
                ((errors++))
            fi
        fi
    done

    # Check config backups
    echo ""
    echo "--- Configuration Backups ---"
    for f in "$CONFIG_BACKUP_DIR"/*.tar.gz; do
        if [[ -f "$f" ]]; then
            if tar -tzf "$f" &>/dev/null; then
                echo -e "  ${GREEN}[OK]${NC} $(basename "$f")"
            else
                echo -e "  ${RED}[CORRUPT]${NC} $(basename "$f")"
                ((errors++))
            fi
        fi
    done

    # Check data backups
    echo ""
    echo "--- Data Backups ---"
    for f in "$DATA_BACKUP_DIR"/*.tar.gz; do
        if [[ -f "$f" ]]; then
            if tar -tzf "$f" &>/dev/null; then
                echo -e "  ${GREEN}[OK]${NC} $(basename "$f")"
            else
                echo -e "  ${RED}[CORRUPT]${NC} $(basename "$f")"
                ((errors++))
            fi
        fi
    done

    echo ""
    if [[ $errors -eq 0 ]]; then
        log_success "All backups verified successfully"
    else
        log_error "$errors backup(s) failed verification"
    fi

    return $errors
}

# =============================================================================
# Main
# =============================================================================

COMMAND="${1:-help}"
shift || true

case $COMMAND in
    setup)
        setup_backup_system
        ;;
    backup)
        run_full_backup
        ;;
    db|database)
        backup_database
        ;;
    config|configuration)
        backup_configuration
        ;;
    data)
        backup_data
        ;;
    restore)
        BACKUP_ID=""
        while [[ $# -gt 0 ]]; do
            case $1 in
                --backup-id)
                    BACKUP_ID="$2"
                    shift 2
                    ;;
                *)
                    shift
                    ;;
            esac
        done

        if [[ -z "$BACKUP_ID" ]]; then
            log_error "Please specify --backup-id"
            exit 1
        fi

        # Find matching backup
        DB_BACKUP=$(find "$DB_BACKUP_DIR" -name "*${BACKUP_ID}*" -type f | head -1)
        CONFIG_BACKUP=$(find "$CONFIG_BACKUP_DIR" -name "*${BACKUP_ID}*" -type f | head -1)

        if [[ -n "$DB_BACKUP" ]]; then
            restore_database "$DB_BACKUP"
        fi

        if [[ -n "$CONFIG_BACKUP" ]]; then
            restore_configuration "$CONFIG_BACKUP"
        fi
        ;;
    list)
        list_backups
        ;;
    cleanup)
        cleanup_old_backups
        ;;
    verify)
        verify_backups
        ;;
    -h|--help|help)
        show_help
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac
