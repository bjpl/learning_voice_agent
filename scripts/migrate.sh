#!/bin/bash
#
# migrate.sh - Database migration script for Learning Voice Agent
#
# Usage: ./scripts/migrate.sh [--status] [--apply] [--rollback VERSION]
#
# Options:
#   --status          Show current migration status
#   --apply           Apply all pending migrations
#   --rollback VER    Rollback to specific version
#   --create NAME     Create new migration file
#   --dry-run         Show what would be done without executing
#
# Exit codes:
#   0 - Success
#   1 - Configuration error
#   2 - Migration failed
#   3 - Rollback failed

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MIGRATIONS_DIR="${SCRIPT_DIR}/migrations"
DB_FILE="${PROJECT_ROOT}/learning_captures.db"
MIGRATION_TABLE="schema_migrations"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Parse arguments
ACTION=""
ROLLBACK_VERSION=""
MIGRATION_NAME=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --status|-s)
            ACTION="status"
            shift
            ;;
        --apply|-a)
            ACTION="apply"
            shift
            ;;
        --rollback|-r)
            ACTION="rollback"
            ROLLBACK_VERSION="$2"
            shift 2
            ;;
        --create|-c)
            ACTION="create"
            MIGRATION_NAME="$2"
            shift 2
            ;;
        --dry-run|-d)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--status] [--apply] [--rollback VERSION] [--create NAME] [--dry-run]"
            echo ""
            echo "Options:"
            echo "  --status, -s       Show current migration status"
            echo "  --apply, -a        Apply all pending migrations"
            echo "  --rollback, -r     Rollback to specific version"
            echo "  --create, -c       Create new migration file"
            echo "  --dry-run, -d      Show what would be done"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Default action is status
[ -z "$ACTION" ] && ACTION="status"

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }

# Check prerequisites
check_prerequisites() {
    if ! command -v sqlite3 &> /dev/null; then
        log_error "sqlite3 is required but not installed"
        exit 1
    fi

    # Create migrations directory if it doesn't exist
    mkdir -p "$MIGRATIONS_DIR"
}

# Initialize migration tracking table
init_migration_table() {
    sqlite3 "$DB_FILE" << EOF
CREATE TABLE IF NOT EXISTS $MIGRATION_TABLE (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    applied_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    checksum TEXT,
    execution_time_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_migration_version ON $MIGRATION_TABLE(version);
EOF
}

# Get current schema version
get_current_version() {
    if [ ! -f "$DB_FILE" ]; then
        echo "0"
        return
    fi

    # Check if migration table exists
    local table_exists
    table_exists=$(sqlite3 "$DB_FILE" "SELECT name FROM sqlite_master WHERE type='table' AND name='$MIGRATION_TABLE';" 2>/dev/null || echo "")

    if [ -z "$table_exists" ]; then
        echo "0"
        return
    fi

    local version
    version=$(sqlite3 "$DB_FILE" "SELECT MAX(version) FROM $MIGRATION_TABLE;" 2>/dev/null || echo "0")

    if [ -z "$version" ] || [ "$version" = "NULL" ]; then
        echo "0"
    else
        echo "$version"
    fi
}

# Get list of applied migrations
get_applied_migrations() {
    if [ ! -f "$DB_FILE" ]; then
        return
    fi

    local table_exists
    table_exists=$(sqlite3 "$DB_FILE" "SELECT name FROM sqlite_master WHERE type='table' AND name='$MIGRATION_TABLE';" 2>/dev/null || echo "")

    if [ -z "$table_exists" ]; then
        return
    fi

    sqlite3 "$DB_FILE" "SELECT version FROM $MIGRATION_TABLE ORDER BY version;" 2>/dev/null || true
}

# Get list of migration files
get_migration_files() {
    if [ -d "$MIGRATIONS_DIR" ]; then
        find "$MIGRATIONS_DIR" -name "*.sql" -type f | sort
    fi
}

# Extract version from migration filename
get_migration_version() {
    local filename="$1"
    basename "$filename" | sed 's/^V\([0-9]*\)__.*/\1/' | sed 's/^0*//'
}

# Extract name from migration filename
get_migration_name() {
    local filename="$1"
    basename "$filename" | sed 's/^V[0-9]*__\(.*\)\.sql/\1/' | tr '_' ' '
}

# Calculate file checksum
get_file_checksum() {
    local file="$1"
    if command -v sha256sum &> /dev/null; then
        sha256sum "$file" | awk '{print $1}'
    elif command -v shasum &> /dev/null; then
        shasum -a 256 "$file" | awk '{print $1}'
    else
        echo "no-checksum"
    fi
}

# Show migration status
show_status() {
    echo -e "${CYAN}=========================================="
    echo "Migration Status"
    echo -e "==========================================${NC}"
    echo ""

    local current_version
    current_version=$(get_current_version)
    echo "Database: $DB_FILE"
    echo "Current Version: $current_version"
    echo ""

    # Get applied migrations
    local applied
    applied=$(get_applied_migrations)

    echo "Applied Migrations:"
    if [ -n "$applied" ]; then
        while IFS= read -r version; do
            local name applied_at
            name=$(sqlite3 "$DB_FILE" "SELECT name FROM $MIGRATION_TABLE WHERE version='$version';")
            applied_at=$(sqlite3 "$DB_FILE" "SELECT applied_at FROM $MIGRATION_TABLE WHERE version='$version';")
            echo -e "  ${GREEN}[APPLIED]${NC} V${version} - $name ($applied_at)"
        done <<< "$applied"
    else
        echo "  (none)"
    fi
    echo ""

    # Check for pending migrations
    echo "Pending Migrations:"
    local has_pending=false

    while IFS= read -r migration_file; do
        [ -z "$migration_file" ] && continue

        local version
        version=$(get_migration_version "$migration_file")

        # Check if already applied
        if ! echo "$applied" | grep -q "^${version}$"; then
            local name
            name=$(get_migration_name "$migration_file")
            echo -e "  ${YELLOW}[PENDING]${NC} V${version} - $name"
            has_pending=true
        fi
    done < <(get_migration_files)

    if [ "$has_pending" = false ]; then
        echo "  (none)"
    fi

    echo ""

    # Show table info if database exists
    if [ -f "$DB_FILE" ]; then
        echo "Database Tables:"
        sqlite3 "$DB_FILE" "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;" 2>/dev/null | while read -r table; do
            local count
            count=$(sqlite3 "$DB_FILE" "SELECT COUNT(*) FROM \"$table\";" 2>/dev/null || echo "?")
            echo "  - $table ($count rows)"
        done
    fi
}

# Apply a single migration
apply_migration() {
    local migration_file="$1"
    local version
    local name
    local checksum

    version=$(get_migration_version "$migration_file")
    name=$(get_migration_name "$migration_file")
    checksum=$(get_file_checksum "$migration_file")

    log_info "Applying migration V${version}: $name"

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would apply: $migration_file"
        cat "$migration_file"
        return 0
    fi

    # Create backup before migration
    local backup_file="${DB_FILE}.backup_v${version}"
    if [ -f "$DB_FILE" ]; then
        cp "$DB_FILE" "$backup_file"
        log_info "Created backup: $backup_file"
    fi

    # Execute migration
    local start_time
    local end_time
    local execution_time

    start_time=$(date +%s%N)

    if ! sqlite3 "$DB_FILE" < "$migration_file"; then
        log_error "Migration failed!"

        # Restore from backup
        if [ -f "$backup_file" ]; then
            cp "$backup_file" "$DB_FILE"
            log_info "Restored from backup"
        fi

        return 1
    fi

    end_time=$(date +%s%N)
    execution_time=$(( (end_time - start_time) / 1000000 ))

    # Record migration
    sqlite3 "$DB_FILE" << EOF
INSERT INTO $MIGRATION_TABLE (version, name, checksum, execution_time_ms)
VALUES ('$version', '$name', '$checksum', $execution_time);
EOF

    # Remove backup on success
    rm -f "$backup_file"

    log_success "Migration V${version} applied (${execution_time}ms)"
}

# Apply all pending migrations
apply_migrations() {
    log_info "Checking for pending migrations..."

    # Initialize migration table
    init_migration_table

    local applied
    applied=$(get_applied_migrations)

    local migrations_applied=0

    while IFS= read -r migration_file; do
        [ -z "$migration_file" ] && continue

        local version
        version=$(get_migration_version "$migration_file")

        # Check if already applied
        if ! echo "$applied" | grep -q "^${version}$"; then
            if ! apply_migration "$migration_file"; then
                log_error "Migration failed, stopping."
                exit 2
            fi
            migrations_applied=$((migrations_applied + 1))
        fi
    done < <(get_migration_files)

    if [ "$migrations_applied" -eq 0 ]; then
        log_info "No pending migrations"
    else
        log_success "Applied $migrations_applied migration(s)"
    fi
}

# Rollback to specific version
do_rollback() {
    local target_version="$1"

    log_info "Rolling back to version $target_version..."

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would rollback to version $target_version"
        return 0
    fi

    local current_version
    current_version=$(get_current_version)

    if [ "$target_version" -ge "$current_version" ]; then
        log_warning "Target version ($target_version) is not less than current version ($current_version)"
        return 0
    fi

    # Create backup before rollback
    local backup_file="${DB_FILE}.backup_rollback_$(date +%Y%m%d_%H%M%S)"
    cp "$DB_FILE" "$backup_file"
    log_info "Created backup: $backup_file"

    # Find rollback files
    local rollback_applied=0

    # Get migrations to rollback (in reverse order)
    while IFS= read -r version; do
        [ -z "$version" ] && continue

        if [ "$version" -gt "$target_version" ]; then
            local rollback_file="${MIGRATIONS_DIR}/V${version}__rollback.sql"

            if [ -f "$rollback_file" ]; then
                log_info "Applying rollback for V${version}..."

                if ! sqlite3 "$DB_FILE" < "$rollback_file"; then
                    log_error "Rollback failed!"
                    cp "$backup_file" "$DB_FILE"
                    log_info "Restored from backup"
                    exit 3
                fi
            else
                log_warning "No rollback file for V${version}, using schema deletion"
            fi

            # Remove from migration table
            sqlite3 "$DB_FILE" "DELETE FROM $MIGRATION_TABLE WHERE version = '$version';"

            log_success "Rolled back V${version}"
            rollback_applied=$((rollback_applied + 1))
        fi
    done < <(get_applied_migrations | sort -rn)

    if [ "$rollback_applied" -eq 0 ]; then
        log_info "No migrations to rollback"
    else
        log_success "Rolled back $rollback_applied migration(s)"
    fi
}

# Create new migration file
create_migration() {
    local name="$1"

    if [ -z "$name" ]; then
        log_error "Migration name is required"
        exit 1
    fi

    # Get next version number
    local last_version=0
    while IFS= read -r migration_file; do
        [ -z "$migration_file" ] && continue
        local version
        version=$(get_migration_version "$migration_file")
        if [ "$version" -gt "$last_version" ]; then
            last_version="$version"
        fi
    done < <(get_migration_files)

    local new_version=$((last_version + 1))
    local formatted_version
    formatted_version=$(printf "%03d" "$new_version")

    # Sanitize name
    local safe_name
    safe_name=$(echo "$name" | tr ' ' '_' | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_]//g')

    local migration_file="${MIGRATIONS_DIR}/V${formatted_version}__${safe_name}.sql"
    local rollback_file="${MIGRATIONS_DIR}/V${formatted_version}__rollback.sql"

    # Create migration file
    cat > "$migration_file" << EOF
-- Migration: V${formatted_version} - $name
-- Created: $(date -Iseconds)
--
-- Description:
-- TODO: Add description of changes
--

-- Your migration SQL here
-- Example:
-- CREATE TABLE new_table (
--     id INTEGER PRIMARY KEY,
--     name TEXT NOT NULL
-- );

EOF

    # Create rollback file
    cat > "$rollback_file" << EOF
-- Rollback: V${formatted_version} - $name
--
-- This file should reverse the changes made in V${formatted_version}__${safe_name}.sql
--

-- Your rollback SQL here
-- Example:
-- DROP TABLE IF EXISTS new_table;

EOF

    log_success "Created migration files:"
    echo "  Migration: $migration_file"
    echo "  Rollback:  $rollback_file"
}

# Main function
main() {
    check_prerequisites

    case "$ACTION" in
        status)
            show_status
            ;;
        apply)
            apply_migrations
            ;;
        rollback)
            if [ -z "$ROLLBACK_VERSION" ]; then
                log_error "Rollback version is required"
                exit 1
            fi
            do_rollback "$ROLLBACK_VERSION"
            ;;
        create)
            create_migration "$MIGRATION_NAME"
            ;;
        *)
            log_error "Unknown action: $ACTION"
            exit 1
            ;;
    esac
}

# Run main
main "$@"
