#!/bin/bash
#
# deploy.sh - Deployment script for Learning Voice Agent
#
# Usage: ./scripts/deploy.sh [--skip-frontend] [--dev]
#
# Options:
#   --skip-frontend    Skip frontend build
#   --dev              Install dev dependencies
#
# Exit codes:
#   0 - Success
#   1 - Prerequisites check failed
#   2 - Dependency installation failed
#   3 - Migration failed
#   4 - Frontend build failed
#   5 - Application start failed

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/deploy_$(date +%Y%m%d_%H%M%S).log"
PYTHON_MIN_VERSION="3.9"
PID_FILE="${PROJECT_ROOT}/app.pid"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
SKIP_FRONTEND=false
DEV_MODE=false

for arg in "$@"; do
    case $arg in
        --skip-frontend)
            SKIP_FRONTEND=true
            ;;
        --dev)
            DEV_MODE=true
            ;;
        --help|-h)
            echo "Usage: $0 [--skip-frontend] [--dev]"
            echo ""
            echo "Options:"
            echo "  --skip-frontend    Skip frontend build"
            echo "  --dev              Install dev dependencies"
            exit 0
            ;;
    esac
done

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Logging functions
log() {
    local level="$1"
    local message="$2"
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

log_info() { log "INFO" "$1"; echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { log "SUCCESS" "$1"; echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { log "WARNING" "$1"; echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { log "ERROR" "$1"; echo -e "${RED}[ERROR]${NC} $1" >&2; }

# Error handler
error_exit() {
    log_error "$1"
    exit "$2"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Version comparison
version_gte() {
    printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1 | grep -q "^$2$"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check Python
    if ! command_exists python3; then
        error_exit "Python 3 is not installed. Please install Python ${PYTHON_MIN_VERSION}+" 1
    fi

    local python_version
    python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')

    if ! version_gte "$python_version" "$PYTHON_MIN_VERSION"; then
        error_exit "Python version ${python_version} is too old. Minimum required: ${PYTHON_MIN_VERSION}" 1
    fi
    log_success "Python ${python_version} found"

    # Check pip
    if ! command_exists pip3 && ! command_exists pip; then
        error_exit "pip is not installed. Please install pip" 1
    fi

    local pip_cmd="pip3"
    command_exists pip3 || pip_cmd="pip"
    log_success "pip found: $($pip_cmd --version)"

    # Check Node.js and npm (optional, for frontend)
    if [ "$SKIP_FRONTEND" = false ]; then
        if command_exists node && command_exists npm; then
            log_success "Node.js $(node --version) and npm $(npm --version) found"
        else
            log_warning "Node.js/npm not found. Skipping frontend build."
            SKIP_FRONTEND=true
        fi
    fi

    # Check for required directories
    if [ ! -d "$PROJECT_ROOT" ]; then
        error_exit "Project root not found: $PROJECT_ROOT" 1
    fi

    # Check for requirements.txt
    if [ ! -f "${PROJECT_ROOT}/requirements.txt" ]; then
        error_exit "requirements.txt not found" 1
    fi

    log_success "Prerequisites check passed"
}

# Create virtual environment if it doesn't exist
setup_virtualenv() {
    log_info "Setting up virtual environment..."

    local venv_path="${PROJECT_ROOT}/venv"

    if [ ! -d "$venv_path" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv "$venv_path"
        log_success "Virtual environment created at $venv_path"
    else
        log_info "Virtual environment already exists"
    fi

    # Activate virtual environment
    source "${venv_path}/bin/activate"
    log_success "Virtual environment activated"
}

# Install/upgrade Python dependencies
install_dependencies() {
    log_info "Installing Python dependencies..."

    local pip_cmd="pip"
    local requirements_file="${PROJECT_ROOT}/requirements.txt"

    # Upgrade pip first
    $pip_cmd install --upgrade pip >> "$LOG_FILE" 2>&1 || {
        error_exit "Failed to upgrade pip" 2
    }

    # Install requirements
    $pip_cmd install -r "$requirements_file" >> "$LOG_FILE" 2>&1 || {
        error_exit "Failed to install requirements from $requirements_file" 2
    }

    # Install dev dependencies if requested
    if [ "$DEV_MODE" = true ]; then
        if [ -f "${PROJECT_ROOT}/requirements-dev.txt" ]; then
            log_info "Installing dev dependencies..."
            $pip_cmd install -r "${PROJECT_ROOT}/requirements-dev.txt" >> "$LOG_FILE" 2>&1 || {
                log_warning "Failed to install dev dependencies"
            }
        fi
    fi

    log_success "Python dependencies installed"
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."

    cd "$PROJECT_ROOT"

    # Check if migration script exists
    local migration_script="${SCRIPT_DIR}/migrate.sh"

    if [ -f "$migration_script" ] && [ -x "$migration_script" ]; then
        "$migration_script" --apply >> "$LOG_FILE" 2>&1 || {
            error_exit "Migration failed" 3
        }
        log_success "Migrations applied successfully"
    else
        # Run Python migration script if available
        if [ -f "${SCRIPT_DIR}/run_phase4_migration.py" ]; then
            python3 "${SCRIPT_DIR}/run_phase4_migration.py" >> "$LOG_FILE" 2>&1 || {
                log_warning "Phase 4 migration script failed (may already be applied)"
            }
        fi

        # Initialize database by running a quick test import
        python3 -c "from app.database import db; import asyncio; asyncio.run(db.initialize())" >> "$LOG_FILE" 2>&1 || {
            error_exit "Database initialization failed" 3
        }
        log_success "Database initialized"
    fi
}

# Build frontend
build_frontend() {
    if [ "$SKIP_FRONTEND" = true ]; then
        log_info "Skipping frontend build"
        return 0
    fi

    log_info "Building frontend..."

    local frontend_dir="${PROJECT_ROOT}/frontend"

    if [ ! -d "$frontend_dir" ]; then
        log_warning "Frontend directory not found, skipping build"
        return 0
    fi

    cd "$frontend_dir"

    # Install npm dependencies
    log_info "Installing npm dependencies..."
    npm install >> "$LOG_FILE" 2>&1 || {
        error_exit "npm install failed" 4
    }

    # Build for production
    log_info "Building frontend for production..."
    npm run build >> "$LOG_FILE" 2>&1 || {
        error_exit "Frontend build failed" 4
    }

    # Copy dist to static if needed
    if [ -d "${frontend_dir}/dist" ]; then
        log_info "Copying frontend build to static directory..."
        cp -r "${frontend_dir}/dist/"* "${PROJECT_ROOT}/static/" 2>/dev/null || true
    fi

    log_success "Frontend build completed"
    cd "$PROJECT_ROOT"
}

# Download NLP models if needed
download_models() {
    log_info "Checking NLP models..."

    if [ -f "${SCRIPT_DIR}/download_nlp_models.py" ]; then
        python3 "${SCRIPT_DIR}/download_nlp_models.py" >> "$LOG_FILE" 2>&1 || {
            log_warning "NLP model download encountered issues (non-critical)"
        }
        log_success "NLP models checked/downloaded"
    fi
}

# Stop existing application
stop_application() {
    log_info "Stopping existing application..."

    if [ -f "$PID_FILE" ]; then
        local pid
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" >> "$LOG_FILE" 2>&1 || true
            sleep 2
            # Force kill if still running
            kill -9 "$pid" 2>/dev/null || true
            log_success "Stopped existing application (PID: $pid)"
        fi
        rm -f "$PID_FILE"
    fi

    # Also try to kill any uvicorn processes on the default port
    pkill -f "uvicorn app.main:app" 2>/dev/null || true
}

# Start application
start_application() {
    log_info "Starting application..."

    cd "$PROJECT_ROOT"

    # Set environment variables
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

    # Check for .env file
    if [ -f "${PROJECT_ROOT}/.env" ]; then
        log_info "Loading environment from .env file"
        set -a
        source "${PROJECT_ROOT}/.env"
        set +a
    else
        log_warning "No .env file found. Using defaults or environment variables."
    fi

    # Start with uvicorn
    local host="${HOST:-0.0.0.0}"
    local port="${PORT:-8000}"
    local workers="${WORKERS:-1}"

    log_info "Starting uvicorn on ${host}:${port}..."

    nohup python3 -m uvicorn app.main:app \
        --host "$host" \
        --port "$port" \
        --workers "$workers" \
        >> "${LOG_DIR}/app.log" 2>&1 &

    local app_pid=$!
    echo "$app_pid" > "$PID_FILE"

    # Wait for application to start
    sleep 3

    # Check if application is running
    if kill -0 "$app_pid" 2>/dev/null; then
        log_success "Application started (PID: $app_pid)"
        log_info "Application logs: ${LOG_DIR}/app.log"

        # Verify health endpoint
        sleep 2
        if command_exists curl; then
            if curl -s "http://localhost:${port}/api/health" > /dev/null 2>&1; then
                log_success "Health check passed"
            else
                log_warning "Health check did not respond yet. Application may still be starting."
            fi
        fi
    else
        error_exit "Application failed to start. Check ${LOG_DIR}/app.log for details." 5
    fi
}

# Main deployment flow
main() {
    log_info "=========================================="
    log_info "Starting deployment at $(date)"
    log_info "Project root: $PROJECT_ROOT"
    log_info "=========================================="

    check_prerequisites
    setup_virtualenv
    install_dependencies
    run_migrations
    build_frontend
    download_models
    stop_application
    start_application

    log_info "=========================================="
    log_success "Deployment completed successfully!"
    log_info "Log file: $LOG_FILE"
    log_info "=========================================="
}

# Run main function
main "$@"
