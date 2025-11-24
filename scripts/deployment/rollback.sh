#!/bin/bash
#
# Rollback Script for Learning Voice Agent
# Enables quick rollback to previous versions in case of deployment issues.
#
# Usage:
#   ./scripts/deployment/rollback.sh status              # Check current state
#   ./scripts/deployment/rollback.sh list                # List available versions
#   ./scripts/deployment/rollback.sh to v1.9.0           # Rollback to specific version
#   ./scripts/deployment/rollback.sh latest              # Rollback to previous version
#   ./scripts/deployment/rollback.sh docker <tag>        # Rollback Docker deployment
#   ./scripts/deployment/rollback.sh railway             # Rollback Railway deployment
#

set -e

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Service configuration
SERVICE_NAME="${SERVICE_NAME:-learning-voice-agent}"
DOCKER_IMAGE="${DOCKER_IMAGE:-learning-voice-agent}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
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

log_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

show_help() {
    cat << EOF
Rollback Script for Learning Voice Agent

Usage: $0 <command> [options]

Commands:
    status          Show current deployment status
    list            List available versions/tags
    to <version>    Rollback to specific version (e.g., v1.9.0)
    latest          Rollback to previous version (HEAD~1)
    docker <tag>    Rollback Docker deployment to specific tag
    railway         Rollback Railway deployment
    systemd         Rollback systemd service
    verify          Verify rollback was successful

Options:
    -h, --help          Show this help message
    -y, --yes           Skip confirmation prompts
    --dry-run           Show what would be done without doing it
    --no-backup         Skip backup before rollback

Environment Variables:
    SERVICE_NAME        Name of systemd service (default: learning-voice-agent)
    DOCKER_IMAGE        Docker image name (default: learning-voice-agent)
    RAILWAY_PROJECT     Railway project ID

Examples:
    $0 status                           # Check current state
    $0 list                             # Show available versions
    $0 to v1.9.0                        # Rollback to v1.9.0
    $0 docker learning-voice-agent:1.9.0
    $0 railway

Rollback Procedures:
    1. Check current status
    2. Backup current state (optional)
    3. Stop current deployment
    4. Revert to target version
    5. Start service
    6. Verify deployment health

EOF
}

confirm_action() {
    local message="$1"
    if [[ "$SKIP_CONFIRM" == "true" ]]; then
        return 0
    fi

    echo -e "${YELLOW}$message${NC}"
    read -p "Continue? (yes/no): " response
    if [[ "$response" != "yes" ]]; then
        log_info "Cancelled"
        exit 0
    fi
}

# =============================================================================
# Status Functions
# =============================================================================

show_status() {
    echo ""
    echo "========================================"
    echo "  Deployment Status"
    echo "========================================"

    # Git status
    echo ""
    echo "--- Git Version ---"
    cd "$PROJECT_ROOT"
    local current_commit=$(git rev-parse --short HEAD 2>/dev/null || echo "N/A")
    local current_branch=$(git branch --show-current 2>/dev/null || echo "N/A")
    local current_tag=$(git describe --tags --exact-match 2>/dev/null || echo "no tag")
    local last_commit_date=$(git log -1 --format="%ci" 2>/dev/null || echo "N/A")

    echo "  Branch:  $current_branch"
    echo "  Commit:  $current_commit"
    echo "  Tag:     $current_tag"
    echo "  Date:    $last_commit_date"

    # Docker status (if applicable)
    if command -v docker &> /dev/null; then
        echo ""
        echo "--- Docker Status ---"
        local container_status=$(docker ps --filter "name=$SERVICE_NAME" --format "{{.Status}}" 2>/dev/null || echo "not running")
        local container_image=$(docker ps --filter "name=$SERVICE_NAME" --format "{{.Image}}" 2>/dev/null || echo "N/A")

        echo "  Container: $SERVICE_NAME"
        echo "  Status:    ${container_status:-not running}"
        echo "  Image:     ${container_image:-N/A}"
    fi

    # Systemd status (if applicable)
    if command -v systemctl &> /dev/null && systemctl list-unit-files | grep -q "$SERVICE_NAME"; then
        echo ""
        echo "--- Systemd Status ---"
        local service_status=$(systemctl is-active "$SERVICE_NAME" 2>/dev/null || echo "not found")
        echo "  Service: $SERVICE_NAME"
        echo "  Status:  $service_status"
    fi

    # Railway status (if CLI installed)
    if command -v railway &> /dev/null; then
        echo ""
        echo "--- Railway Status ---"
        if railway status &>/dev/null; then
            railway status 2>/dev/null | head -5 || echo "  Unable to fetch Railway status"
        else
            echo "  Not connected to Railway project"
        fi
    fi

    # Health check
    echo ""
    echo "--- Health Check ---"
    local health_url="${HEALTH_URL:-http://localhost:8000/health}"
    if curl -s -o /dev/null -w "%{http_code}" "$health_url" 2>/dev/null | grep -q "200"; then
        echo -e "  ${GREEN}[HEALTHY]${NC} Service responding at $health_url"
    else
        echo -e "  ${RED}[UNHEALTHY]${NC} Service not responding at $health_url"
    fi

    echo ""
    echo "========================================"
}

list_versions() {
    echo ""
    echo "========================================"
    echo "  Available Versions"
    echo "========================================"

    cd "$PROJECT_ROOT"

    # Git tags
    echo ""
    echo "--- Git Tags (last 10) ---"
    git tag --sort=-v:refname | head -10 | while read tag; do
        local date=$(git log -1 --format="%ci" "$tag" 2>/dev/null | cut -d' ' -f1)
        echo "  $tag ($date)"
    done

    # Recent commits
    echo ""
    echo "--- Recent Commits (last 10) ---"
    git log --oneline -10

    # Docker images (if applicable)
    if command -v docker &> /dev/null; then
        echo ""
        echo "--- Docker Images ---"
        docker images "$DOCKER_IMAGE" --format "{{.Tag}}\t{{.CreatedAt}}" 2>/dev/null | head -10 | \
            while read line; do echo "  $line"; done || echo "  No images found"
    fi

    echo ""
    echo "========================================"
}

# =============================================================================
# Rollback Functions
# =============================================================================

rollback_git() {
    local target="$1"

    log_step "Rolling back Git repository to $target..."

    cd "$PROJECT_ROOT"

    # Verify target exists
    if ! git rev-parse "$target" &>/dev/null; then
        log_error "Target version not found: $target"
        log_info "Use '$0 list' to see available versions"
        exit 1
    fi

    # Get current state for potential undo
    local current_commit=$(git rev-parse HEAD)
    log_info "Current commit: $current_commit"
    log_info "Target: $target"

    confirm_action "This will checkout $target. Uncommitted changes will be stashed."

    # Stash any changes
    if [[ -n $(git status --porcelain) ]]; then
        log_step "Stashing uncommitted changes..."
        git stash push -m "Auto-stash before rollback $(date +%Y%m%d_%H%M%S)"
    fi

    # Checkout target
    log_step "Checking out $target..."
    git checkout "$target"

    log_success "Git rollback complete"
    log_info "To undo: git checkout $current_commit"
}

rollback_docker() {
    local target_tag="$1"
    local full_image="$DOCKER_IMAGE:$target_tag"

    log_step "Rolling back Docker deployment to $full_image..."

    # Check if image exists
    if ! docker image inspect "$full_image" &>/dev/null; then
        log_error "Docker image not found: $full_image"
        log_info "Available images:"
        docker images "$DOCKER_IMAGE" --format "  {{.Repository}}:{{.Tag}}"
        exit 1
    fi

    confirm_action "This will stop the current container and start $full_image"

    # Stop current container
    if docker ps -q --filter "name=$SERVICE_NAME" | grep -q .; then
        log_step "Stopping current container..."
        docker stop "$SERVICE_NAME" || true
        docker rm "$SERVICE_NAME" || true
    fi

    # Start new container
    log_step "Starting container with $full_image..."
    docker run -d \
        --name "$SERVICE_NAME" \
        --restart unless-stopped \
        --env-file "${PROJECT_ROOT}/.env.production" \
        -p 8000:8000 \
        "$full_image"

    # Wait for startup
    log_step "Waiting for service to start..."
    sleep 5

    # Verify
    verify_deployment

    log_success "Docker rollback complete"
}

rollback_railway() {
    log_step "Rolling back Railway deployment..."

    if ! command -v railway &> /dev/null; then
        log_error "Railway CLI not installed"
        log_info "Install with: npm install -g @railway/cli"
        exit 1
    fi

    confirm_action "This will rollback the Railway deployment"

    # Railway rollback
    cd "$PROJECT_ROOT"
    log_step "Initiating Railway rollback..."
    railway rollback

    # Wait for deployment
    log_step "Waiting for deployment..."
    sleep 30

    log_success "Railway rollback initiated"
    log_info "Check deployment status at: https://railway.app"
}

rollback_systemd() {
    local target="$1"

    log_step "Rolling back systemd service..."

    confirm_action "This will restart the $SERVICE_NAME service"

    # Checkout target version
    if [[ -n "$target" ]]; then
        rollback_git "$target"
    fi

    # Restart service
    log_step "Restarting $SERVICE_NAME service..."
    sudo systemctl restart "$SERVICE_NAME"

    # Check status
    sleep 3
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        log_success "Service restarted successfully"
    else
        log_error "Service failed to start"
        log_info "Check logs: journalctl -u $SERVICE_NAME -n 50"
        exit 1
    fi

    verify_deployment
}

rollback_to_previous() {
    log_step "Rolling back to previous version..."

    cd "$PROJECT_ROOT"

    # Get previous commit
    local current=$(git rev-parse HEAD)
    local previous=$(git rev-parse HEAD~1)

    log_info "Current:  $current"
    log_info "Previous: $previous"

    rollback_git "HEAD~1"
}

# =============================================================================
# Verification
# =============================================================================

verify_deployment() {
    log_step "Verifying deployment..."

    local health_url="${HEALTH_URL:-http://localhost:8000/health}"
    local max_attempts=12
    local attempt=1

    while [[ $attempt -le $max_attempts ]]; do
        log_info "Health check attempt $attempt/$max_attempts..."

        if curl -s -o /dev/null -w "%{http_code}" "$health_url" 2>/dev/null | grep -q "200"; then
            log_success "Health check passed"
            return 0
        fi

        sleep 5
        ((attempt++))
    done

    log_error "Health check failed after $max_attempts attempts"
    log_warn "Service may not be healthy. Check logs."
    return 1
}

# =============================================================================
# Incident Response
# =============================================================================

incident_response() {
    echo ""
    echo "========================================"
    echo "  INCIDENT RESPONSE CHECKLIST"
    echo "========================================"
    echo ""
    echo "1. [ ] ASSESS: Check error logs and metrics"
    echo "       - View logs: journalctl -u $SERVICE_NAME -n 100"
    echo "       - Check monitoring dashboard"
    echo ""
    echo "2. [ ] COMMUNICATE: Notify stakeholders"
    echo "       - Update status page if available"
    echo "       - Notify team via Slack/email"
    echo ""
    echo "3. [ ] ROLLBACK: Execute rollback"
    echo "       - $0 to <version>"
    echo "       - Or: $0 docker <tag>"
    echo ""
    echo "4. [ ] VERIFY: Confirm rollback success"
    echo "       - $0 verify"
    echo "       - Check health endpoint"
    echo ""
    echo "5. [ ] DOCUMENT: Record incident details"
    echo "       - Time of incident"
    echo "       - Symptoms observed"
    echo "       - Actions taken"
    echo "       - Root cause (if known)"
    echo ""
    echo "6. [ ] POST-MORTEM: Schedule follow-up"
    echo "       - Identify root cause"
    echo "       - Document lessons learned"
    echo "       - Create action items"
    echo ""
    echo "========================================"
}

# =============================================================================
# Main
# =============================================================================

COMMAND="${1:-help}"
shift || true

# Parse global options
SKIP_CONFIRM=false
DRY_RUN=false
NO_BACKUP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -y|--yes)
            SKIP_CONFIRM=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-backup)
            NO_BACKUP=true
            shift
            ;;
        *)
            break
            ;;
    esac
done

case $COMMAND in
    status)
        show_status
        ;;
    list)
        list_versions
        ;;
    to)
        TARGET="${1:-}"
        if [[ -z "$TARGET" ]]; then
            log_error "Please specify target version"
            log_info "Usage: $0 to <version>"
            exit 1
        fi
        rollback_git "$TARGET"
        ;;
    latest|previous)
        rollback_to_previous
        ;;
    docker)
        TAG="${1:-}"
        if [[ -z "$TAG" ]]; then
            log_error "Please specify Docker tag"
            log_info "Usage: $0 docker <tag>"
            exit 1
        fi
        rollback_docker "$TAG"
        ;;
    railway)
        rollback_railway
        ;;
    systemd)
        TARGET="${1:-}"
        rollback_systemd "$TARGET"
        ;;
    verify)
        verify_deployment
        ;;
    incident)
        incident_response
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
