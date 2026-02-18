#!/bin/bash
#
# Local Container Testing Script for PSAP Agent
# 
# This script builds and runs all containers locally with Podman
#
# Credentials Setup:
# - Google API Key: Loaded from GOOGLE_API_KEY env var or psap-agent/.env
# - Grafana: Loaded from GRAFANA_* env vars or psap-mcp-server/.env
# - Langfuse: Prompted interactively during startup (http://localhost:3000)
#
# Usage:
#   ./test-local-containers.sh          # Full setup
#   ./test-local-containers.sh cleanup  # Remove all containers
#   ./test-local-containers.sh rebuild  # Rebuild images only
#   ./test-local-containers.sh restart  # Restart app containers (keep DB)
#   ./test-local-containers.sh logs     # View live logs
#
# IMPORTANT: After making code changes:
#   1. Run: ./test-local-containers.sh rebuild
#   2. Then: ./test-local-containers.sh restart
#   This ensures containers use the updated code!
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NETWORK_NAME="psap-network"

# ── Load .env file (infrastructure credentials) ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"
if [ -f "$ENV_FILE" ]; then
    # Export all variables defined in .env (bash handles # comments natively)
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
else
    echo -e "${RED}ERROR: .env file not found at $ENV_FILE${NC}"
    echo "  Create one from the template:  cp .env.example .env"
    echo "  Then fill in the required credentials."
    exit 1
fi

# Validate required infrastructure credentials from .env
if [ -z "$POSTGRES_PASSWORD" ]; then
    echo -e "${RED}ERROR: POSTGRES_PASSWORD is not set in .env${NC}"
    exit 1
fi

# Credentials (will be prompted if not set)
GOOGLE_API_KEY="${GOOGLE_API_KEY:-}"
GRAFANA_URL="${GRAFANA_URL:-}"
GRAFANA_API_TOKEN="${GRAFANA_API_TOKEN:-}"
GRAFANA_DATASOURCE_UID="${GRAFANA_DATASOURCE_UID:-}"

# S3 Configuration (for loading performance data from S3)
S3_BUCKET="${S3_BUCKET:-}"
S3_KEY="${S3_KEY:-main/rhaiis-dashboard/consolidated_dashboard.csv}"
S3_REGION="${S3_REGION:-us-east-1}"
S3_CACHE_TTL_SECONDS="${S3_CACHE_TTL_SECONDS:-300}"
AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-}"
AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-}"

# Functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

prompt_for_credentials() {
    log_info "Setting up credentials..."
    echo ""
    
    # Google API Key
    if [ -z "$GOOGLE_API_KEY" ]; then
        # Try to load from psap-agent/.env
        if [ -f psap-agent/.env ] && grep -q "GOOGLE_API_KEY" psap-agent/.env; then
            GOOGLE_API_KEY=$(grep "GOOGLE_API_KEY" psap-agent/.env | cut -d '=' -f2- | tr -d '"' | tr -d "'")
            log_success "Google API Key loaded from psap-agent/.env"
        else
            log_warning "Google API Key not found"
            read -p "Enter your GOOGLE_API_KEY (or press Enter to skip): " GOOGLE_API_KEY
        fi
        
        if [ -n "$GOOGLE_API_KEY" ]; then
            export GOOGLE_API_KEY
            log_success "Google API Key configured"
        else
            log_error "Google API Key is required for the agent to work"
            exit 1
        fi
    else
        log_success "Google API Key detected in environment"
    fi
    echo ""
    
    # Grafana Credentials
    if [ -z "$GRAFANA_URL" ] || [ -z "$GRAFANA_API_TOKEN" ] || [ -z "$GRAFANA_DATASOURCE_UID" ]; then
        # Try to load from psap-mcp-server/.env
        if [ -f psap-mcp-server/.env ]; then
            log_info "Loading Grafana credentials from psap-mcp-server/.env..."
            
            if [ -z "$GRAFANA_URL" ] && grep -q "GRAFANA_URL" psap-mcp-server/.env; then
                GRAFANA_URL=$(grep "GRAFANA_URL" psap-mcp-server/.env | grep -v '^#' | cut -d '=' -f2- | tr -d '"' | tr -d "'")
                log_success "Grafana URL loaded: $GRAFANA_URL"
            fi
            
            if [ -z "$GRAFANA_API_TOKEN" ] && grep -q "GRAFANA_API_TOKEN" psap-mcp-server/.env; then
                GRAFANA_API_TOKEN=$(grep "GRAFANA_API_TOKEN" psap-mcp-server/.env | grep -v '^#' | cut -d '=' -f2- | tr -d '"' | tr -d "'")
                log_success "Grafana API Token loaded"
            fi
            
            if [ -z "$GRAFANA_DATASOURCE_UID" ] && grep -q "GRAFANA_DATASOURCE_UID" psap-mcp-server/.env; then
                GRAFANA_DATASOURCE_UID=$(grep "GRAFANA_DATASOURCE_UID" psap-mcp-server/.env | grep -v '^#' | cut -d '=' -f2- | tr -d '"' | tr -d "'")
                log_success "Grafana Datasource UID loaded: $GRAFANA_DATASOURCE_UID"
            fi
        fi
        
        # Prompt for any missing Grafana credentials
        if [ -z "$GRAFANA_URL" ]; then
            log_warning "Grafana URL not found"
            read -p "Enter your GRAFANA_URL (e.g., https://grafana.example.com): " GRAFANA_URL
        fi
        
        if [ -z "$GRAFANA_API_TOKEN" ]; then
            log_warning "Grafana API Token not found"
            read -p "Enter your GRAFANA_API_TOKEN (glsa_...): " GRAFANA_API_TOKEN
        fi
        
        if [ -z "$GRAFANA_DATASOURCE_UID" ]; then
            log_warning "Grafana Datasource UID not found"
            read -p "Enter your GRAFANA_DATASOURCE_UID: " GRAFANA_DATASOURCE_UID
        fi
        
        # Export Grafana credentials
        if [ -n "$GRAFANA_URL" ] && [ -n "$GRAFANA_API_TOKEN" ] && [ -n "$GRAFANA_DATASOURCE_UID" ]; then
            export GRAFANA_URL
            export GRAFANA_API_TOKEN
            export GRAFANA_DATASOURCE_UID
            log_success "Grafana credentials configured"
        else
            log_warning "Grafana credentials incomplete - Grafana metrics will be unavailable"
        fi
    else
        log_success "Grafana credentials detected in environment"
    fi
    echo ""
    
    # Load S3 Configuration from psap-mcp-server/.env
    # Helper function to extract value and strip inline comments
    extract_env_value() {
        local var_name="$1"
        local file="$2"
        grep "^${var_name}=" "$file" | grep -v '^#' | cut -d '=' -f2- | sed 's/#.*//' | tr -d '"' | tr -d "'" | xargs
    }
    
    if [ -f psap-mcp-server/.env ]; then
        if [ -z "$S3_BUCKET" ] && grep -q "^S3_BUCKET" psap-mcp-server/.env; then
            S3_BUCKET=$(extract_env_value "S3_BUCKET" "psap-mcp-server/.env")
            if [ -n "$S3_BUCKET" ]; then
                log_success "S3 Bucket loaded: $S3_BUCKET"
            fi
        fi
        
        if grep -q "^S3_KEY" psap-mcp-server/.env; then
            S3_KEY=$(extract_env_value "S3_KEY" "psap-mcp-server/.env")
        fi
        
        if grep -q "^S3_REGION" psap-mcp-server/.env; then
            S3_REGION=$(extract_env_value "S3_REGION" "psap-mcp-server/.env")
        fi
        
        if grep -q "^S3_CACHE_TTL_SECONDS" psap-mcp-server/.env; then
            S3_CACHE_TTL_SECONDS=$(extract_env_value "S3_CACHE_TTL_SECONDS" "psap-mcp-server/.env")
        fi
        
        # Load AWS credentials
        if [ -z "$AWS_ACCESS_KEY_ID" ] && grep -q "^AWS_ACCESS_KEY_ID" psap-mcp-server/.env; then
            AWS_ACCESS_KEY_ID=$(extract_env_value "AWS_ACCESS_KEY_ID" "psap-mcp-server/.env")
            if [ -n "$AWS_ACCESS_KEY_ID" ]; then
                log_success "AWS Access Key ID loaded"
            fi
        fi
        
        if [ -z "$AWS_SECRET_ACCESS_KEY" ] && grep -q "^AWS_SECRET_ACCESS_KEY" psap-mcp-server/.env; then
            AWS_SECRET_ACCESS_KEY=$(extract_env_value "AWS_SECRET_ACCESS_KEY" "psap-mcp-server/.env")
            if [ -n "$AWS_SECRET_ACCESS_KEY" ]; then
                log_success "AWS Secret Access Key loaded"
            fi
        fi
    fi
    echo ""
    
    # Show summary
    log_info "📋 Credentials Summary:"
    echo "  ✓ Google API Key:          ${GOOGLE_API_KEY:0:20}..."
    if [ -n "$GRAFANA_URL" ]; then
        echo "  ✓ Grafana URL:             $GRAFANA_URL"
        echo "  ✓ Grafana API Token:       ${GRAFANA_API_TOKEN:0:20}..."
        echo "  ✓ Grafana Datasource UID:  $GRAFANA_DATASOURCE_UID"
    else
        echo "  ⚠ Grafana:                 Not configured"
    fi
    if [ -n "$S3_BUCKET" ]; then
        echo "  ✓ S3 Bucket:               $S3_BUCKET"
        echo "  ✓ S3 Key:                  $S3_KEY"
        echo "  ✓ S3 Cache TTL:            ${S3_CACHE_TTL_SECONDS}s"
        if [ -n "$AWS_ACCESS_KEY_ID" ]; then
            echo "  ✓ AWS Access Key:          ${AWS_ACCESS_KEY_ID:0:10}..."
        else
            echo "  ⚠ AWS Credentials:         Not set (using IAM role or anonymous)"
        fi
    else
        echo "  ⚠ S3:                      Not configured (using local files)"
    fi
    echo ""
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if podman is installed
    if ! command -v podman &> /dev/null; then
        log_error "Podman not found. Please install Podman first."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

cleanup_existing() {
    log_info "Cleaning up existing containers..."
    
    # List of all containers managed by this script
    ALL_CONTAINERS="streamlit-ui psap-agent psap-mcp-server langfuse-web langfuse-worker langfuse-minio langfuse-redis langfuse-clickhouse langfuse psap-postgres pgvector-agent"
    
    for container in $ALL_CONTAINERS; do
        if podman container exists $container 2>/dev/null; then
            podman stop $container 2>/dev/null || true
            podman rm -f $container 2>/dev/null || true
        fi
    done
    
    log_success "Cleanup complete"
}

create_network() {
    log_info "Creating network: $NETWORK_NAME"
    
    # Check if network already exists
    if podman network exists $NETWORK_NAME 2>/dev/null; then
        log_success "Network $NETWORK_NAME already exists - reusing"
    else
        podman network create $NETWORK_NAME
        log_success "Network created"
    fi
}

build_images() {
    log_info "Building container images..."
    
    # Build MCP Server (build from mcp/ to include performance-dashboard)
    # Build for local platform (ARM64 on Mac, AMD64 on Linux)
    log_info "Building MCP Server..."
    podman build --platform linux/$(uname -m | sed 's/x86_64/amd64/; s/aarch64/arm64/') \
        -t psap-mcp-server:local -f psap-mcp-server/Containerfile .
    log_success "MCP Server image built"
    
    # Build Agent
    log_info "Building PSAP Agent..."
    cd psap-agent
    podman build --platform linux/$(uname -m | sed 's/x86_64/amd64/; s/aarch64/arm64/') \
        -t psap-agent:local -f Containerfile .
    cd ..
    log_success "Agent image built"
    
    # Build Streamlit UI
    log_info "Building Streamlit UI..."
    cd psap-agent/examples
    podman build --platform linux/$(uname -m | sed 's/x86_64/amd64/; s/aarch64/arm64/') \
        -t streamlit-ui:local -f Containerfile .
    cd ../..
    log_success "Streamlit UI image built"
}

start_postgres() {
    log_info "Starting PostgreSQL..."
    
    podman run -d \
      --name psap-postgres \
      --network $NETWORK_NAME \
      -e POSTGRES_DB=psap \
      -e POSTGRES_USER=psap_user \
      -e POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
      -p 5432:5432 \
      docker.io/library/postgres:15-alpine
    
    log_info "Waiting for PostgreSQL to be ready..."
    sleep 10
    
    # Check if PostgreSQL is ready
    if podman exec psap-postgres pg_isready -U psap_user -d psap &> /dev/null; then
        log_success "PostgreSQL is ready"
    else
        log_error "PostgreSQL failed to start"
        exit 1
    fi
    
    # Create langfuse database
    log_info "Creating Langfuse database..."
    podman exec psap-postgres psql -U psap_user -d psap -c "CREATE DATABASE langfuse;" 2>/dev/null || true
    log_success "Langfuse database created"
}

start_langfuse() {
    log_info "Starting Langfuse v3 (ClickHouse + Redis + MinIO + Worker + Web)..."
    
    # Langfuse infra credentials come from .env (loaded at script start).
    # Validate they are present.
    for var in LF_CLICKHOUSE_USER LF_CLICKHOUSE_PASSWORD LF_REDIS_AUTH LF_MINIO_USER LF_MINIO_PASSWORD; do
        if [ -z "${!var:-}" ]; then
            log_error "$var is not set in .env — cannot start Langfuse"
            exit 1
        fi
    done
    
    # Auto-generate auth secrets if not provided in .env (first run)
    if [ -z "${NEXTAUTH_SECRET:-}" ]; then
        NEXTAUTH_SECRET=$(openssl rand -base64 32)
        log_info "Generated NEXTAUTH_SECRET (set it in .env to persist across restarts)"
    fi
    if [ -z "${SALT:-}" ]; then
        SALT=$(openssl rand -base64 32)
        log_info "Generated SALT (set it in .env to persist across restarts)"
    fi
    if [ -z "${ENCRYPTION_KEY:-}" ]; then
        ENCRYPTION_KEY=$(openssl rand -hex 32)
        log_info "Generated ENCRYPTION_KEY (set it in .env to persist across restarts)"
    fi
    
    # 1. Start ClickHouse
    # Note: --security-opt seccomp=unconfined is required on Apple Silicon (ARM64)
    # to prevent SIGILL (exit code 132) crashes in ClickHouse
    log_info "Starting ClickHouse..."
    podman run -d \
      --name langfuse-clickhouse \
      --network $NETWORK_NAME \
      --memory=4g \
      --security-opt seccomp=unconfined \
      -e CLICKHOUSE_DB=default \
      -e CLICKHOUSE_USER=$LF_CLICKHOUSE_USER \
      -e CLICKHOUSE_PASSWORD=$LF_CLICKHOUSE_PASSWORD \
      docker.io/clickhouse/clickhouse-server:24.3-alpine
    
    # 2. Start Redis
    log_info "Starting Redis..."
    podman run -d \
      --name langfuse-redis \
      --network $NETWORK_NAME \
      docker.io/redis:7 \
      --requirepass $LF_REDIS_AUTH --maxmemory-policy noeviction
    
    # 3. Start MinIO (S3-compatible storage)
    log_info "Starting MinIO..."
    podman run -d \
      --name langfuse-minio \
      --network $NETWORK_NAME \
      -e MINIO_ROOT_USER=$LF_MINIO_USER \
      -e MINIO_ROOT_PASSWORD=$LF_MINIO_PASSWORD \
      --entrypoint sh \
      cgr.dev/chainguard/minio \
      -c 'mkdir -p /data/langfuse && minio server --address ":9000" --console-address ":9001" /data'
    
    log_info "Waiting for Langfuse dependencies to start..."
    sleep 10
    
    # Common env vars for Langfuse web and worker
    LANGFUSE_COMMON_ENV="\
      -e DATABASE_URL=postgresql://psap_user:$POSTGRES_PASSWORD@psap-postgres:5432/langfuse \
      -e NEXTAUTH_URL=http://localhost:3000 \
      -e SALT=$SALT \
      -e ENCRYPTION_KEY=$ENCRYPTION_KEY \
      -e TELEMETRY_ENABLED=false \
      -e CLICKHOUSE_MIGRATION_URL=clickhouse://langfuse-clickhouse:9000 \
      -e CLICKHOUSE_URL=http://langfuse-clickhouse:8123 \
      -e CLICKHOUSE_USER=$LF_CLICKHOUSE_USER \
      -e CLICKHOUSE_PASSWORD=$LF_CLICKHOUSE_PASSWORD \
      -e CLICKHOUSE_CLUSTER_ENABLED=false \
      -e REDIS_HOST=langfuse-redis \
      -e REDIS_PORT=6379 \
      -e REDIS_AUTH=$LF_REDIS_AUTH \
      -e REDIS_TLS_ENABLED=false \
      -e LANGFUSE_S3_EVENT_UPLOAD_BUCKET=langfuse \
      -e LANGFUSE_S3_EVENT_UPLOAD_REGION=auto \
      -e LANGFUSE_S3_EVENT_UPLOAD_ACCESS_KEY_ID=$LF_MINIO_USER \
      -e LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY=$LF_MINIO_PASSWORD \
      -e LANGFUSE_S3_EVENT_UPLOAD_ENDPOINT=http://langfuse-minio:9000 \
      -e LANGFUSE_S3_EVENT_UPLOAD_FORCE_PATH_STYLE=true \
      -e LANGFUSE_S3_EVENT_UPLOAD_PREFIX=events/ \
      -e LANGFUSE_S3_MEDIA_UPLOAD_BUCKET=langfuse \
      -e LANGFUSE_S3_MEDIA_UPLOAD_REGION=auto \
      -e LANGFUSE_S3_MEDIA_UPLOAD_ACCESS_KEY_ID=$LF_MINIO_USER \
      -e LANGFUSE_S3_MEDIA_UPLOAD_SECRET_ACCESS_KEY=$LF_MINIO_PASSWORD \
      -e LANGFUSE_S3_MEDIA_UPLOAD_ENDPOINT=http://langfuse-minio:9000 \
      -e LANGFUSE_S3_MEDIA_UPLOAD_FORCE_PATH_STYLE=true \
      -e LANGFUSE_S3_MEDIA_UPLOAD_PREFIX=media/ \
      -e LANGFUSE_S3_BATCH_EXPORT_ENABLED=false \
      -e LANGFUSE_USE_AZURE_BLOB=false"
    
    # 4. Start Langfuse Worker
    log_info "Starting Langfuse Worker..."
    eval podman run -d \
      --name langfuse-worker \
      --network $NETWORK_NAME \
      $LANGFUSE_COMMON_ENV \
      docker.io/langfuse/langfuse-worker:3
    
    # 5. Start Langfuse Web
    log_info "Starting Langfuse Web..."
    eval podman run -d \
      --name langfuse-web \
      --network $NETWORK_NAME \
      $LANGFUSE_COMMON_ENV \
      -e NEXTAUTH_SECRET=$NEXTAUTH_SECRET \
      -e HOSTNAME=0.0.0.0 \
      -p 3000:3000 \
      docker.io/langfuse/langfuse:3
    
    log_info "Waiting for Langfuse v3 to initialize (migrations take ~30-60 seconds)..."
    sleep 40
    
    # Check if Langfuse is responding
    if curl -s -f http://localhost:3000/api/public/health &> /dev/null; then
        log_success "Langfuse v3 is ready"
        
        # Only prompt for keys if they're not already set
        if [ -z "$LANGFUSE_PUBLIC_KEY" ] || [ -z "$LANGFUSE_SECRET_KEY" ]; then
            echo ""
            log_warning "⏸️  IMPORTANT: Langfuse Setup Required!"
            echo "  1. Open http://localhost:3000 in your browser"
            echo "  2. Create an account and project"
            echo "  3. Generate API keys (Settings → API Keys)"
            echo ""
            read -p "Enter your LANGFUSE_PUBLIC_KEY (pk-lf-...): " LANGFUSE_PUBLIC_KEY
            read -p "Enter your LANGFUSE_SECRET_KEY (sk-lf-...): " LANGFUSE_SECRET_KEY
            
            if [ -n "$LANGFUSE_PUBLIC_KEY" ] && [ -n "$LANGFUSE_SECRET_KEY" ]; then
                export LANGFUSE_PUBLIC_KEY
                export LANGFUSE_SECRET_KEY
                log_success "Langfuse API keys saved for this session"
            else
                log_warning "No keys provided - agent will start without Langfuse"
            fi
        else
            log_success "Langfuse API keys detected - skipping setup"
        fi
    else
        log_warning "Langfuse might not be ready yet (migrations still running)"
    fi
}

start_mcp_server() {
    log_info "Starting MCP Server..."
    
    # Build S3 env vars if configured
    S3_ENV_VARS=""
    if [ -n "$S3_BUCKET" ]; then
        S3_ENV_VARS="-e S3_BUCKET=$S3_BUCKET -e S3_KEY=$S3_KEY -e S3_REGION=$S3_REGION -e S3_CACHE_TTL_SECONDS=$S3_CACHE_TTL_SECONDS"
        # Add AWS credentials if set
        if [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$AWS_SECRET_ACCESS_KEY" ]; then
            S3_ENV_VARS="$S3_ENV_VARS -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY"
        fi
        log_info "S3 data loading enabled: s3://$S3_BUCKET/$S3_KEY"
    else
        log_info "S3 not configured, using local data files"
    fi
    
    # PyTorch profile S3 prefix (profiles are loaded from S3 at runtime)
    PROFILE_ENV=""
    PROFILE_S3_PREFIX="${PROFILE_S3_PREFIX:-profiles/rhaiis}"
    if [ -n "$S3_BUCKET" ]; then
        PROFILE_ENV="-e PROFILE_S3_PREFIX=$PROFILE_S3_PREFIX"
        log_info "PyTorch profiles (S3): s3://$S3_BUCKET/$PROFILE_S3_PREFIX"
    else
        log_warning "S3_BUCKET not set -- PyTorch profile tools will be unavailable"
    fi
    PROFILE_MOUNT=""
    
    podman run -d \
      --name psap-mcp-server \
      --network $NETWORK_NAME \
      --memory=8g \
      -e MCP_HOST=0.0.0.0 \
      -e MCP_PORT=5001 \
      -e ENABLE_AUTH=false \
      -e POSTGRES_HOST=psap-postgres \
      -e POSTGRES_PORT=5432 \
      -e POSTGRES_DB=psap \
      -e POSTGRES_USER=psap_user \
      -e POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
      -e GRAFANA_URL="$GRAFANA_URL" \
      -e GRAFANA_API_TOKEN="$GRAFANA_API_TOKEN" \
      -e GRAFANA_DATASOURCE_UID="$GRAFANA_DATASOURCE_UID" \
      $S3_ENV_VARS \
      $PROFILE_ENV \
      $PROFILE_MOUNT \
      -p 5001:5001 \
      psap-mcp-server:local
    
    log_info "Waiting for MCP Server to be ready..."
    sleep 5
    
    # Check if MCP Server is responding
    if curl -s -f http://localhost:5001/health &> /dev/null; then
        log_success "MCP Server is ready"
    else
        log_warning "MCP Server might not be ready yet (this is sometimes normal)"
    fi
}

start_agent() {
    log_info "Starting PSAP Agent..."
    echo ""
    
    # Always prompt for Langfuse keys before starting the agent
    log_info "🔍 Langfuse Configuration"
    
    # Check if Langfuse is running
    if curl -s -f http://localhost:3000/api/public/health &> /dev/null; then
        log_success "Langfuse is running at http://localhost:3000"
        echo ""
        log_info "Please enter your Langfuse API keys:"
        echo "  (If you don't have keys yet, open http://localhost:3000 → Settings → API Keys)"
        echo "  (Press Enter to skip - agent will start without Langfuse)"
        echo ""
        
        read -p "LANGFUSE_PUBLIC_KEY (pk-lf-...): " LANGFUSE_PUBLIC_KEY
        read -p "LANGFUSE_SECRET_KEY (sk-lf-...): " LANGFUSE_SECRET_KEY
        
        if [ -n "$LANGFUSE_PUBLIC_KEY" ] && [ -n "$LANGFUSE_SECRET_KEY" ]; then
            export LANGFUSE_PUBLIC_KEY
            export LANGFUSE_SECRET_KEY
            log_success "Langfuse API keys configured ✅"
        else
            log_warning "No Langfuse keys provided - agent will start without Langfuse"
            LANGFUSE_PUBLIC_KEY=""
            LANGFUSE_SECRET_KEY=""
        fi
    else
        log_warning "Langfuse not accessible at http://localhost:3000"
        log_info "Agent will start without Langfuse integration"
        LANGFUSE_PUBLIC_KEY=""
        LANGFUSE_SECRET_KEY=""
    fi
    echo ""
    
    podman run -d \
      --name psap-agent \
      --network $NETWORK_NAME \
      -e AGENT_PORT=5002 \
      -e POSTGRES_HOST=psap-postgres \
      -e POSTGRES_PORT=5432 \
      -e POSTGRES_DB=psap \
      -e POSTGRES_USER=psap_user \
      -e POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
      -e GOOGLE_API_KEY="$GOOGLE_API_KEY" \
      -e MCP_SERVER_URL="http://psap-mcp-server:5001/mcp/" \
      -e ENABLE_PROMPT_CACHING=true \
      -e CACHE_TTL_HOURS=4 \
      -e LANGFUSE_PUBLIC_KEY="${LANGFUSE_PUBLIC_KEY:-}" \
      -e LANGFUSE_SECRET_KEY="${LANGFUSE_SECRET_KEY:-}" \
      -e LANGFUSE_HOST="http://langfuse-web:3000" \
      -p 5002:5002 \
      psap-agent:local
    
    log_info "Waiting for Agent to be ready..."
    sleep 10
    
    # Check if Agent is responding
    if curl -s -f http://localhost:5002/health &> /dev/null; then
        log_success "Agent is ready"
    else
        log_warning "Agent might not be ready yet"
    fi
}

start_streamlit() {
    log_info "Starting Streamlit UI..."
    
    podman run -d \
      --name streamlit-ui \
      --network $NETWORK_NAME \
      -e AGENT_API_URL="http://psap-agent:5002" \
      -p 8501:8501 \
      streamlit-ui:local
    
    log_info "Waiting for Streamlit to be ready..."
    sleep 5
    
    # Check if Streamlit is responding
    if curl -s -f http://localhost:8501/_stcore/health &> /dev/null; then
        log_success "Streamlit UI is ready"
    else
        log_warning "Streamlit might not be ready yet"
    fi
}

show_status() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    log_success "All containers are running!"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    
    # Show running containers
    log_info "Container Status:"
    podman ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "psap-|streamlit-ui|NAMES"
    echo ""
    
    # Show access URLs
    log_info "Access URLs:"
    echo "  🌐 Streamlit UI:  http://localhost:8501"
    echo "  📊 Langfuse UI:   http://localhost:3000"
    echo "  🤖 Agent API:     http://localhost:5002"
    echo "  🔧 MCP Server:    http://localhost:5001"
    echo "  🗄️  PostgreSQL:    localhost:5432"
    echo ""
    
    # Show useful commands
    log_info "Useful Commands:"
    echo "  📊 View container stats:  podman stats"
    echo "  📝 View logs:             podman logs -f <container-name>"
    echo "  🔍 Check health:          curl http://localhost:5002/health"
    echo "  🛑 Stop all:              podman stop streamlit-ui psap-agent psap-mcp-server langfuse-web langfuse-worker langfuse-minio langfuse-redis langfuse-clickhouse psap-postgres"
    echo ""
    
    log_info "Next Steps:"
    echo "  1. Open http://localhost:8501 in your browser"
    echo "  2. Login with your Red Hat email"
    echo "  3. Test a query: 'Show me throughput for Llama-3.1-70B on H200'"
    echo "  4. View traces and feedback in Langfuse: http://localhost:3000"
    echo ""
    
    log_warning "To view logs in real-time:"
    echo "  podman logs -f psap-agent"
    echo ""
    
    log_warning "📝 After making code changes:"
    echo "  ./test-local-containers.sh rebuild   # Rebuild images"
    echo "  ./test-local-containers.sh restart   # Restart containers"
    echo ""
}

view_logs() {
    log_info "Viewing container logs (Ctrl+C to exit)..."
    echo ""
    
    # Show logs from all containers
    podman logs -f psap-agent 2>&1 | sed 's/^/[AGENT] /' &
    AGENT_PID=$!
    
    podman logs -f psap-mcp-server 2>&1 | sed 's/^/[MCP] /' &
    MCP_PID=$!
    
    podman logs -f streamlit-ui 2>&1 | sed 's/^/[STREAMLIT] /' &
    STREAMLIT_PID=$!
    
    podman logs -f langfuse-web 2>&1 | sed 's/^/[LANGFUSE] /' &
    LANGFUSE_PID=$!
    
    # Wait for user to interrupt
    trap "kill $AGENT_PID $MCP_PID $STREAMLIT_PID $LANGFUSE_PID 2>/dev/null" EXIT
    wait
}

# Main execution
main() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "   🚀 PSAP Agent - Local Container Testing"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    
    check_prerequisites
    prompt_for_credentials
    cleanup_existing
    create_network
    build_images
    start_postgres
    start_langfuse
    start_mcp_server
    start_agent
    start_streamlit
    show_status
    
    # Ask if user wants to view logs
    read -p "View container logs? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        view_logs
    fi
}

# Handle script arguments
case "${1:-}" in
    cleanup)
        log_info "Running cleanup only..."
        cleanup_existing
        podman network rm $NETWORK_NAME 2>/dev/null || true
        log_success "Cleanup complete"
        ;;
    rebuild)
        log_info "Rebuilding images..."
        build_images
        log_success "Rebuild complete"
        ;;
    restart)
        log_info "Restarting app containers (keeping PostgreSQL & Langfuse)..."
        
        # Prompt for credentials in case they need updating
        prompt_for_credentials
        
        # Stop and remove only app containers
        for container in streamlit-ui psap-agent psap-mcp-server; do
            if podman ps -a --format '{{.Names}}' | grep -q "^${container}$"; then
                podman stop $container 2>/dev/null || true
                podman rm $container 2>/dev/null || true
            fi
        done
        
        # Check if network exists, create if needed
        if ! podman network exists $NETWORK_NAME 2>/dev/null; then
            create_network
        fi
        
        # Restart app containers
        start_mcp_server
        start_agent
        start_streamlit
        show_status
        ;;
    logs)
        view_logs
        ;;
    *)
        main "$@"
        ;;
esac

