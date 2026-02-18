#!/bin/bash
# Build and Push Container Images for OpenShift
# Usage: ./BUILD_AND_PUSH.sh [component]
#   component: mcp | agent | streamlit | all (default: all)

set -e

# Configuration — update REGISTRY to your Quay.io namespace
REGISTRY="quay.io/YOUR_QUAY_USERNAME"
MCP_IMAGE="$REGISTRY/psap-mcp-server:latest"
AGENT_IMAGE="$REGISTRY/psap-agent:latest"
STREAMLIT_IMAGE="$REGISTRY/streamlit-ui:latest"

# Go to project root — update this to your local project path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

COMPONENT="${1:-all}"

usage() {
    echo "Usage: $0 [component]"
    echo ""
    echo "Components:"
    echo "  mcp        Build & push MCP Server only"
    echo "  agent      Build & push PSAP Agent only"
    echo "  streamlit  Build & push Streamlit UI only"
    echo "  all        Build & push all images (default)"
    echo ""
    echo "Examples:"
    echo "  $0                # build & push everything"
    echo "  $0 streamlit      # build & push Streamlit UI only"
    echo "  $0 mcp            # build & push MCP Server only"
    exit 1
}

# Validate component
case "$COMPONENT" in
    mcp|agent|streamlit|all) ;;
    -h|--help) usage ;;
    *) echo "❌ Unknown component: $COMPONENT"; echo ""; usage ;;
esac

echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║          🏗️  BUILD & PUSH IMAGES TO QUAY                          ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""
if [ "$COMPONENT" != "all" ]; then
    echo "ℹ️  Building only: $COMPONENT"
    echo ""
fi

echo "Step 1: Login to Quay.io"
echo "─────────────────────────────────────────────────────────────────"
podman login quay.io
echo "✅ Logged in to Quay"
echo ""

# --- Build ---

if [ "$COMPONENT" = "all" ] || [ "$COMPONENT" = "mcp" ]; then
    echo "Build: MCP Server (includes CSV files)"
    echo "─────────────────────────────────────────────────────────────────"
    echo "Building from mcp/ directory to include performance-dashboard..."
    podman build -t $MCP_IMAGE \
        -f psap-mcp-server/Containerfile \
        .
    echo "✅ MCP Server image built"
    echo ""
fi

if [ "$COMPONENT" = "all" ] || [ "$COMPONENT" = "agent" ]; then
    echo "Build: PSAP Agent"
    echo "─────────────────────────────────────────────────────────────────"
    podman build -t $AGENT_IMAGE \
        -f psap-agent/Containerfile \
        psap-agent/
    echo "✅ Agent image built"
    echo ""
fi

if [ "$COMPONENT" = "all" ] || [ "$COMPONENT" = "streamlit" ]; then
    echo "Build: Streamlit UI"
    echo "─────────────────────────────────────────────────────────────────"
    podman build -t $STREAMLIT_IMAGE \
        -f psap-agent/examples/Containerfile \
        psap-agent/examples/
    echo "✅ Streamlit UI image built"
    echo ""
fi

# --- Push ---

if [ "$COMPONENT" = "all" ] || [ "$COMPONENT" = "mcp" ]; then
    echo "Push: MCP Server to Quay"
    echo "─────────────────────────────────────────────────────────────────"
    podman push $MCP_IMAGE
    echo "✅ MCP Server pushed"
    echo ""
fi

if [ "$COMPONENT" = "all" ] || [ "$COMPONENT" = "agent" ]; then
    echo "Push: PSAP Agent to Quay"
    echo "─────────────────────────────────────────────────────────────────"
    podman push $AGENT_IMAGE
    echo "✅ Agent pushed"
    echo ""
fi

if [ "$COMPONENT" = "all" ] || [ "$COMPONENT" = "streamlit" ]; then
    echo "Push: Streamlit UI to Quay"
    echo "─────────────────────────────────────────────────────────────────"
    podman push $STREAMLIT_IMAGE
    echo "✅ Streamlit UI pushed"
    echo ""
fi

echo "╔═══════════════════════════════════════════════════════════════════╗"
if [ "$COMPONENT" = "all" ]; then
    echo "║          ✅ ALL IMAGES BUILT AND PUSHED!                          ║"
else
    echo "║          ✅ $COMPONENT IMAGE BUILT AND PUSHED!                    ║"
fi
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Images on Quay.io:"
if [ "$COMPONENT" = "all" ] || [ "$COMPONENT" = "mcp" ]; then
    echo "  • $MCP_IMAGE"
fi
if [ "$COMPONENT" = "all" ] || [ "$COMPONENT" = "agent" ]; then
    echo "  • $AGENT_IMAGE"
fi
if [ "$COMPONENT" = "all" ] || [ "$COMPONENT" = "streamlit" ]; then
    echo "  • $STREAMLIT_IMAGE"
fi
echo ""
echo "Langfuse v3 components use public images (no build needed):"
echo "  • clickhouse/clickhouse-server:24.3-alpine"
echo "  • redis:7"
echo "  • minio/minio:latest"
echo "  • langfuse/langfuse-worker:3"
echo "  • langfuse/langfuse:3"
echo ""
echo "Next steps:"
echo "  1. Update secrets in 01-secrets.yaml (psap-secrets + langfuse-v3-infra)"
echo "  2. Deploy to OpenShift — see DEPLOY_GUIDE.md"
