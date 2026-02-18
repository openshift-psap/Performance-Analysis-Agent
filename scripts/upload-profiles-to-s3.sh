#!/usr/bin/env bash
# =============================================================================
# Upload PyTorch profile traces to S3
# =============================================================================
# Uploads all .json trace files from a local folder into the S3 structure
# expected by the MCP server:
#
#   s3://<BUCKET>/<PREFIX>/<model>/<version>/<filename>.json
#
# The MCP server discovers profiles by listing S3 folders automatically.
# No manifest or stats files are needed -- just upload the raw Chrome trace
# JSON files and the agent will find them.
#
# Usage:
#   ./scripts/upload-profiles-to-s3.sh <model> <version> <folder>
#
# Examples:
#   ./scripts/upload-profiles-to-s3.sh deepseek-r1 vLLM-0.11.2 ~/profiler/deepseek-profiles
#   ./scripts/upload-profiles-to-s3.sh gpt-oss vLLM-0.13.0 /tmp/gpt-oss-traces
#   ./scripts/upload-profiles-to-s3.sh llama-70b vLLM-0.14.0 ./my-traces
#
# The files should contain "rank" in their name (e.g. trace_rank0_pid455.json)
# so the agent can identify which GPU rank each trace belongs to.
#
# Prerequisites:
#   - AWS CLI configured (aws configure) or environment variables set
#   - The S3 bucket must already exist
# =============================================================================

set -euo pipefail

# ---------- Configuration ----------
BUCKET="${S3_BUCKET:-psap-dashboard-data}"
PREFIX="${PROFILE_S3_PREFIX:-profiles/rhaiis}"

# ---------- Functions ----------
log_info()  { echo "  ℹ️  $*"; }
log_ok()    { echo "  ✅ $*"; }
log_skip()  { echo "  ⏭️  $*"; }
log_err()   { echo "  ❌ $*" >&2; }

usage() {
    echo "Usage: $0 <model-name> <version> <local-folder>"
    echo ""
    echo "Arguments:"
    echo "  model-name    S3 model folder name (e.g. deepseek-r1, gpt-oss, llama-70b)"
    echo "  version       Version folder name   (e.g. vLLM-0.11.2, vLLM-0.13.0, vLLM-0.14.0)"
    echo "  local-folder  Path to directory containing .json trace files"
    echo ""
    echo "Examples:"
    echo "  $0 deepseek-r1 vLLM-0.11.2 ~/profiler/deepseek-profiles"
    echo "  $0 gpt-oss vLLM-0.13.0 /tmp/gpt-oss-traces"
    echo ""
    echo "Environment variables (optional):"
    echo "  S3_BUCKET           S3 bucket name     (default: psap-dashboard-data)"
    echo "  PROFILE_S3_PREFIX   S3 key prefix      (default: profiles/rhaiis)"
    exit 1
}

# ---------- Argument parsing ----------
if [ $# -lt 3 ]; then
    usage
fi

MODEL_NAME="$1"
VERSION="$2"
LOCAL_FOLDER="$3"

# ---------- Validation ----------
echo "========================================"
echo "Upload PyTorch Profiles to S3"
echo "========================================"
echo ""
echo "  Model:       $MODEL_NAME"
echo "  Version:     $VERSION"
echo "  Source:      $LOCAL_FOLDER"
echo "  Destination: s3://$BUCKET/$PREFIX/$MODEL_NAME/$VERSION/"
echo ""

if ! command -v aws &> /dev/null; then
    log_err "AWS CLI not found. Install with: brew install awscli"
    exit 1
fi

if [ ! -d "$LOCAL_FOLDER" ]; then
    log_err "Local folder not found: $LOCAL_FOLDER"
    exit 1
fi

# ---------- Scan files ----------
UPLOAD_FILES=()
SKIP_FILES=()

for json_file in "$LOCAL_FOLDER"/*.json; do
    [ -f "$json_file" ] || continue
    filename="$(basename "$json_file")"

    if [[ ! "$filename" =~ rank ]]; then
        SKIP_FILES+=("$filename")
    else
        UPLOAD_FILES+=("$filename")
    fi
done

if [ ${#UPLOAD_FILES[@]} -eq 0 ] && [ ${#SKIP_FILES[@]} -eq 0 ]; then
    log_err "No .json files found in $LOCAL_FOLDER"
    exit 1
fi

# ---------- Display file listing ----------
echo "  Files to upload (${#UPLOAD_FILES[@]}):"
for f in "${UPLOAD_FILES[@]}"; do
    echo "    ✔  $f"
done

if [ ${#SKIP_FILES[@]} -gt 0 ]; then
    echo ""
    echo "  Files to skip — no 'rank' in filename (${#SKIP_FILES[@]}):"
    for f in "${SKIP_FILES[@]}"; do
        echo "    ⏭  $f"
    done
fi

echo ""

if [ ${#UPLOAD_FILES[@]} -eq 0 ]; then
    log_err "No uploadable files (all files missing 'rank' in filename)"
    exit 1
fi

# ---------- Confirmation ----------
read -p "Proceed with upload? [y/N] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi
echo ""

# ---------- Upload ----------
TOTAL=0

for filename in "${UPLOAD_FILES[@]}"; do
    json_file="${LOCAL_FOLDER}/${filename}"
    s3_key="${PREFIX}/${MODEL_NAME}/${VERSION}/${filename}"

    log_info "$filename -> s3://$BUCKET/$s3_key"
    if ! aws s3 cp "$json_file" "s3://$BUCKET/$s3_key"; then
        log_err "Failed to upload $filename"
        exit 1
    fi
    TOTAL=$((TOTAL + 1))
done

# ---------- Summary ----------
echo ""
echo "========================================"
log_ok "Uploaded $TOTAL file(s) to s3://$BUCKET/$PREFIX/$MODEL_NAME/$VERSION/"
if [ ${#SKIP_FILES[@]} -gt 0 ]; then
    log_skip "Skipped ${#SKIP_FILES[@]} file(s) (no 'rank' in filename)"
fi
echo "========================================"
echo ""
echo "Verify with:"
echo "  aws s3 ls s3://$BUCKET/$PREFIX/$MODEL_NAME/$VERSION/"
echo ""
echo "The MCP server will discover these profiles automatically."
echo "No restart is needed -- profiles are discovered on each request"
echo "(cached for 5 minutes)."
