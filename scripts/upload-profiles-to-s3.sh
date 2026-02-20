#!/usr/bin/env bash
# =============================================================================
# Upload PyTorch profile traces to S3
# =============================================================================
# Uploads a single rank-0 .json trace file from a local folder into the S3
# structure expected by the MCP server:
#
#   s3://<BUCKET>/<PREFIX>/<model>/<version>/<filename>.json
#
# Only rank 0 is uploaded (one file per folder). This keeps S3 costs low
# while giving the agent a representative single-GPU trace to analyze.
#
# The MCP server discovers profiles by listing S3 folders automatically.
# No manifest or stats files are needed -- just upload the raw Chrome trace
# JSON file and the agent will find it.
#
# Usage:
#   ./scripts/upload-profiles-to-s3.sh <model> <version> <folder>
#
# Examples:
#   ./scripts/upload-profiles-to-s3.sh deepseek-r1 vLLM-0.11.2 ~/profiler/deepseek-profiles
#   ./scripts/upload-profiles-to-s3.sh gpt-oss vLLM-0.13.0 /tmp/gpt-oss-traces
#   ./scripts/upload-profiles-to-s3.sh llama-70b vLLM-0.14.0 ./my-traces
#
# The selected file must contain "rank0" in its name (e.g. trace_rank0_pid455.json).
# If multiple rank-0 files exist, the first one (alphabetically) is used.
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

# ---------- Scan for rank-0 file ----------
RANK0_FILES=()
SKIP_FILES=()

for json_file in "$LOCAL_FOLDER"/*.json; do
    [ -f "$json_file" ] || continue
    filename="$(basename "$json_file")"

    if [[ "$filename" =~ rank0 ]]; then
        RANK0_FILES+=("$filename")
    else
        SKIP_FILES+=("$filename")
    fi
done

if [ ${#RANK0_FILES[@]} -eq 0 ] && [ ${#SKIP_FILES[@]} -eq 0 ]; then
    log_err "No .json files found in $LOCAL_FOLDER"
    exit 1
fi

if [ ${#RANK0_FILES[@]} -eq 0 ]; then
    log_err "No rank-0 trace files found (expected filename containing 'rank0')"
    echo ""
    echo "  Files in folder:"
    for f in "${SKIP_FILES[@]}"; do
        echo "    ⏭  $f"
    done
    exit 1
fi

UPLOAD_FILE="${RANK0_FILES[0]}"
UPLOAD_FILES=("$UPLOAD_FILE")

# ---------- Display file listing ----------
echo "  File to upload (rank 0):"
echo "    ✔  $UPLOAD_FILE"

if [ ${#RANK0_FILES[@]} -gt 1 ]; then
    echo ""
    echo "  Other rank-0 files skipped (only uploading first match):"
    for f in "${RANK0_FILES[@]:1}"; do
        echo "    ⏭  $f"
    done
fi

if [ ${#SKIP_FILES[@]} -gt 0 ]; then
    echo ""
    echo "  Non-rank-0 files skipped (${#SKIP_FILES[@]}):"
    for f in "${SKIP_FILES[@]}"; do
        echo "    ⏭  $f"
    done
fi

echo ""

# ---------- Confirmation ----------
read -p "Proceed with upload? [y/N] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi
echo ""

# ---------- Upload with retries ----------
MAX_RETRIES=3
SUCCEEDED=0
FAILED=0
FAILED_FILES=()

upload_and_verify() {
    local json_file="$1"
    local s3_key="$2"

    aws s3 cp --no-progress "$json_file" "s3://$BUCKET/$s3_key" >/dev/null 2>&1 || return 1
    aws s3api head-object --bucket "$BUCKET" --key "$s3_key" >/dev/null 2>&1 || return 1
    return 0
}

PENDING_FILES=("${UPLOAD_FILES[@]}")

for attempt in $(seq 1 $MAX_RETRIES); do
    STILL_FAILING=()

    if [ "$attempt" -gt 1 ]; then
        echo ""
        log_info "Retry attempt $attempt/$MAX_RETRIES for ${#PENDING_FILES[@]} file(s)..."
        sleep 2
    fi

    for filename in "${PENDING_FILES[@]}"; do
        json_file="${LOCAL_FOLDER}/${filename}"
        s3_key="${PREFIX}/${MODEL_NAME}/${VERSION}/${filename}"

        [ "$attempt" -eq 1 ] && log_info "$filename -> s3://$BUCKET/$s3_key"

        if upload_and_verify "$json_file" "$s3_key"; then
            log_ok "$filename"
            SUCCEEDED=$((SUCCEEDED + 1))
        else
            if [ "$attempt" -eq "$MAX_RETRIES" ]; then
                log_err "$filename (failed after $MAX_RETRIES attempts)"
                FAILED=$((FAILED + 1))
                FAILED_FILES+=("$filename")
            else
                log_err "$filename (will retry)"
                STILL_FAILING+=("$filename")
            fi
        fi
    done

    PENDING_FILES=("${STILL_FAILING[@]}")
    [ ${#PENDING_FILES[@]} -eq 0 ] && break
done

# ---------- Summary ----------
echo ""
echo "========================================"
if [ $SUCCEEDED -gt 0 ]; then
    log_ok "Uploaded $UPLOAD_FILE to s3://$BUCKET/$PREFIX/$MODEL_NAME/$VERSION/"
fi
if [ $FAILED -gt 0 ]; then
    log_err "Failed to upload $UPLOAD_FILE after $MAX_RETRIES attempts"
fi
echo "========================================"
echo ""
echo "Verify with:"
echo "  aws s3 ls s3://$BUCKET/$PREFIX/$MODEL_NAME/$VERSION/"
echo ""
echo "The MCP server will discover these profiles automatically."
echo "No restart is needed -- profiles are discovered on each request"
echo "(cached for 5 minutes)."

if [ $FAILED -gt 0 ]; then
    exit 1
fi
