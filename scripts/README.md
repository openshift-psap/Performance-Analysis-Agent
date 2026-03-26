# Upload PyTorch Profiles to S3

A helper script to upload a single rank-0 PyTorch profiler trace file to the S3 bucket used by the PSAP MCP server for performance analysis.

## How It Works

The MCP server discovers PyTorch profiles by listing the S3 bucket at runtime. This script uploads one rank-0 `.json` trace file per invocation into the expected folder structure so the agent can find and analyze it automatically. Only rank 0 is uploaded to keep S3 costs low while providing a representative single-GPU trace for analysis.

**S3 structure:**

```
s3://<BUCKET>/<PREFIX>/<model>/<version>/<trace_file>.json
```

For example:

```
s3://psap-dashboard-data/profiles/rhaiis/
├── deepseek-r1/
│   ├── vLLM-0.11.2/
│   │   └── trace_rank0_pid455_range2000-2010.json
│   └── vLLM-0.13.0/
│       └── trace_rank0_pid467_range2000-2010.json
├── gpt-oss/
│   ├── vLLM-0.11.2/
│   │   └── trace_rank0_pid480_range2000-2010.json
│   └── vLLM-0.13.0/
│       └── trace_rank0_pid490_range2000-2010.json
└── llama-70b/
    └── ...
```

## Prerequisites

1. **AWS CLI** installed and configured:

   ```bash
   brew install awscli   # macOS
   aws configure         # set access key, secret, region (us-east-1)
   ```

2. The S3 bucket must already exist.

## Usage

```bash
./scripts/upload-profiles-to-s3.sh <accelerator> <model-name> <version> <local-folder>
```

| Argument       | Description                                          | Examples                          |
|----------------|------------------------------------------------------|-----------------------------------|
| `accelerator`  | Accelerator type                                     | `H200`, `MI300X`                  |
| `model-name`   | S3 model folder name                                 | `deepseek-r1`, `gpt-oss`, `llama-70b` |
| `version`      | Version folder name                                  | `rhaiis-3.2.5`, `vLLM-0.13.0`    |
| `local-folder` | Path to directory containing `.json` trace files     | `~/profiler/traces`, `/tmp/run1`  |

### Examples

```bash
# Upload DeepSeek traces for H200
./scripts/upload-profiles-to-s3.sh H200 deepseek-r1 rhaiis-3.2.5 ~/profiler/deepseek-v325

# Upload GPT-OSS traces for H200
./scripts/upload-profiles-to-s3.sh H200 gpt-oss vLLM-0.13.0 /tmp/gpt-oss-traces

# Upload for MI300X
./scripts/upload-profiles-to-s3.sh MI300X llama-70b vLLM-0.14.0 ./my-llama-traces
```

## File Selection

The script scans the local folder for `.json` files that are rank-0 traces. It supports two naming conventions:

- **Explicit rank**: `trace_rank0_pid455_range2000-2010.json` (contains `rank0`)
- **Positional rank**: `trace_1050_1060_0_20260225_002159.json` (3rd underscore-delimited field is `0`)

If multiple rank-0 files exist, the first one alphabetically is selected. All other files are skipped with a message.

## Environment Variables

| Variable             | Default                  | Description                                |
|----------------------|--------------------------|--------------------------------------------|
| `S3_BUCKET`          | `psap-dashboard-data`    | S3 bucket name                             |
| `PROFILE_S3_PREFIX`  | `profiles/rhaiis`        | S3 key prefix under the bucket             |

Override them if you're using a different bucket:

```bash
S3_BUCKET=my-bucket PROFILE_S3_PREFIX=my-prefix ./scripts/upload-profiles-to-s3.sh H200 deepseek-r1 rhaiis-3.2.5 ./traces
```

## Verifying Uploads

After uploading, verify the files are in S3:

```bash
aws s3 ls s3://psap-dashboard-data/profiles/rhaiis/H200/deepseek-r1/rhaiis-3.2.5/
```

## Troubleshooting: Incomplete Multipart Uploads

If an upload fails or is cancelled mid-transfer, S3 may leave behind incomplete multipart upload fragments. These are invisible in the S3 console but still consume storage. AWS will show a banner:

> *"Clean up incomplete multipart uploads — you might be storing multipart uploads that can't be viewed on the console."*

**List incomplete uploads:**

```bash
aws s3api list-multipart-uploads --bucket psap-dashboard-data
```

**Abort a specific incomplete upload:**

```bash
aws s3api abort-multipart-upload \
  --bucket psap-dashboard-data \
  --key "profiles/rhaiis/<model>/<version>/<filename>.json" \
  --upload-id "<UploadId from list command>"
```

**Abort all incomplete uploads at once:**

```bash
aws s3api list-multipart-uploads --bucket psap-dashboard-data \
  --query 'Uploads[].{Key: Key, UploadId: UploadId}' --output json \
  | python3 -c "
import json, subprocess, sys
for u in json.load(sys.stdin):
    print(f\"Aborting: {u['Key']} ({u['UploadId'][:20]}...)\")
    subprocess.run(['aws', 's3api', 'abort-multipart-upload',
                    '--bucket', 'psap-dashboard-data',
                    '--key', u['Key'], '--upload-id', u['UploadId']])
"
```



## Uploading vLLM Log Files

In addition to profiler traces, you can upload vLLM server log files into the same version folder. The MCP server will discover them automatically and make them available via `fetch_vllm_logs` and `compare_vllm_logs` tools.

**Naming**: The log file must contain "log" in its name and have a `.txt` or `.log` extension. Examples: `3.3-logs.txt`, `vllm-server.log`.

**Upload manually** (the upload script handles traces only):

```bash
aws s3 cp ./3.3-logs.txt s3://psap-dashboard-data/profiles/rhaiis/H200/gpt-oss-120b/rhaiis-3.3/3.3-logs.txt
```

The agent uses these logs to compare engine configurations (quantization, attention backend, CUDA graph settings, memory allocation) between versions, which is critical for explaining performance differences not visible in kernel-level profiling.

## What Happens After Upload

- The MCP server will **automatically discover** new profiles and log files on the next request (no restart needed).
- Discovery results are cached for **5 minutes**. After that, new uploads will be picked up.
- The AI agent can then analyze, compare, and provide insights on the uploaded traces and logs.
