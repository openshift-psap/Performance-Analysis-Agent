# Upload PyTorch Profiles to S3

A helper script to upload PyTorch profiler trace files to the S3 bucket used by the PSAP MCP server for performance analysis.

## How It Works

The MCP server discovers PyTorch profiles by listing the S3 bucket at runtime. This script uploads `.json` trace files into the expected folder structure so the agent can find and analyze them automatically.

**S3 structure:**

```
s3://<BUCKET>/<PREFIX>/<model>/<version>/<trace_file>.json
```

For example:

```
s3://psap-dashboard-data/profiles/rhaiis/
├── deepseek-r1/
│   ├── vLLM-0.11.2/
│   │   ├── trace_rank0_pid455_range2000-2010.json
│   │   ├── trace_rank1_pid456_range2000-2010.json
│   │   └── ...
│   └── vLLM-0.13.0/
│       ├── trace_rank0_pid467_range2000-2010.json
│       └── ...
├── gpt-oss/
│   ├── vLLM-0.11.2/
│   │   └── ...
│   └── vLLM-0.13.0/
│       └── ...
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
./scripts/upload-profiles-to-s3.sh <model-name> <version> <local-folder>
```

| Argument       | Description                                          | Examples                          |
|----------------|------------------------------------------------------|-----------------------------------|
| `model-name`   | S3 model folder name                                 | `deepseek-r1`, `gpt-oss`, `llama-70b` |
| `version`      | vLLM version folder name                             | `vLLM-0.11.2`, `vLLM-0.13.0`, `vLLM-0.14.0` |
| `local-folder` | Path to directory containing `.json` trace files     | `~/profiler/traces`, `/tmp/run1`  |

### Examples

```bash
# Upload DeepSeek traces for vLLM 0.11.2
./scripts/upload-profiles-to-s3.sh deepseek-r1 vLLM-0.11.2 ~/profiler/deepseek-v0112

# Upload GPT-OSS traces for vLLM 0.13.0
./scripts/upload-profiles-to-s3.sh gpt-oss vLLM-0.13.0 /tmp/gpt-oss-traces

# Upload a new model
./scripts/upload-profiles-to-s3.sh llama-70b vLLM-0.14.0 ./my-llama-traces
```

## File Naming Requirements

Trace filenames **must** contain `rank` so the agent can identify which GPU rank each trace belongs to. Common patterns:

- `trace_rank0_pid455_range2000-2010.json`
- `trace_rank3_pid470_range2000-2010.json`
- `rank0_forward_pass.json`

Files without `rank` in the name will be skipped with a warning.

## Environment Variables

| Variable             | Default                  | Description                                |
|----------------------|--------------------------|--------------------------------------------|
| `S3_BUCKET`          | `psap-dashboard-data`    | S3 bucket name                             |
| `PROFILE_S3_PREFIX`  | `profiles/rhaiis`        | S3 key prefix under the bucket             |

Override them if you're using a different bucket:

```bash
S3_BUCKET=my-bucket PROFILE_S3_PREFIX=my-prefix ./scripts/upload-profiles-to-s3.sh deepseek-r1 vLLM-0.11.2 ./traces
```

## Verifying Uploads

After uploading, verify the files are in S3:

```bash
aws s3 ls s3://psap-dashboard-data/profiles/rhaiis/deepseek-r1/vLLM-0.11.2/
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



## What Happens After Upload

- The MCP server will **automatically discover** the new profiles on the next request (no restart needed).
- Profile discovery results are cached for **5 minutes**. After that, new uploads will be picked up.
- The AI agent can then analyze, compare, and provide insights on the uploaded profiles.
