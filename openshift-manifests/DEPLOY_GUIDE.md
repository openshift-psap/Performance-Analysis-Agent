# OpenShift Console Deployment Guide

**Namespace**: `psap-ai-agent`
**Registry**: `quay.io/<your-username>`

---

## Prerequisites Checklist

Before starting, ensure you have:

- [ ] Access to OpenShift Console
- [ ] Created 3 public repositories on Quay.io:
  - `quay.io/<your-username>/psap-mcp-server`
  - `quay.io/<your-username>/psap-agent`
  - `quay.io/<your-username>/streamlit-ui`
- [ ] Your Google Gemini API key
- [ ] Your Grafana credentials (optional)
- [ ] All 3 custom container images built and pushed to Quay

**Langfuse v3 Architecture Note**: This deployment uses Langfuse v3 which requires 5 backend components (ClickHouse, Redis, MinIO, Worker, Web) in addition to PostgreSQL. These use public Docker Hub images and do not need to be built. The deployment also requires 2 PersistentVolumeClaims (10Gi each for ClickHouse and MinIO data).

---

## Step 1: Build and Push Images (Local Machine)

Run the build script or use the manual commands below:

```bash
# Option A: Use the build script (update REGISTRY inside it first)
./BUILD_AND_PUSH.sh

# Option B: Manual commands
export REGISTRY="quay.io/<your-username>"
podman login quay.io

# Build and push MCP Server (from project root to include CSV)
cd /path/to/mcp
podman build -t $REGISTRY/psap-mcp-server:latest -f psap-mcp-server/Containerfile .
podman push $REGISTRY/psap-mcp-server:latest

# Build and push Agent
podman build -t $REGISTRY/psap-agent:latest -f psap-agent/Containerfile psap-agent/
podman push $REGISTRY/psap-agent:latest

# Build and push Streamlit UI
podman build -t $REGISTRY/streamlit-ui:latest -f psap-agent/examples/Containerfile psap-agent/examples/
podman push $REGISTRY/streamlit-ui:latest
```

**Verify on Quay.io**: Check that all 3 images are visible in your repositories.

---

## Step 2: Update Secrets

**BEFORE deploying**, update these files with your actual credentials.

There are **three** Secret resources in `01-secrets.yaml`:

### Secret 1: `psap-secrets` (application credentials)

1. Generate strong secrets:
   ```bash
   echo "POSTGRES_PASSWORD=$(openssl rand -base64 24)"
   echo "NEXTAUTH_SECRET=$(openssl rand -base64 32)"
   echo "LANGFUSE_SALT=$(openssl rand -hex 32)"
   echo "ENCRYPTION_KEY=$(openssl rand -hex 32)"
   ```

2. Update the following in `01-secrets.yaml` under `psap-secrets`:
   - `POSTGRES_PASSWORD`: Your generated password
   - `GOOGLE_API_KEY`: Your Google Gemini API key
   - `GRAFANA_API_TOKEN`: Your Grafana token (or remove if not using)
   - `GRAFANA_URL`: Your Grafana URL (or remove if not using)
   - `GRAFANA_DATASOURCE_UID`: Your datasource UID (or remove if not using)
   - `NEXTAUTH_SECRET`: Generated value
   - `LANGFUSE_SALT`: Generated value
   - `ENCRYPTION_KEY`: Generated value (new in Langfuse v3)

### Secret 2: `langfuse-v3-infra` (infrastructure credentials)

Update the following for the Langfuse v3 backend components:
   - `CLICKHOUSE_USER`: Username for ClickHouse (default: `clickhouse`)
   - `CLICKHOUSE_PASSWORD`: Strong password for ClickHouse
   - `REDIS_AUTH`: Password for Redis
   - `MINIO_ROOT_USER`: Username for MinIO (default: `minio`)
   - `MINIO_ROOT_PASSWORD`: Strong password for MinIO (minimum 8 characters)

### Secret 3: `langfuse-api-keys` (generated after deployment)

These are placeholder values. You will generate the actual keys in Step 3.7 after Langfuse is running.

**DO NOT deploy with `CHANGE_ME` values!**

---

## Step 3: Deploy via OpenShift Console

### 3.1: Create Namespace

1. Go to OpenShift Console
2. Click **Projects** -> **Create Project**
3. Name: `psap-ai-agent`
4. Click **Create**

### 3.2: Deploy Secrets

1. In the left menu, click **Workloads** -> **Secrets**
2. Click **Create** -> **From YAML**
3. Copy the entire content of `01-secrets.yaml` (this creates all 3 secrets)
4. Click **Create**

### 3.3: Deploy PostgreSQL

1. Click **Workloads** -> **Deployments**
2. Click **Create Deployment** -> **From YAML**
3. Copy the entire content of `02-postgresql.yaml`
4. Click **Create**
5. Wait for PostgreSQL pod to be **Running** (check **Pods** section)

### 3.4: Deploy Langfuse v3 Stack

`03-langfuse.yaml` deploys the complete Langfuse v3 stack in one apply:
- 2 PersistentVolumeClaims (ClickHouse data, MinIO data)
- 5 Deployments (ClickHouse, Redis, MinIO, Worker, Web)
- 4 Services (internal) + 1 Service (`langfuse` on port 3000)
- 1 Route (TLS edge termination)

**Deploy:**
1. Click **Workloads** -> **Deployments**
2. Click **Create Deployment** -> **From YAML**
3. Copy the entire content of `03-langfuse.yaml`
4. Click **Create**

**Verify startup order:**
1. Go to **Workloads** -> **Pods**
2. Wait for infrastructure pods to become **Running** first:
   - `langfuse-clickhouse-*`
   - `langfuse-redis-*`
   - `langfuse-minio-*`
3. Then verify application pods start:
   - `langfuse-worker-*`
   - `langfuse-web-*`

This may take 3-5 minutes. ClickHouse runs migrations on first boot. If worker or web pods crash-loop initially, they will recover once ClickHouse and Redis are ready.

### 3.5: Get Langfuse URL

1. Click **Networking** -> **Routes**
2. Find the `langfuse` route
3. Copy the **Location** URL (e.g., `https://langfuse-psap-ai-agent.apps.your-cluster.com`)

### 3.6: Update Langfuse NEXTAUTH_URL

The `NEXTAUTH_URL` must match the actual Route URL. Update it in **two** deployments:

1. In OpenShift Console, go to **Workloads** -> **Deployments** -> `langfuse-web`
2. Click **YAML** tab
3. Find `NEXTAUTH_URL` and update it with the route URL
4. Click **Save**
5. Repeat for **Workloads** -> **Deployments** -> `langfuse-worker`

Also update `NEXTAUTH_URL` in your local `03-langfuse.yaml` so future re-applies use the correct value.

### 3.7: Generate Langfuse API Keys

1. Open the Langfuse URL in your browser
2. Login with:
   - Email: from `LANGFUSE_INIT_USER_EMAIL` in `03-langfuse.yaml` (default: `admin@example.com`)
   - Password: from `LANGFUSE_INIT_USER_PASSWORD` in `03-langfuse.yaml`
3. Go to **Settings** -> **API Keys**
4. Click **Create New API Key**
5. Copy the **Public Key** (pk-lf-...) and **Secret Key** (sk-lf-...)

### 3.8: Update Langfuse API Keys Secret

1. In OpenShift Console, go to **Workloads** -> **Secrets**
2. Click on `langfuse-api-keys`
3. Click **Actions** -> **Edit Secret**
4. Update:
   - `LANGFUSE_PUBLIC_KEY`: Your public key (pk-lf-...)
   - `LANGFUSE_SECRET_KEY`: Your secret key (sk-lf-...)
5. Click **Save**

### 3.9: Deploy MCP Server

1. Click **Workloads** -> **Deployments**
2. Click **Create Deployment** -> **From YAML**
3. Copy the entire content of `04-mcp-server.yaml`
4. Click **Create**
5. Wait for pod to be **Running**

### 3.10: Deploy PSAP Agent

1. Click **Workloads** -> **Deployments**
2. Click **Create Deployment** -> **From YAML**
3. Copy the entire content of `05-agent.yaml`
4. Click **Create**
5. Wait for pod to be **Running**

### 3.11: Deploy Streamlit UI

1. Click **Workloads** -> **Deployments**
2. Click **Create Deployment** -> **From YAML**
3. Copy the entire content of `06-streamlit.yaml`
4. Click **Create**
5. Wait for pod to be **Running**

### 3.12: Get Streamlit URL

1. Click **Networking** -> **Routes**
2. Find the `streamlit-ui` route
3. Click the **Location** URL to open the app

---

## Step 4: Verify Deployment

### Check All Pods are Running:

1. Go to **Workloads** -> **Pods**
2. Ensure all 9 pods show **Running** status:
   - `psap-postgres-*`
   - `langfuse-clickhouse-*`
   - `langfuse-redis-*`
   - `langfuse-minio-*`
   - `langfuse-worker-*`
   - `langfuse-web-*`
   - `psap-mcp-server-*`
   - `psap-agent-*`
   - `streamlit-ui-*`

### Check Pod Logs:

If any pod is failing:
1. Click on the pod name
2. Click **Logs** tab
3. Look for error messages

### Test the Application:

1. Open the Streamlit URL
2. Enter your Red Hat email
3. Ask a test question like: "What models are available?"
4. Verify you get a response

### Check Langfuse Traces:

1. Open the Langfuse URL
2. Go to **Traces**
3. You should see traces from your test queries

---

## Troubleshooting

### Pod is CrashLoopBackOff

1. Check pod logs for errors
2. Common issues:
   - **PostgreSQL**: Check PVC is bound (`postgresql-data`)
   - **ClickHouse**: Check PVC is bound (`clickhouse-data`), check logs for SIGILL (CPU architecture mismatch -- the image must match your cluster architecture)
   - **Redis**: Check `REDIS_AUTH` secret is set in `langfuse-v3-infra`
   - **MinIO**: Check PVC is bound (`minio-data`), check `MINIO_ROOT_PASSWORD` is at least 8 characters
   - **Langfuse Worker/Web**: These depend on ClickHouse + Redis. If they crash-loop, check that ClickHouse and Redis pods are healthy first
   - **Agent/MCP**: Check secrets are correct

### Agent Can't Connect to MCP

- Verify MCP Server pod is running
- Check MCP Server service exists: `psap-mcp-server`
- Verify MCP Server logs show "ready to accept connections"

### Streamlit Can't Reach Agent

- Verify Agent pod is running
- Check Agent service exists: `psap-agent`
- Verify Agent logs show "Uvicorn running"

### Langfuse Web Shows "Internal Server Error"

- Check ClickHouse pod is running and healthy (logs should show "Ready for connections")
- Check Redis pod is running
- Check MinIO pod is running and `/minio/health/ready` returns 200
- Verify `CLICKHOUSE_URL`, `REDIS_HOST`, and S3 endpoint env vars are correct in the langfuse-web deployment
- Check langfuse-worker logs for migration errors

### ClickHouse Not Starting

- Verify PVC `clickhouse-data` is bound
- Check logs for `SIGILL` (illegal instruction) -- this means a CPU architecture mismatch. The `clickhouse-server:24.3-alpine` image supports both amd64 and arm64, but verify your cluster architecture
- Ensure the pod has enough memory (limit is 4Gi)

### Langfuse Worker/Web Can't Connect to ClickHouse

- Verify service `langfuse-clickhouse` exists and resolves
- Check ClickHouse is listening on ports 8123 (HTTP) and 9000 (native)
- Verify `CLICKHOUSE_USER` and `CLICKHOUSE_PASSWORD` match between `langfuse-v3-infra` secret and what ClickHouse was initialized with

---

## Post-Deployment Tasks

### 1. Update Grafana Dashboard URL (if using)

The agent's dashboard link generation tool is currently set to:
`https://aidash.app.intlab.redhat.com`

If this needs to be changed, you'll need to rebuild the MCP server image.

### 2. Configure Resource Limits

Monitor your pods and adjust resource requests/limits in the deployment YAMLs as needed. Default limits:
- ClickHouse: 4Gi memory
- Langfuse Web: 2Gi memory
- Langfuse Worker: 1Gi memory
- Redis: 512Mi memory
- MinIO: 1Gi memory

### 3. Set Up Monitoring

Configure alerts for:
- Pod health
- PostgreSQL storage usage
- ClickHouse and MinIO PVC storage usage
- API response times

---

## Success Indicators

Your deployment is successful when:

- All 9 pods are Running
- Streamlit UI is accessible
- You can ask questions and get responses
- Langfuse shows traces for your queries
- Feedback buttons work in Streamlit
- No errors in any pod logs

---

## Additional Resources

- Full deployment guide: `../OPENSHIFT_DEPLOYMENT.md`
- Technical architecture: `../TECHNICAL_ARCHITECTURE.md`
- Langfuse self-hosting: `../LANGFUSE_SELF_HOSTED.md`
