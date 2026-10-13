# Ansible Log Monitor - OpenShift Deployment

A Helm-based deployment for the Ansible Log Monitor system on OpenShift.

> **Note**: All commands use the current OpenShift project by default. No need to specify `NAMESPACE` unless you want to use a different one.

## Components

The Ansible Log Monitor stack includes the following components:

- **Backend** - FastAPI application with LangGraph agentic workflow
- **UI** - Gradio-based user interface for log analysis
- **Annotation Interface** - Tool for improving AI workflow
- **AAP Mock** - Mock Ansible Automation Platform log generator (for testing)
- **Loki Stack** - Log aggregation (Loki + Alloy/Promtail + Grafana)
- **PostgreSQL** (pgvector) - Database for storing processed alerts
- **aws-compatible-storage** (S4) - S3-compatible object storage for artifacts and Loki chunks
- **MCP Servers** - Model Context Protocol servers (Loki integration)
- **Phoenix** - Observability and tracing

## Prerequisites

- OpenShift CLI (`oc`) installed and configured
- Helm 3.x installed
- Access to an OpenShift cluster
- MaaS API Token for AI services

## Quick Start

### 1. Install the Application

```bash
# Basic installation (uses current OpenShift project)
make install OPENAI_API_TOKEN=your-token-here

# With custom AI model settings
make install NAMESPACE=ansible-logs-monitor \
  OPENAI_API_TOKEN=your-token-here \
  OPENAI_MODEL=llama-4-scout-17b-16e-w4a16 \
  OPENAI_TEMPERATURE=0.7

# Specify a custom namespace (optional)
make install NAMESPACE=ansible-logs-monitor OPENAI_API_TOKEN=your-token-here
```

### 2. Access the Services

```bash
# Forward UI to localhost:7860 (uses current OpenShift project)
make port-forward-ui

# Forward Backend API to localhost:8000
make port-forward-backend

# Forward Annotation Interface to localhost:7861
make port-forward-annotation

# Forward Grafana to localhost:3000
make port-forward-grafana

# Specify a custom namespace (optional)
make port-forward-ui NAMESPACE=ansible-logs-monitor
```

### 3. Uninstall

```bash
# Uninstall from current OpenShift project
make uninstall

# Uninstall from specific namespace (optional)
make uninstall NAMESPACE=ansible-logs-monitor
```

## Available Commands

| Command | Description | Required Parameters |
|---------|-------------|-------------------|
| `make install` | Deploy the application | `OPENAI_API_TOKEN` |
| `make uninstall` | Remove the application | None |
| `make namespace` | Create/switch to namespace | None |
| `make port-forward-ui` | Forward UI service | None |
| `make port-forward-backend` | Forward Backend service | None |
| `make port-forward-annotation` | Forward Annotation Interface | None |
| `make port-forward-grafana` | Forward Grafana | None |
| `make help` | Show all available commands | None |

**Note**: All commands use the current OpenShift project by default. Use `NAMESPACE=your-namespace` to override.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_TOKEN` | *Required* | MaaS API Token for AI services |
| `OPENAI_API_ENDPOINT` | `https://llama-4-scout-17b-16e-w4a16-maas-apicast-production.apps.prod.rhoai.rh-aiservices-bu.com:443/v1` | AI API endpoint |
| `OPENAI_MODEL` | `llama-4-scout-17b-16e-w4a16` | AI model to use |
| `OPENAI_TEMPERATURE` | `0.7` | AI temperature setting |

### Examples

#### Basic Installation
```bash
# Uses current OpenShift project
make install OPENAI_API_TOKEN=abc123

# Specify custom namespace
make install NAMESPACE=my-ansible-logs OPENAI_API_TOKEN=abc123
```

#### Custom Configuration
```bash
# Uses current OpenShift project with custom settings
make install \
  OPENAI_API_TOKEN=abc123 \
  OPENAI_MODEL=custom-model \
  OPENAI_TEMPERATURE=0.5
```

#### Disable AAP Mock (for production with real AAP logs)

To disable the mock log generator, edit `ansible-log-monitor/values.yaml`:

```yaml
aap-mock:
  enabled: false
```

Then run:
```bash
make install
```

## Sub-Charts

The `ansible-log-monitor` chart includes several sub-charts that can be individually configured:

### AAP Mock (aap-mock)
Mock Ansible Automation Platform log generator for testing and demonstration.

- **Documentation**: `charts/aap-mock/README.md`
- **Enabled by default**: Yes
- **Disable**: Set `aap-mock.enabled: false` in `values.yaml`
- **Configure**: See `charts/aap-mock/values.yaml`

Example - to increase storage for aap-mock, edit `ansible-log-monitor/values.yaml`:
```yaml
aap-mock:
  persistence:
    data:
      size: 5Gi
    logs:
      size: 2Gi
```

Then run:
```bash
make install
```

### Other Sub-Charts
- **backend** - Main ALM backend service
- **ui** - Gradio UI (`charts/ui/README.md`)
- **annotation-interface** - Annotation tool (`charts/annotation-interface/README.md`)
- **pgvector** - PostgreSQL database
- **aws-compatible-storage** - S3-compatible object storage (S4)
- **mcp-servers** - MCP protocol servers
- **phoenix** - Observability platform

### Object Storage (aws-compatible-storage / S4)

Object storage is provided by the shared
[`aws-compatible-storage`](https://github.com/rh-ai-quickstart/ai-architecture-charts/tree/main/aws-compatible-storage)
chart from ai-architecture-charts, which packages
[S4](https://github.com/rh-aiservices-bu/s4) (Ceph RGW + web UI). It replaces the
MinIO subchart that earlier releases used, and also backs Loki's chunk store.

| Item | Value |
|------|-------|
| In-cluster S3 endpoint | `http://aws-compatible-storage:7480` |
| Web UI (Route) | Route `aws-compatible-storage`, service port `5000` |
| Health endpoint | `http://aws-compatible-storage:5000/api` (there is no `/minio/health/*`) |
| Credentials Secret | `aws-compatible-storage-credentials` → `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` |
| S3 API Route | Disabled (`route.s3Api.enabled: false`) - all consumers are in-cluster |
| Data PVC | `aws-compatible-storage-data` (10Gi) |

Buckets are **not** created by the subchart. The parent chart runs a regular Job,
`<release>-object-storage-buckets`, which waits for the S3 API, creates each bucket
in `awsCompatibleStorageBuckets.names` idempotently, and runs a read/write smoke
test. It is deliberately a regular Job rather than a `post-install` hook, because
Helm runs `--wait` *before* hooks and both Loki and the RAG init job block until
their buckets exist.

| Bucket | Used by |
|--------|---------|
| `rag-index` | RAG FAISS index, metadata, `LATEST.json` |
| `clustering-model` | `clustering_model.joblib` |
| `loki-chunks` / `loki-ruler` / `loki-admin` | Loki |

Application services use `S3_*` environment variable names. The app talks
plain S3 through the `minio-py` library.

> **Credentials:** `values.yaml` ships demo credentials (`s4admin` / `s4secret`, UI
> `admin` / `alm_password`). Override them with a secrets overlay
> (`helm install ... -f secrets.yaml`) for anything beyond a throwaway namespace.
> `aws-compatible-storage.s3.*` and `loki.loki.storage.s3.*` must stay in sync.

To point at a different service name, change `aws-compatible-storage.fullnameOverride`
**and** `global.objectStorage.host` in `values.yaml`, plus the literal endpoint in
`charts/{backend,clustering,rag}/values.yaml` and `loki.loki.storage.s3.endpoint`.

## Log Collection

The main `ansible-log-monitor` chart includes Alloy configuration that automatically collects logs from all pods in the cluster, including `aap-mock`. No additional configuration is needed.

For a standalone Alloy configuration example specific to aap-mock, see `config/alloy/alloy-aap-mock.alloy`.
