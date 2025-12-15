# RAG Service Migration Guide

## Overview

This document describes the migration from PVC-based RAG storage to a dedicated RAG microservice with PostgreSQL storage. This change eliminates ReadWriteOnce (RWO) constraints, reduces resource duplication, and simplifies the architecture.

## What Changed

### Before (PVC-based)
- RAG index stored on PersistentVolumeClaim (PVC)
- Each backend pod loaded FAISS index from PVC
- All backend pods required to be on same node (RWO constraint)
- N backend pods = N copies of FAISS index in memory
- Index updates required PVC rebuild and pod restarts

### After (RAG Service + PostgreSQL)
- RAG index stored in PostgreSQL (`ragembedding` table)
- Single RAG service loads FAISS index from PostgreSQL
- Backend pods make HTTP calls to RAG service
- Backend pods can run on any node (no constraints)
- 1 RAG service = 1 copy of FAISS index in memory
- Index updates via PostgreSQL (no pod restarts needed)

## Architecture

```
┌─────────────────────┐
│   Init Job Pod      │
│  (alm-backend-init) │
│                     │
│  1. Parse PDFs      │
│  2. Generate        │
│     embeddings      │
│  3. Save to        │
│     PostgreSQL      │
└──────────┬──────────┘
           │
           │ Writes embeddings
           ▼
┌─────────────────────┐
│   PostgreSQL        │
│                     │
│  - ragembedding     │
│    table            │
│  - pgvector         │
│    extension        │
└──────────┬──────────┘
           │
           │ Reads embeddings
           │ (polls every 5s)
           ▼
┌─────────────────────┐
│   RAG Service Pod   │
│  (alm-rag)          │
│                     │
│  ┌───────────────┐  │
│  │ Background    │  │
│  │ Task: Poll    │  │
│  │ PostgreSQL    │  │
│  └───────────────┘  │
│                     │
│  ┌───────────────┐  │
│  │ FAISS Index   │  │ (in-memory)
│  │ (loaded from  │  │
│  │  PostgreSQL)  │  │
│  └───────────────┘  │
└──────────┬──────────┘
           │
           │ HTTP /rag/query
           │
           ▼
┌─────────────────────┐
│  Backend Pods       │
│  (alm-backend)      │
│                     │
│  - Pod 1            │
│  - Pod 2            │
│  - Pod N            │
│                     │
│  All make HTTP      │
│  calls to RAG       │
│  service            │
└─────────────────────┘
```

## Init Job and RAG Service Relationship

### Overview

The init job and RAG service have a **producer-consumer relationship** coordinated through PostgreSQL:

- **Init Job** = **Producer**: Creates and saves embeddings to PostgreSQL
- **RAG Service** = **Consumer**: Reads embeddings from PostgreSQL and serves queries
- **PostgreSQL** = **Coordination Point**: Shared data store, no direct communication needed

### Key Characteristics

1. **No Direct Dependency**: Services don't wait for each other to start
2. **Asynchronous Coordination**: RAG service polls PostgreSQL, init job polls RAG service HTTP endpoint
3. **Graceful Degradation**: Both services can start independently and handle missing data gracefully
4. **Data Persistence**: Embeddings persist in PostgreSQL across pod restarts

### Detailed Flow and Timeline

```
Time    Init Job                    PostgreSQL              RAG Service
─────────────────────────────────────────────────────────────────────────
T+0s    Pod starts                 ──                      Pod starts
        │                          │                       │
T+5s    Wait for PostgreSQL        ──                      Wait for PostgreSQL
        │                          │                       │
T+10s   ──                         Ready                   ──
        │                          │                       │
T+15s   PostgreSQL ready!          ──                      PostgreSQL ready!
        │                          │                       │
        │                          │                       Start background task
        │                          │                       Poll for embeddings...
        │                          │                       (no embeddings yet)
        │                          │                       │
T+30s   Building RAG index...      ──                      Still polling...
        - Parse PDFs                │                       (every 5 seconds)
        - Generate embeddings       │                       │
        │                          │                       │
T+60s   Saving embeddings...       Writing embeddings...   ──
        │                          │                       │
T+65s   Index complete!            Embeddings saved!       ──
        │                          │                       │
T+70s   ──                         ──                      Found embeddings!
        │                          │                       Loading index...
        │                          │                       │
T+75s   ──                         ──                      Index loaded! ✓
        │                          │                       Service ready!
        │                          │                       │
T+80s   Waiting for RAG service... ──                      ──
        (polls /ready endpoint)     │                       │
        │                          │                       │
T+85s   RAG service ready!          ──                      ──
        │                          │                       │
T+90s   Running training pipeline   ──                      ──
        (uses RAG service)          │                       │
        │                          │                       │
T+95s   Querying RAG service...    ──                      Serving queries ✓
        │                          │                       │
```

### Phase-by-Phase Breakdown

#### Phase 1: Parallel Startup (T+0s to T+15s)
- **Init Job**: Starts, waits for PostgreSQL via initContainer
- **RAG Service**: Starts, waits for PostgreSQL via initContainer
- **No Dependency**: Both can start simultaneously, no blocking

#### Phase 2: Data Preparation - Init Job (T+15s to T+65s)
- **Init Job Actions**:
  1. Checks if embeddings already exist (skips if found, unless `RAG_FORCE_REBUILD=true`)
  2. Reads PDFs from container image (`/app/data/knowledge_base`)
  3. Parses PDFs into chunks using `AnsibleErrorParser`
  4. Generates embeddings using embedding service (TEI)
  5. Saves embeddings to PostgreSQL `ragembedding` table
- **PostgreSQL**: Receives and stores embeddings
- **RAG Service**: Continues polling PostgreSQL (embeddings not found yet)

#### Phase 3: Index Loading - RAG Service (T+65s to T+75s)
- **RAG Service Actions**:
  1. Background task polls PostgreSQL every 5 seconds
  2. When embeddings found: queries all embeddings from `ragembedding` table
  3. Parses pgvector string format to numpy arrays
  4. Builds FAISS IndexFlatIP in memory
  5. Creates error store and index-to-error-id mapping
  6. Marks service as ready (`/ready` endpoint returns 200)
- **PostgreSQL**: Serves embedding queries
- **Init Job**: Continues waiting for RAG service

#### Phase 4: Coordination - Init Job Waits (T+75s to T+85s)
- **Init Job Actions**:
  1. After saving embeddings, calls `wait_for_rag_service()`
  2. Polls `http://alm-rag:8002/ready` endpoint every 5 seconds
  3. Timeout: 5 minutes (300 seconds)
  4. Once RAG service ready, proceeds to training pipeline
- **RAG Service**: Responds to `/ready` checks (returns 200 when ready)
- **If Timeout**: Init job continues with warning, RAG queries may fail

#### Phase 5: Runtime - Training Pipeline (T+85s+)
- **Init Job**: Runs `training_pipeline()` which:
  - Processes alerts
  - Uses RAG service for context retrieval (HTTP calls)
  - Saves results to database
- **RAG Service**: Serves queries via `/rag/query` endpoint
- **Backend Pods**: (After init job completes) Can query RAG service for context

### Communication Patterns

#### Init Job → PostgreSQL
- **Method**: Direct database writes via SQLModel
- **When**: During `build_rag_index()` function
- **What**: Inserts/updates `ragembedding` table
- **Frequency**: Once per init job run

#### RAG Service → PostgreSQL
- **Method**: Raw SQL queries via asyncpg
- **When**: Background polling task (every 5 seconds)
- **What**: SELECT queries from `ragembedding` table
- **Frequency**: Every 5 seconds until embeddings found, then once at startup

#### Init Job → RAG Service
- **Method**: HTTP GET requests
- **When**: After saving embeddings, before training pipeline
- **What**: Polls `/ready` endpoint
- **Frequency**: Every 5 seconds, timeout 5 minutes

#### Backend → RAG Service
- **Method**: HTTP POST requests
- **When**: During training pipeline and runtime queries
- **What**: `/rag/query` endpoint with query text
- **Frequency**: As needed for context retrieval

### Error Handling and Resilience

1. **RAG Service Startup Failure**:
   - Service starts but stays in "not ready" state
   - Background task continues polling
   - Service becomes ready when embeddings available
   - No crash, graceful degradation

2. **Init Job Failure**:
   - RAG service continues polling (will timeout after 10 minutes)
   - Can be restarted independently
   - No impact on RAG service pod

3. **Embeddings Not Found**:
   - RAG service logs warning, continues polling
   - Init job can be rerun to populate embeddings
   - No data loss (embeddings persist in PostgreSQL)

4. **RAG Service Not Ready**:
   - Init job waits up to 5 minutes
   - If timeout: continues with warning
   - Training pipeline proceeds, RAG queries may fail gracefully

### Why This Design?

1. **Eliminates Circular Dependencies**: 
   - Old design: RAG service waited for init job, init job needed RAG service → deadlock
   - New design: Both start independently, coordinate via PostgreSQL

2. **Faster Startup**:
   - Services don't block each other
   - Parallel execution possible
   - No sequential waiting

3. **Resilience**:
   - Services can restart independently
   - Data persists in PostgreSQL
   - Graceful degradation if one service fails

4. **Scalability**:
   - RAG service can scale independently
   - Multiple backend pods share single RAG service
   - No resource duplication

## Key Design Decisions

### 1. Non-Blocking Startup
- **Problem**: RAG service was crashing if embeddings weren't available immediately
- **Solution**: Background task loads index asynchronously, service starts immediately
- **Benefit**: No circular dependencies, service can start before init job completes

### 2. PostgreSQL as Coordination Point
- **Problem**: Need to coordinate between init job and RAG service
- **Solution**: PostgreSQL acts as shared data store, both services read/write independently
- **Benefit**: No direct dependencies, both services can start in parallel

### 3. Embedding Persistence
- **Problem**: Training pipeline was deleting `ragembedding` table
- **Solution**: Modified `init_tables()` to preserve `ragembedding` table when `delete_tables=True`
- **Benefit**: Embeddings persist across training pipeline runs

### 4. pgvector String Parsing
- **Problem**: pgvector returns embeddings as strings when queried via raw SQL
- **Solution**: Added parsing logic to handle both array and string representations
- **Benefit**: Robust handling of different PostgreSQL response formats

## Components

### 1. Database Schema
- **Table**: `ragembedding` (SQLModel)
- **Fields**:
  - `error_id` (primary key) - Unique identifier for each error
  - `embedding` (Vector(768)) - pgvector type, 768 dimensions for nomic-embed-text-v1.5
  - `error_title` - Title of the error
  - `error_metadata` (JSON) - Complete error metadata including sections
  - `model_name` - Embedding model used
  - `embedding_dim` - Dimension of embedding vector (768)
  - `created_at`, `updated_at` - Timestamps

### 2. RAG Service (`services/rag/`)
- **Technology**: FastAPI
- **Port**: 8002
- **Endpoints**:
  - `POST /rag/query` - Query knowledge base for relevant errors
  - `GET /health` - Health check (returns status even if index not loaded)
  - `GET /ready` - Readiness check (returns 503 until index loaded)
  - `POST /rag/reload` - Reload index from PostgreSQL without restart

### 3. Backend Changes
- **Removed**: FAISS loading, PVC mounting, local index management
- **Added**: HTTP client (`httpx.AsyncClient`) for RAG service communication
- **Interface**: Same API (no changes to calling code)
- **Cleanup**: Proper HTTP client shutdown on application shutdown

### 4. Init Job
- **Changed**: Saves embeddings to PostgreSQL instead of PVC
- **PDFs**: Read directly from container image (`/app/data/knowledge_base`)
- **Coordination**: Waits for RAG service to be ready before running training pipeline
- **Persistence**: Embeddings persist across training pipeline runs

## Migration Steps

### Step 1: Database Migration

The database schema is automatically created when `init_tables()` is called. The pgvector extension is enabled automatically:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Step 2: Build and Deploy RAG Service

1. **Build RAG service image** (from project root):
   ```bash
   podman build -f services/rag/Containerfile -t quay.io/rh-ai-quickstart/alm-rag:latest .
   podman push quay.io/rh-ai-quickstart/alm-rag:latest
   ```

2. **Deploy RAG service** (via Helm):
   ```bash
   helm upgrade --install ansible-log-monitor ./deploy/helm/ansible-log-monitor
   ```

### Step 3: Run Init Job

The init job will:
1. Read PDFs from container image
2. Generate embeddings
3. Save to PostgreSQL
4. Wait for RAG service to be ready
5. Run training pipeline

```bash
# Check init job status
oc get jobs -n <namespace> -l app.kubernetes.io/component=init

# View logs
oc logs -n <namespace> -l job-name=alm-backend-init --tail=100
```

### Step 4: Verify RAG Service

```bash
# Check service is running
oc get pods -n <namespace> -l app.kubernetes.io/name=rag

# Check service health
RAG_POD=$(oc get pods -n <namespace> -l app.kubernetes.io/name=rag -o jsonpath='{.items[0].metadata.name}')
oc exec -n <namespace> $RAG_POD -- curl -s http://localhost:8002/health | jq

# Check readiness (should return 200 when index is loaded)
oc exec -n <namespace> $RAG_POD -- curl -s http://localhost:8002/ready | jq
```

### Step 5: Test RAG Query

```bash
# Test query from within cluster
oc run -it --rm test-rag-query -n <namespace> --image=curlimages/curl --restart=Never -- \
  curl -X POST http://alm-rag:8002/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "ansible playbook execution failed",
    "top_k": 5,
    "top_n": 3,
    "similarity_threshold": 0.6
  }' | jq
```

## Configuration

### Environment Variables

#### RAG Service
- `DATABASE_URL` - PostgreSQL connection URL (required, from secret `pgvector`)
- `EMBEDDINGS_LLM_URL` - Embedding service URL (default: `http://alm-embedding:8080`)
- `RAG_MODEL_NAME` - Model name (default: `nomic-ai/nomic-embed-text-v1.5`)
- `PORT` - Service port (default: `8002`)

#### Backend
- `RAG_ENABLED` - Enable/disable RAG (default: `true`, accepts: `true`, `1`, `yes`)
- `RAG_SERVICE_URL` - RAG service URL (default: `http://alm-rag:8002`)
- `RAG_TOP_K` - Top K candidates to retrieve (default: `10`)
- `RAG_TOP_N` - Top N final results to return (default: `3`)
- `RAG_SIMILARITY_THRESHOLD` - Minimum similarity threshold (default: `0.6`)

### Helm Values

```yaml
rag:
  enabled: true
  serviceUrl: "http://alm-rag:8002"
  query:
    topK: 4
    topN: 1
    similarityThreshold: 0.6
```

## Testing

### 1. Unit Tests

Test the RAG service locally:

```bash
cd services/rag
# Set environment variables
export DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/dbname"
export EMBEDDINGS_LLM_URL="http://localhost:8080"

# Run service
uvicorn main:app --host 0.0.0.0 --port 8002
```

### 2. Integration Tests

Test backend → RAG service communication:

```python
# In backend pod or test environment
from alm.agents.get_more_context_agent.rag_handler import RAGHandler

handler = RAGHandler()
context = await handler.get_cheat_sheet_context("ansible error message")
print(context)
```

### 3. End-to-End Test

1. Deploy all components
2. Run init job
3. Verify RAG service loads index
4. Trigger an alert that requires RAG context
5. Verify RAG service is called and returns results

## Troubleshooting

### RAG Service Not Starting

**Problem**: Service fails to start or index doesn't load

**Check**:
```bash
# Check logs
oc logs -n <namespace> -l app.kubernetes.io/name=rag --tail=50

# Verify PostgreSQL connection
RAG_POD=$(oc get pods -n <namespace> -l app.kubernetes.io/name=rag -o jsonpath='{.items[0].metadata.name}')
oc exec -n <namespace> $RAG_POD -- env | grep DATABASE_URL

# Check if embeddings exist (replace <dbname> with actual database name)
PG_POD=$(oc get pods -n <namespace> -l app=postgresql -o jsonpath='{.items[0].metadata.name}')
oc exec -n <namespace> $PG_POD -- psql -U postgres -d <dbname> -c "SELECT COUNT(*) FROM ragembedding;"
```

### RAG Service Stuck in "Not Ready" State

**Problem**: Service starts but `/ready` endpoint returns 503

**Possible Causes**:
1. Embeddings not yet available (init job still running)
2. Database connection issue
3. Embedding parsing error

**Check**:
```bash
# Check RAG service logs for polling messages
oc logs -n <namespace> -l app.kubernetes.io/name=rag | grep -i "embedding"

# Verify embeddings exist in database
oc exec -n <namespace> $PG_POD -- psql -U postgres -d <dbname> -c "SELECT COUNT(*) FROM ragembedding;"

# Check init job status
oc get jobs -n <namespace> -l app.kubernetes.io/component=init
```

### Backend Can't Reach RAG Service

**Problem**: Backend returns empty context

**Check**:
```bash
# Verify service exists
oc get svc -n <namespace> alm-rag

# Test connectivity from backend pod
BACKEND_POD=$(oc get pods -n <namespace> -l app.kubernetes.io/name=backend -o jsonpath='{.items[0].metadata.name}')
oc exec -n <namespace> $BACKEND_POD -- curl -s http://alm-rag:8002/health

# Check backend logs
oc logs -n <namespace> $BACKEND_POD | grep -i rag
```

### No Embeddings in Database

**Problem**: Init job didn't populate embeddings

**Check**:
```bash
# Check init job logs
oc logs -n <namespace> -l job-name=alm-backend-init --tail=100

# Verify PDFs in image
INIT_POD=$(oc get pods -n <namespace> -l app.kubernetes.io/component=init -o jsonpath='{.items[0].metadata.name}')
oc exec -n <namespace> $INIT_POD -- ls -la /app/data/knowledge_base/

# Check database
oc exec -n <namespace> $PG_POD -- psql -U postgres -d <dbname> -c "SELECT error_id, model_name FROM ragembedding LIMIT 5;"
```

### Embeddings Deleted After Training Pipeline

**Problem**: Embeddings disappear after init job completes

**Solution**: This was fixed - `init_tables(delete_tables=True)` no longer deletes `ragembedding` table. If you see this issue, ensure you're using the latest backend image.

**Verify**:
```bash
# Check database.py has the fix
oc exec -n <namespace> $PG_POD -- psql -U postgres -d <dbname> -c "SELECT COUNT(*) FROM ragembedding;"
# Should return > 0 even after training pipeline runs
```

### Performance Issues

**Problem**: Slow query responses

**Solutions**:
- Increase RAG service resources (memory/CPU)
- Check PostgreSQL connection pool
- Verify FAISS index is loaded (check `/ready` endpoint)
- Consider adding RAG service replicas with load balancing

## Rollback Plan

If issues occur, you can rollback:

1. **Disable RAG service**:
   ```yaml
   rag:
     enabled: false
   ```

2. **Revert to PVC** (if needed):
   - Restore `rag-pvc.yaml` template
   - Update `init_pipeline.py` to save to disk
   - Update backend to load from PVC

3. **Database cleanup** (optional):
   ```sql
   DROP TABLE IF EXISTS ragembedding;
   ```

## Benefits Achieved

✅ **No RWO Constraints**: Backend pods can run on any node  
✅ **Reduced Memory**: Single FAISS index instead of N copies  
✅ **Simplified Storage**: Single source of truth (PostgreSQL)  
✅ **Easier Updates**: Update embeddings via SQL, no pod restarts  
✅ **Better Scaling**: Independent scaling of RAG vs backend  
✅ **No PVC Management**: Eliminated persistent volume complexity  
✅ **Resilient Startup**: No circular dependencies, graceful degradation  
✅ **Data Persistence**: Embeddings survive training pipeline runs  

## Files Changed

### New Files

#### `services/rag/main.py`
FastAPI application for the RAG service. Implements:
- Background task for loading index (non-blocking startup)
- HTTP endpoints for querying, health checks, and reloading
- Query processing: generates embeddings, searches FAISS, returns results
- Graceful error handling and service state management

#### `services/rag/index_loader.py`
Loads embeddings from PostgreSQL and builds FAISS index. Handles:
- PostgreSQL connection and querying
- Parsing pgvector string format to numpy arrays
- Building FAISS IndexFlatIP for similarity search
- Error store and index-to-error-id mapping

#### `services/rag/pyproject.toml`
Python dependencies for RAG service:
- FastAPI, uvicorn for web framework
- sqlmodel, asyncpg, psycopg2-binary for database access
- faiss-cpu, numpy for similarity search
- httpx for embedding service calls

#### `services/rag/Containerfile`
Container image definition for RAG service:
- Based on UBI8 Python 3.12
- Uses `uv` for dependency management
- Copies service code and dependencies
- Exposes port 8002

#### `deploy/helm/ansible-log-monitor/charts/rag/`
Complete Helm chart for deploying RAG service:
- Deployment with initContainer for PostgreSQL readiness
- Service for cluster-internal access
- ServiceAccount and RBAC (if needed)
- HPA for autoscaling (optional)
- ConfigMap and environment variable management

### Modified Files

#### `src/alm/models.py`
**Change**: Added `RAGEmbedding` SQLModel class
- Defines database schema for storing embeddings
- Uses `pgvector.sqlalchemy.Vector(768)` for embedding column
- Includes error metadata as JSON field
- Tracks model name and embedding dimensions

#### `src/alm/database.py`
**Changes**:
1. Added `RAGEmbedding` to table creation/dropping
2. Added automatic pgvector extension enablement
3. **Critical Fix**: Modified `init_tables()` to NOT delete `ragembedding` table when `delete_tables=True`
   - Prevents training pipeline from deleting embeddings
   - Ensures embeddings persist across runs

#### `src/alm/rag/embed_and_index.py`
**Changes**:
1. Added `_embeddings_array` attribute to store embeddings before FAISS
2. Added `save_to_postgresql()` method to persist embeddings
3. Added `ingest_and_index_to_postgresql()` async entry point
4. Modified `build_faiss_index()` to store embeddings array for PostgreSQL saving

#### `src/alm/agents/get_more_context_agent/rag_handler.py`
**Changes**:
1. Replaced local FAISS loading with HTTP client
2. Added `httpx.AsyncClient` for RAG service communication
3. Implemented lazy initialization of HTTP client
4. Added `cleanup()` method for graceful shutdown
5. Updated `_format_rag_results()` to parse JSON response from service

#### `src/alm/main_fastapi.py`
**Change**: Added shutdown event handler
- Calls `RAGHandler().cleanup()` on application shutdown
- Ensures HTTP client is properly closed
- Prevents resource leaks

#### `init_pipeline.py`
**Changes**:
1. Removed PVC-related logic (PDF copying, volume mounting)
2. Updated `build_rag_index()` to always use PostgreSQL
3. Added `wait_for_rag_service()` function to coordinate with RAG service
4. Updated main flow: build index → wait for RAG service → run training pipeline
5. Simplified data directory setup (PDFs now in container image)

#### `deploy/helm/ansible-log-monitor/charts/backend/templates/deployment.yaml`
**Changes**:
- Removed `volumeMounts` and `volumes` for `rag-data` PVC
- Backend no longer needs direct access to RAG storage

#### `deploy/helm/ansible-log-monitor/charts/backend/templates/init-job.yaml`
**Changes**:
- Removed `volumeMounts` and `volumes` for `rag-data` PVC
- Removed conditional PVC checks
- Always assumes PostgreSQL storage

#### `deploy/helm/ansible-log-monitor/charts/backend/templates/configmap.yaml`
**Changes**:
- Added `RAG_SERVICE_URL` environment variable
- Updated comments to reflect new architecture

#### `deploy/helm/ansible-log-monitor/charts/backend/values.yaml`
**Changes**:
- Removed `rag.persistence` section (no PVC needed)
- Added `rag.serviceUrl` configuration
- Updated `rag.knowledgeBaseDir` to reflect PDFs in image

#### `deploy/helm/ansible-log-monitor/global-values.yaml`
**Change**: Added `rag: "alm-rag"` to `servicesNames` for service discovery

#### `pyproject.toml` (root)
**Change**: Added `pgvector>=0.2.5` dependency
- Required for `Vector` type in `RAGEmbedding` model
- Needed for backend to create tables with pgvector columns

### Deleted Files

#### `deploy/helm/ansible-log-monitor/charts/backend/templates/rag-pvc.yaml`
**Reason**: No longer needed - RAG data stored in PostgreSQL, not PVC

## Key Fixes Applied

### 1. Circular Dependency Resolution
- **Problem**: RAG service waited for init job, init job needed RAG service
- **Solution**: 
  - RAG service starts independently, polls PostgreSQL for embeddings
  - Init job waits for RAG service after building index
  - Both can start in parallel, coordinate via PostgreSQL

### 2. Non-Blocking Startup
- **Problem**: RAG service crashed if embeddings not available immediately
- **Solution**: Background task loads index asynchronously, service starts immediately
- **Result**: Service stays in "not ready" state until embeddings available

### 3. Embedding Persistence
- **Problem**: Training pipeline deleted `ragembedding` table
- **Solution**: Modified `init_tables()` to preserve `ragembedding` when `delete_tables=True`
- **Result**: Embeddings persist across training pipeline runs

### 4. pgvector String Parsing
- **Problem**: pgvector returns embeddings as strings in raw SQL queries
- **Solution**: Added parsing logic using JSON and `ast.literal_eval()`
- **Result**: Handles both array and string representations

## Next Steps

1. **Deploy RAG service** to your cluster
2. **Run init job** to populate embeddings
3. **Monitor** RAG service health and performance
4. **Test** end-to-end RAG queries
5. **Optimize** resource allocation based on usage

## Support

For issues or questions:
- Check service logs: `oc logs -n <namespace> -l app.kubernetes.io/name=rag`
- Check backend logs: `oc logs -n <namespace> -l app.kubernetes.io/name=backend`
- Verify database: Check `ragembedding` table in PostgreSQL
- Check init job: `oc logs -n <namespace> -l job-name=alm-backend-init`
