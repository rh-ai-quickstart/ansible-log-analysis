# Ansible Log Analysis Quick Start

An AI agent for AAP clusters that detects Ansible log errors, suggests step-by-step fixes using cluster-wide logs, and routes issues to the right experts.

## Table of contents

1. [Overview](#overview)
2. [Problem We Solve](#problem-we-solve)
3. [Current Manual Process](#current-manual-process)
4. [Our Solution Stack](#our-solution-stack)
5. [High-Level Solution](#high-level-solution)
6. [Agentic Workflow](#agentic-workflow)
   - [Step 1: Embedding and Clustering](#step-1-embedding-and-clustering)
   - [Step 2: Summary and Expert Classification per Log Template](#step-2-summary-and-expert-classification-per-log-template)
   - [Step 3: Creating a step-by-step solution](#step-3-creating-a-step-by-step-solution)
   - [Step 4: Store the data](#step-4-store-the-data)
   - [Training and Inference stages](#training-and-inference-stages)
7. [User Interface](#user-interface)
8. [Annotation Interface](#annotation-interface)
9. [Requirements](#requirements)
   - [Software Requirements](#software-requirements)
   - [Minimum Hardware Requirements](#minimum-hardware-requirements)
10. [Deployment](#deployment)
    - [Quick Start - Local Development](#quick-start---local-development)
    - [Deploy on the Cluster](#deploy-on-the-cluster)
11. [Developer Workflow & CI/CD](#developer-workflow--cicd)
    - [One-Time Setup](#one-time-setup)
    - [How the CI/CD Flow Works](#how-the-cicd-flow-works)

## Detailed description

**The Challenge:** Organizations running Ansible automation at scale face significant challenges when errors occur. Log analysis is manual, time-consuming, and requires specialized knowledge across multiple domains (AWS, Kubernetes, networking, etc.). When failures happen, teams spend valuable time searching through logs, identifying the right experts, and waiting for solutions.

**Our Solution:** An AI-powered log analysis system that automatically:
- Detects and categorizes Ansible errors in real-time
- Routes issues to appropriate experts based on authorization levels
- Provides contextual, step-by-step solutions using AI agents
- Learns from historical resolutions to improve future recommendations

### Current Manual Process

A human analyst is:

* Searching for error logs.  
* Talk with the person who is authorized with the credentials to solve the problem:  
  * Examples:   
    AWS provisioning failed requires talking with the AWS person who is authorized.  
    Bug in the playbook source code \- talk with the programmer.  
* The authenticated person needs to **understand how to solve the problem**.  
* Solve the problem.

## Our Solution Stack

* Loki \- as a log database.  
* Alloy/Promtail \- log ingestion and label definer.  
* OpenShiftAI \- model serving, data science pipeline, notebooks.  
* Backend:  
  * FASTAPI \- for api endpoints.  
  * Langchain.  
  * LangGraph \- for building the agentic workflow.  
  * PostgreSQL.  
  * Sentence Transformers \- for clustering embeddings (log template grouping).  
  * TEI (text-embeddings-inference) \- for RAG embeddings (knowledge base retrieval).  
* UI:  
  * Gradio (for now)  
* Annotation interface: an interface that is used for evaluation and workflow improvement  
  * Gradio

## High-Level Solution

1. Data is being **ingested** from the Red Hat Ansible Automation Platform (AAP) clusters, using Alloy or Promtail, into Loki (a time series database designed for logs).  
2. An **error log is alerted** using a Grafana alert and sent into the agentic workflow.  
3. The **agentic workflow** processes the log and stores the processed data into a PostgreSQL database.  
4. The log analyst using the **UI** interacts with the logs and gets suggestions on how to solve the error, depending on their authorization. 

<img src="figures/high_level_architecture.png" alt="high_level_architecture" style="width:65%;">

## Agentic Workflow:

<img src="figures/workflow.png" alt="Workflow" style="width:65%;">

### Step 1: Embedding and Clustering

Many logs are generated from the same log template. To group them, we embed a subset of each log, then cluster all the embeddings into groups. Each group represents a log template. For example, let’s look at the following three logs:

```
1. error: user id 10 already exits.
2. error: user id 15 already exits.
3. error: password of user itayk is wrong.
```

As we can see here, logs 1 and 2 are from the same template, and we want to group them together.

Then the user will be able to filter by templates.

### Step 2: Summary and Expert Classification per Log Template

For each log template, create a summary of the log and classify it by authorization.  
For example, an analyst who has AWS authentication will filter by their authorization and will see only relevant error summaries in the UI.

### Step 3: Creating a step-by-step solution 

We will have a router that will determine if we need more context to solve the problem or if the log error alone is sufficient to generate the step-by-step solution.  
If we need more context, we will spin up an agent that will accumulate context as needed by using the following:

* **Loki MCP**, which is able to query the log database for additional log context. This project uses an [enhanced fork](https://github.com/RHEcosystemAppEng/loki-mcp/tree/downstream/loki-mcp-for-alm) with additional query capabilities.
  
* **RAG** for retrieving an error cheat sheet of already solved questions.  
* **Ansible MCP** for obtaining code source data to suggest a better solution.

### Step 4: Store the data

* Store a payload of the generated values for each log in a PostgreSQL database.

### Training and Inference stages

Currently, the **only difference** between the training and inference stages is the clustering algorithm.

#### Training

Train the clustering algorithm to cluster the logs by log-template.

#### Inference 

Load the trained clustering model.

## User Interface

* Each expert selects their rule, dependent on their authorization. Current rules are:  
  * Kubernetes / OpenShift Cluster Admins  
  * DevOps / CI/CD Engineers (Ansible \+ Automation Platform)  
  * Networking / Security Engineers  
  * System Administrators / OS Engineers  
  * Application Developers / GitOps / Platform Engineers  
  * Identity & Access Management (IAM) Engineers  
  * Other / Miscellaneous  
* Each expert can filter by labels (cluster\_name, log\_file\_name, …)  
* A summary of each log is listed to the expert, the expert can click on the log summary and view the whole log, and a step-by-step solution, timestamp, and labels

<img src="figures/experts_option.png" alt="Experts Option" style="width:40%;">

After selecting the authorization class "expert":

<img src="figures/ui_view.png" alt="UI View" style="width:65%;">

<img src="figures/step-by-step.png" alt="Step-by-step Solution" style="width:65%;">

## Annotation Interface

For improving our agentic workflow, context PDFs, and other context we need to understand the errors. To do so, we have a data annotation interface for annotating Ansible error log pipeline outputs,  
Where we see the agentic workflow:

* **Input** of the left (error log)  
* **Outputs** in the center (summary, and step-by-step solution)  
* **Annotation window** on the right.

See the interface below:

<img src="figures/anotation_interface.png" alt="Annotation Interface" style="width:65%;">

## Requirements

### Software Requirements

#### For Production Cluster Deployment
- **OpenShift Cluster** <TODO add version>
- **Helm** <TODO add version>
- **oc CLI** (for OpenShift)


### Minimum Hardware Requirements

Storage / PC of size <TODO>


#### Production Cluster Environment

<TODO>


#### Scalability Considerations

<TODO>
- **GPU** for faster embedding.


## Deployment

The Ansible Log Monitor can be deployed in multiple environments depending on your needs. Choose the deployment method that best fits your requirements:

### Quick Start - Local Development

For development and testing, you can run all services locally using the provided Makefile:

#### Mock Data (Temporary for Development)

To use add data during development, add your log files to the `data/logs/failed` directory. 

Each log should be saved as a separate `.txt` file (e.g., `<filename>.txt`).
For example `data/logs/failed/example.txt`

#### Prerequisites
- Docker and Docker Compose
- `uv 0.9.x` package manager with Python 3.12+
- Make (for running deployment commands)
- Make sure you have added the mock data as described in the [### Mock Data (Temporary for Development)](#mock-data-temporary-for-development) section.

#### Deploy Locally

Follow these steps to set up and run the Ansible Log Monitor on your local development environment:

**1. Clone and Setup Repository**
```bash
# Clone the repository
git clone <repository-url>
cd ansible-logs

# Install dependencies and alm package (install alm in editable mode by default)
uv sync
```

**2. Configure Environment Variables**
```bash
# Copy the environment template and configure your settings
cp .env.example .env

# Edit .env with your API keys and configuration:
# - OPENAI_API_ENDPOINT: VLLM (OpenAI) compitable endpoint (some endpoint need to add /v1 as suffix)
# - OPENAI_API_TOKEN: your token to the endpoint
# - OPENAI_MODEL: Model to use (e.g., llama-4-scout-17b-16e-w4a16	)
# - LANGSMITH_API_KEY: Optional, for LangSmith tracing
# Configure RAG
# Model is hardcoded to nomic-ai/nomic-embed-text-v1.5 (no config needed)
# Optional: Override TEI service URL (defaults to http://alm-embedding:8080)
# - EMBEDDINGS_LLM_URL=http://alm-embedding:8080
# Optional: Customize query parameters
# - RAG_TOP_K=10 (the number of candidates retrieved from the FAISS index)
# - RAG_TOP_N=3 (the number of final results returned after filtering by the similarity threshold)
# - RAG_SIMILARITY_THRESHOLD=0.6
```

**3. Start All Services**
In short:
```bash
make local/install
make local/train
```

```bash
# Launch all services in the background
make local/install

# Run the complete training pipeline (do it after local/install)
make local/train

# Perform status check to see which services are running
make local/status

# uninstall all services when done
make local/uninstall
```

**Additional Commands**
```bash
# Restart all services
make local/restart

# View all available local commands
make local/help
```

### Deploy on the Cluster

For production deployment on OpenShift clusters:

#### Prerequisites
- OpenShift CLI (`oc`) installed and configured
- Helm 3.x installed
- Access to an OpenShift cluster
- MaaS API Token and endpoint, or OpenAI token and endpoint (for LLM only; embeddings use TEI service)

#### Quick Deployment
```bash
# Install the application (uses current OpenShift project)
make cluster/install

# With custom namespace
make cluster/install NAMESPACE=ansible-logs-monitor
```
You will be prompted to add your API token, endpoint, model, and temperature.
- API token is required; the others have default values that you can leave empty.

#### Access Services
```bash
# Forward UI to localhost:7860
make cluster/port-forward-ui

# Forward Backend API to localhost:8000
make cluster/port-forward-backend

# Forward Annotation Interface to localhost:7861
make cluster/port-forward-annotation

# Forward Grafana to localhost:3000
make cluster/port-forward-grafana
```

#### Uninstall
```bash
# Remove from current project
make cluster/uninstall

# Remove from specific namespace
make cluster/uninstall NAMESPACE=ansible-logs-monitor
```

#### Addional commands
```bash
# upgrade
make cluster/upgrade

# restart
make cluster/restart
```

For detailed configuration options and troubleshooting, see [deploy/helm/README.md](deploy/helm/README.md).

## Developer Workflow & CI/CD

Automated CI/CD workflow for building and publishing container images to Quay.io.

### One-Time Setup

```bash
pre-commit install    # Install git hooks
podman login quay.io  # Authenticate to Quay
```

**Prerequisites:** Ensure `data/logs/failed/` and `data/knowledge_base/` directories exist with required files.

### How the CI/CD Flow Works

**On every commit** (`git commit`):
- Ruff linter checks code style
- Ruff formatter auto-formats code
- Pre-push hook synced from `.githooks/pre-push` to `.git/hooks/pre-push`

**On every push** (`git push`):
- Backend image builds automatically with branch name as tag
- Image pushed to `quay.io/rh-ai-quickstart/alm-backend:<branch-name>`
- Branch names sanitized: `feature/add-logging` → `feature-add-logging`

**When PR merged to main**:
- GitHub Actions builds: `alm-ui`, `alm-annotation-interface`, `alm-clustering`
- GitHub Actions tags the backend image built from the merged PR's branch as `latest`
