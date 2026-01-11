# Ansible Logs Viewer (Gradio UI)

A modern Gradio UI to browse and analyze Grafana alerts by category. This UI connects to a FastAPI backend to display log alerts in an organized, filterable table with detailed view capabilities.

## Features

- **Category-based filtering**: Select from predefined log categories
- **Sortable table view**: Alerts sorted by timestamp (newest first)
- **Detailed log view**: Click any alert to see full log message and metadata
- **Real-time data**: Fetches fresh data from the backend API

## Log Categories
- GPU Autoscaling & Node Management Issues
- Cert-Manager & Certificate Creation Issues
- KubeVirt VM Provisioning & PVC Issues
- Vault Pod & Secret Storage Issues

## Backend Integration

The UI connects to a FastAPI backend with the following endpoints:
- `GET /grafana-alert` - Fetches all alerts
- `GET /by-category/{category}` - Fetches alerts filtered by category

### Configuration
- Configure backend base URL via environment variable `BACKEND_URL`
- Default is `http://localhost:8000`

## Installation & Setup

1) Install dependencies using uv (recommended):

```bash
uv sync
```

Or with pip:

```bash
pip install gradio httpx pandas
```

2) Set the backend URL (optional):

```bash
export BACKEND_URL=http://your-backend-url:8000
```

3) Start the UI:

```bash
python app.py
```

The UI will be available at `http://0.0.0.0:7860`.

## Usage

1. **Select Category**: Use the dropdown to select a log category
2. **Browse Alerts**: View alerts in the table, sorted by timestamp
3. **View Details**: Click on any alert row to see the full log message, classification, and labels
4. **Switch Categories**: Select different categories to filter alerts

## Data Structure

The UI expects alerts with the following structure:
- `logTimestamp`: Alert timestamp
- `logMessage`: Full log message
- `logSummary`: Brief summary of the alert
- `expertClassification`: Category classification
- `labels`: Key-value metadata labels

# Deployment of all apps

See [Local deployment](../../README.md#deploy-locally)