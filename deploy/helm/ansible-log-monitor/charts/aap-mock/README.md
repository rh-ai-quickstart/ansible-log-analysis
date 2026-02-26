# AAP Mock - Ansible Automation Platform Log Generator

This Helm chart deploys the AAP Log Generator (aap-mock), a mock Ansible Automation Platform that generates realistic AAP logs for testing and demonstration purposes.

> **📚 Complete Documentation:**  
> For detailed information about the AAP Log Generator application, API endpoints, features, and usage examples, see the upstream repository:  
> **https://github.com/RHEcosystemAppEng/aap-log-generator**

## Overview

The `aap-mock` service provides:
- **Mock AAP Logs**: Generates realistic Ansible Automation Platform logs
- **API Interface**: RESTful API for controlling log generation and replay
- **Sample Data Management**: Auto-loads sample log files for replay
- **Configurable Replay**: Control replay rate, looping, and filtering
- **Health Endpoints**: Liveness and readiness probes for Kubernetes

## Prerequisites

- Kubernetes 1.19+ or OpenShift 4.x+
- Helm 3.x
- Persistent storage provider (for PVCs)

## Installation

### As Part of Ansible Log Monitor Stack

When deploying the full `ansible-log-monitor` parent chart, `aap-mock` is included by default:

```bash
# Install the full stack (includes aap-mock)
helm install alm ../.. -n alm-infra --create-namespace

# Install with aap-mock disabled
helm install alm ../.. -n alm-infra --set aap-mock.enabled=false
```

### Standalone Installation

To install only the `aap-mock` chart:

```bash
# Install from this directory
helm install aap-mock . -n my-namespace --create-namespace
```

## Configuration

### Key Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `enabled` | Enable/disable aap-mock deployment | `true` |
| `replicaCount` | Number of replicas | `1` |
| `image.repository` | Container image repository | `quay.io/rh-ai-quickstart/alm-aap-mock` |
| `image.tag` | Container image tag | `latest` |
| `persistence.data.enabled` | Enable data PVC for sample logs | `true` |
| `persistence.data.size` | Data PVC size | `2Gi` |
| `persistence.logs.enabled` | Enable logs PVC | `true` |
| `persistence.logs.size` | Logs PVC size | `1Gi` |
| `route.enabled` | Create OpenShift route | `true` |
| `app.env` | Environment variables | `{}` |

### Example: Custom Configuration

Create a `custom-values.yaml`:

```yaml
aap-mock:
  enabled: true
  
  # Increase storage
  persistence:
    data:
      size: 5Gi
    logs:
      size: 2Gi
  
  # Configure automatic log replay on startup
  app:
    env:
      REPLAY_RATE: "50"
      LOOP_ENABLED: "true"
  
  # Adjust resources
  resources:
    limits:
      cpu: 1000m
      memory: 1Gi
    requests:
      cpu: 200m
      memory: 256Mi
```

Apply the configuration:

```bash
helm install alm ../.. -n alm-infra -f custom-values.yaml
```

## Usage

### Quick Start

```bash
# Get pod name
POD=$(kubectl get pods -l app.kubernetes.io/name=aap-mock -n alm-infra -o jsonpath='{.items[0].metadata.name}')

# Copy sample logs (if needed)
kubectl cp ./sample-logs/ $POD:/app/sample-logs/ -n alm-infra

# Refresh to detect new files
kubectl exec -n alm-infra $POD -- curl -X POST "http://localhost:8080/api/auto-loaded/refresh"

# Start log replay (100 logs/sec, loop indefinitely)
kubectl exec -n alm-infra $POD -- curl -X POST \
  "http://localhost:8080/api/logs/replay/all?rate=100&loop=true"

# Check status
kubectl exec -n alm-infra $POD -- curl -s "http://localhost:8080/api/status" | jq
```

> **📖 For complete API documentation and advanced usage, see:**  
> https://github.com/RHEcosystemAppEng/aap-log-generator#api-endpoints

### Integration with Loki

The `aap-mock` service is designed to work seamlessly with the Ansible Log Monitor stack:

1. **Log Generation**: `aap-mock` outputs logs to stdout
2. **Collection**: Alloy/Promtail collects logs from pod stdout
3. **Storage**: Logs are sent to Loki
4. **Processing**: Backend processes logs via the agentic workflow
5. **Visualization**: Grafana and UI display processed logs

## Upgrading

```bash
# Upgrade the full stack (includes aap-mock)
helm upgrade alm ../.. -n alm-infra

# Upgrade with new values
helm upgrade alm ../.. -n alm-infra -f custom-values.yaml
```

## Uninstalling

```bash
# Uninstall full stack (includes aap-mock)
helm uninstall alm -n alm-infra

# Note: PVCs are not automatically deleted
# Delete PVCs manually if needed
kubectl delete pvc -l app.kubernetes.io/name=aap-mock -n alm-infra
```

## Troubleshooting

### Pod Not Starting

```bash
# Check pod status
kubectl get pods -l app.kubernetes.io/name=aap-mock -n alm-infra

# View logs
kubectl logs -l app.kubernetes.io/name=aap-mock -n alm-infra

# Describe pod for events
kubectl describe pod -l app.kubernetes.io/name=aap-mock -n alm-infra
```

### Storage Issues

```bash
# Check PVC status
kubectl get pvc -l app.kubernetes.io/name=aap-mock -n alm-infra

# Check disk usage in pod
kubectl exec -n alm-infra $POD -- df -h

# Clean up logs if volume is full
kubectl exec -n alm-infra $POD -- truncate -s 0 /var/log/aap-mock/output.log
```

## Additional Resources

- **Source Code**: https://github.com/RHEcosystemAppEng/aap-log-generator
- **Container Images**: https://quay.io/repository/rh-ai-quickstart/alm-aap-mock
- **API Documentation**: https://github.com/RHEcosystemAppEng/aap-log-generator#api-endpoints
- **Issues & Support**: https://github.com/RHEcosystemAppEng/aap-log-generator/issues

