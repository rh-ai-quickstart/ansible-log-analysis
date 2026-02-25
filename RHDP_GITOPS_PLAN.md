# RHDP GitOps Compatibility Plan

## What is this and why are we doing it?

The RHDP (Red Hat Demo Platform) team wants to add Ansible Log Monitor as a showcase demo. RHDP provisions demos for customers and partners across Red Hat — getting listed there significantly increases the project's visibility.

The catch: RHDP deploys everything through ArgoCD GitOps. That means a pure `helm install` with a `values.yaml` file must bring the entire application up end-to-end. There can be no Makefile wrapper, no interactive prompts, and no manual `oc` commands run outside the chart. RHDP will pre-install any required operators (like the GPU operator if needed), manage secrets through their own pipeline, and inject configuration values at deploy time.

Our chart is ~90% GitOps-ready today. This plan addresses the remaining gaps. **Nothing here changes the existing Makefile-driven workflow** — all new behavior is activated only when a deployer explicitly opts in via values overrides.

---

## Prerequisite Operators / Cluster Requirements

RHDP installs prerequisite operators before deploying the chart. Here is what's needed:

| Requirement | Required? | Notes |
|-------------|-----------|-------|
| LLM Inference Endpoint | Yes | RHDP MaaS (Model as a Service) provides this; endpoint/token passed via values |
| GPU Operator | ? | TEI embedding runs on CPU - slow but functional - GPU may enhance training |
| Additional Operators | No | Chart is self-contained — deploys its own Loki, Grafana, MinIO, PostgreSQL |
| Minimum Resources | Yes | ~10Gi RAM, 3-4 CPU cores for the namespace; storage amount dependant on log volume |

---

## Change Summary (Priority Order)

| # | Change | Priority | Effort | Impact | Description |
|---|--------|----------|--------|--------|-------------|
| 1 | Add `model-secret` Helm template | **Required** (Blocker) | Small | High | Without this, ArgoCD cannot deploy — the secret is created outside Helm today |
| 2 | Make secret name configurable in sub-charts | **Required** | Small | Medium | Sub-charts hardcode `model-secret`; should reference a global value for flexibility |
| 3 | Fix Job immutability on `helm upgrade` | **Required** | Small | High | Init Jobs fail on upgrade/re-sync because K8s Jobs are immutable after creation |
| 4 | Document RHDP prerequisites and values | **Required** | Small | High | RHDP team needs to know what to pre-install and which values to override |
| 5 | Pin `origin-cli` image tags | Nice-to-have | Minimal | Low | Inconsistent tags (`:latest` vs `:4.15`) hurt reproducibility |
| 6 | Provide example RHDP values overlay | Nice-to-have | Minimal | Medium | A ready-to-use `rhdp-values-example.yaml` accelerates RHDP onboarding |
| 7 | Make `origin-cli` tag configurable | Optional | Minimal | Low | Lets RHDP or users control which OCP CLI version is used |

---

## Change Details (Expand each for more info)

<details>

<summary>Change 1: Create `model-secret` Helm Template (BLOCKER)</summary>
<br/>

**Problem:** The `model-secret` (containing `OPENAI_API_TOKEN`, `OPENAI_API_ENDPOINT`, `OPENAI_MODEL`, `OPENAI_TEMPERATURE`) is created outside Helm by the Makefile at `deploy/helm/Makefile:31-86` via interactive `oc create secret generic`. ArgoCD cannot run this.

**Three templates depend on this secret:**
- `charts/backend/templates/deployment.yaml:81`
- `charts/backend/templates/init-job.yaml:81`
- `charts/annotation-interface/templates/deployment.yaml:94`

**Approach:** Add an optional Helm-managed secret controlled by `modelSecret.create`. When `false` (default), the chart expects the secret to already exist (current Makefile behavior). When `true`, the chart creates it from values (GitOps mode).

**File to create:** `deploy/helm/ansible-log-monitor/templates/model-secret.yaml`
```yaml
{{- if .Values.modelSecret.create }}
apiVersion: v1
kind: Secret
metadata:
  name: {{ .Values.modelSecret.name | default "model-secret" }}
  labels:
    app.kubernetes.io/managed-by: {{ .Release.Service }}
type: Opaque
stringData:
  OPENAI_API_TOKEN: {{ .Values.modelSecret.apiToken | quote }}
  OPENAI_API_ENDPOINT: {{ .Values.modelSecret.apiEndpoint | quote }}
  OPENAI_MODEL: {{ .Values.modelSecret.model | quote }}
  OPENAI_TEMPERATURE: {{ .Values.modelSecret.temperature | default "0.7" | quote }}
{{- end }}
```

**File to modify:** `deploy/helm/ansible-log-monitor/values.yaml` — add block:
```yaml
modelSecret:
  create: false          # Set to true for GitOps/RHDP (chart creates the secret)
  name: "model-secret"   # Name of the K8s secret (used whether chart-managed or external)
  apiToken: ""           # LLM API token
  apiEndpoint: ""        # LLM API endpoint (include /v1 suffix)
  model: ""              # LLM model name
  temperature: "0.7"     # LLM temperature
```

Default `create: false` preserves existing Makefile workflow. RHDP sets `create: true`.
</details>

<details>

<summary>Change 2: Make Secret Name Configurable in Sub-Charts</summary>
<br/>

**Problem:** Sub-charts hardcode `name: model-secret` in their `secretRef`. If the secret name ever changes, or RHDP uses a different naming convention, these break.

**Approach:** Pass the secret name via `global` values and reference it in sub-chart templates.

**File to modify:** `deploy/helm/ansible-log-monitor/global-values.yaml`
```yaml
global:
  secrets:
    modelSecretName: "model-secret"
```

**Files to modify** (replace hardcoded `model-secret` with template reference):
- `charts/backend/templates/deployment.yaml:81` — `name: {{ .Values.global.secrets.modelSecretName | default "model-secret" }}`
- `charts/backend/templates/init-job.yaml:81` — same pattern
- `charts/annotation-interface/templates/deployment.yaml:94` — same pattern

</details>
<details>

<summary>Change 3: Fix Job Immutability on `helm upgrade`</summary>
<br/>

**Problem:** Two `batch/v1 Job` resources are created during install:
- `charts/backend/templates/init-job.yaml` — trains clustering model, initializes DB
- `charts/rag/templates/rag-init-job.yaml` — builds FAISS index from PDFs

Kubernetes Jobs are immutable after creation. On `helm upgrade`, if any spec value changes, Helm/ArgoCD will error. Even if specs haven't changed, completed Jobs block re-creation.

**Approach:** Add `ttlSecondsAfterFinished` so completed Jobs auto-clean up, and add a config checksum annotation so Helm detects meaningful changes. This is the one change that applies unconditionally (not behind a flag), but it is safe for all users — Jobs still run to completion normally, they just clean themselves up after 5 minutes instead of lingering in the namespace.

**Files to modify:**

`deploy/helm/ansible-log-monitor/charts/backend/templates/init-job.yaml`:
- Add under `spec:`:
  ```yaml
  ttlSecondsAfterFinished: 300
  ```
- Add annotation to `spec.template.metadata.annotations`:
  ```yaml
  checksum/config: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
  ```

`deploy/helm/ansible-log-monitor/charts/rag/templates/rag-init-job.yaml`:
- Add under `spec:`:
  ```yaml
  ttlSecondsAfterFinished: 300
  ```
- Add annotation with checksum of env values

`ttlSecondsAfterFinished` is GA since Kubernetes 1.23 — no operator or feature gate needed.

</details>
<details>

<summary>Change 4: Document RHDP Prerequisites and Values</summary>
<br/>

**Problem:** RHDP team needs clear documentation of what's required.

**Approach:** Include in the final plan document (this document) and optionally in the project README.

Content to document:
1. Prerequisites table (see above)
2. Which `values.yaml` keys RHDP must override
3. Example deployment command
4. What the Makefile does vs. what ArgoCD replaces

</details>
<details>

<summary>Change 5: Pin `origin-cli` Image Tags</summary>
<br/>

**Problem:** initContainers pulling `quay.io/openshift/origin-cli` use inconsistent tags:

| File | Current Tag |
|------|-------------|
| `charts/backend/templates/deployment.yaml:52` | `:4.15` |
| `charts/backend/templates/init-job.yaml:47` | `:4.15` |
| `charts/clustering/templates/deployment.yaml:33` | `:latest` |
| `charts/annotation-interface/templates/deployment.yaml:37` | `:latest` |

**Approach:** Pin all to `:4.15` for consistency and reproducibility.

**Files to modify:**
- `charts/clustering/templates/deployment.yaml:33` — change `:latest` to `:4.15`
- `charts/annotation-interface/templates/deployment.yaml:37` — change `:latest` to `:4.15`

</details>
<details>

<summary>Change 6: Provide Example RHDP Values Overlay</summary>
<br/>

**Approach:** Create an example values file for RHDP.

**File to create:** `deploy/helm/ansible-log-monitor/rhdp-values-example.yaml`
```yaml
# RHDP Values Overlay for Ansible Log Monitor
# Pass this file to helm install: -f rhdp-values-example.yaml

# Enable chart-managed model secret (replaces Makefile secret creation)
modelSecret:
  create: true
  apiToken: "<RHDP-MANAGED-TOKEN>"
  apiEndpoint: "https://<RHDP-MAAS-ENDPOINT>/v1"
  model: "<MODEL-NAME>"
  temperature: "0.7"

# Override default credentials (optional — defaults work for demos)
# grafana:
#   adminPassword: "<RHDP-MANAGED>"
# minio:
#   secret:
#     password: "<RHDP-MANAGED>"
```

</details>
<details>

<summary>Change 7: Make `origin-cli` Tag Configurable (Optional)</summary>
<br/>

**Problem:** If RHDP runs a different OCP version, they may need a different CLI version.

**Approach:** Add a global value and reference it in all 4 initContainer templates.

**File to modify:** `deploy/helm/ansible-log-monitor/global-values.yaml`
```yaml
global:
  originCli:
    tag: "4.15"
```

**Templates to update** (all 4 initContainers):
```yaml
image: quay.io/openshift/origin-cli:{{ .Values.global.originCli.tag | default "4.15" }}
```
</details>


## Deployment Options

### Existing workflow (UNCHANGED!)

```bash
make cluster/install NAMESPACE=ansible-log-monitor
# Still prompts for credentials, creates secret via oc, runs helm install
# modelSecret.create defaults to false — chart expects the external secret
```

No values overrides needed. Everything works exactly as before.

### GitOps / RHDP workflow (NEW!)

RHDP creates an overrides file that flips the opt-in flags and supplies credentials. This is what activates the GitOps features — without this file, the chart behaves identically to the Makefile workflow.

**Example overrides file** (`rhdp-overrides.yaml`):
```yaml
# This is the file that activates GitOps mode.
# modelSecret.create: true tells the chart to create the K8s Secret itself,
# instead of expecting the Makefile to have created it externally.
modelSecret:
  create: true
  apiToken: "sk-actual-token-from-rhdp-vault"
  apiEndpoint: "https://maas.rhdp.example.com/v1"
  model: "gpt-4"
  temperature: "0.7"
```

**Deploy command:**
```bash
helm install alm ./deploy/helm/ansible-log-monitor \
  -f deploy/helm/ansible-log-monitor/global-values.yaml \
  -f deploy/helm/ansible-log-monitor/values.yaml \
  -f rhdp-overrides.yaml \
  -n ansible-log-monitor
```

Or via an ArgoCD Application manifest pointing to the chart with the RHDP values overlay. The overrides file is the only difference between the two workflows — it tells Helm "create the secret for me" instead of "assume someone already created it."

---

## How Opt-In Works

Every change in this plan either (a) adds new values that default to current behavior, or (b) is unconditionally safe for all users:

| Value | Default | Effect at default | What RHDP overrides it to |
|-------|---------|-------------------|---------------------------|
| `modelSecret.create` | `false` | Chart does NOT create the model secret — expects the Makefile (or user) to have created it externally, exactly as today | `true` — chart creates the secret from values, no Makefile needed |
| `modelSecret.name` | `"model-secret"` | Same name used by hardcoded references today | Can be changed if RHDP needs a different naming convention |
| `global.secrets.modelSecretName` | `"model-secret"` | Templates resolve to the same hardcoded name they use today | Same as `modelSecret.name` — keeps sub-chart references in sync |
| `global.originCli.tag` | `"4.15"` | Same tag already hardcoded in backend templates | Can be changed if RHDP runs a different OCP version |
| `ttlSecondsAfterFinished` (on Jobs) | `300` | **Always active** — completed Jobs auto-delete after 5 minutes instead of lingering forever. This is the one non-opt-in change, but it's benign: it prevents `helm upgrade` failures without affecting Job execution | N/A — same for everyone |

**In practice:** If you run `make cluster/install` today and change nothing, the chart behaves identically. The opt-in flags are only activated when a deployer passes an overrides file (like shown above) that explicitly sets `modelSecret.create: true` and supplies the LLM credentials.

---

## Change Verifications

1. **Backward compatibility:** Run existing `make cluster/install` — should work identically (`modelSecret.create` defaults to `false`)
2. **GitOps mode:** `helm template --set modelSecret.create=true,modelSecret.apiToken=test,modelSecret.apiEndpoint=http://test/v1,modelSecret.model=test` — verify `model-secret` Secret appears in rendered output
3. **Upgrade safety:** Run `helm template` twice with different config values — verify Job annotations change (checksum diff)
4. **Lint:** `helm lint deploy/helm/ansible-log-monitor/ -f global-values.yaml -f values.yaml`
