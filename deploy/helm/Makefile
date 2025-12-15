# Makefile for Ansible Log Monitor Deployment
# Replaces the original deploy.sh script with additional uninstall functionality

MAKEFLAGS += --no-print-directory

# Default values
NAMESPACE ?= $(shell oc project -q 2>/dev/null || echo "default")
# OPENAI_API_TOKEN = 
OPENAI_API_ENDPOINT ?= https://llama-4-scout-17b-16e-w4a16-maas-apicast-production.apps.prod.rhoai.rh-aiservices-bu.com:443/v1
OPENAI_MODEL ?= llama-4-scout-17b-16e-w4a16
OPENAI_TEMPERATURE ?= 0.7
ANSIBLE_LOG_MONITOR_CHART := ansible-log-monitor

# Parameter validation functions
define check_cluster_login
	@if ! oc whoami > /dev/null 2>&1; then \
		echo "❌ Error: Not logged into OpenShift cluster"; \
		echo "   Run: oc login <cluster-url>"; \
		exit 1; \
	fi
endef

define check_namespace
	$(call check_cluster_login)
	@echo "Using namespace: $(NAMESPACE)"
endef

# Function to prompt for OpenAI credentials
define prompt_openai_credentials
	@bash -c '\
	DEFAULT_ENDPOINT="$(OPENAI_API_ENDPOINT)"; \
	DEFAULT_MODEL="$(OPENAI_MODEL)"; \
	DEFAULT_TEMP="$(OPENAI_TEMPERATURE)"; \
	NAMESPACE_VAR="$(NAMESPACE)"; \
	echo "============================================"; \
	echo "   LLM API Configuration Required"; \
	echo "============================================"; \
	echo ""; \
	if [ -z "$$OPENAI_API_TOKEN" ]; then \
		echo -n "Enter MaaS LLM API TOKEN: "; \
		read -s OPENAI_API_TOKEN_INPUT; \
		echo ""; \
		export OPENAI_API_TOKEN=$$OPENAI_API_TOKEN_INPUT; \
	else \
		echo "Using provided OPENAI_API_TOKEN"; \
	fi; \
	if [ -z "$$OPENAI_API_ENDPOINT" ]; then \
		echo -n "Enter MaaS LLM API ENDPOINT (include /v1 suffix) [default: $$DEFAULT_ENDPOINT]: "; \
		read OPENAI_API_ENDPOINT_INPUT; \
		if [ -z "$$OPENAI_API_ENDPOINT_INPUT" ]; then \
			export OPENAI_API_ENDPOINT="$$DEFAULT_ENDPOINT"; \
		else \
			export OPENAI_API_ENDPOINT=$$OPENAI_API_ENDPOINT_INPUT; \
		fi; \
	else \
		echo "Using provided OPENAI_API_ENDPOINT: $$OPENAI_API_ENDPOINT"; \
	fi; \
	if [ -z "$$OPENAI_MODEL" ]; then \
		echo -n "Enter MaaS LLM MODEL NAME [default: $$DEFAULT_MODEL]: "; \
		read OPENAI_MODEL_INPUT; \
		if [ -z "$$OPENAI_MODEL_INPUT" ]; then \
			export OPENAI_MODEL="$$DEFAULT_MODEL"; \
		else \
			export OPENAI_MODEL=$$OPENAI_MODEL_INPUT; \
		fi; \
	else \
		echo "Using provided OPENAI_MODEL: $$OPENAI_MODEL"; \
	fi; \
	if [ -z "$$OPENAI_TEMPERATURE" ]; then \
		echo -n "Enter LLM TEMPERATURE [default: $$DEFAULT_TEMP]: "; \
		read OPENAI_TEMPERATURE_INPUT; \
		if [ -z "$$OPENAI_TEMPERATURE_INPUT" ]; then \
			export OPENAI_TEMPERATURE="$$DEFAULT_TEMP"; \
		else \
			export OPENAI_TEMPERATURE=$$OPENAI_TEMPERATURE_INPUT; \
		fi; \
	else \
		echo "Using provided OPENAI_TEMPERATURE: $$OPENAI_TEMPERATURE"; \
	fi; \
	echo ""; \
	echo "Creating Kubernetes secret '\''model-secret'\''..."; \
	oc create secret generic model-secret \
		--from-literal=OPENAI_API_TOKEN=$$OPENAI_API_TOKEN \
		--from-literal=OPENAI_API_ENDPOINT=$$OPENAI_API_ENDPOINT \
		--from-literal=OPENAI_MODEL=$$OPENAI_MODEL \
		--from-literal=OPENAI_TEMPERATURE=$$OPENAI_TEMPERATURE \
		-n "$$NAMESPACE_VAR" --dry-run=client -o yaml | oc apply -f -; \
	echo "Secret '\''model-secret'\'' created successfully"; \
	echo ""'
endef

env_args = \
	-f ansible-log-monitor/global-values.yaml \
	-f ansible-log-monitor/values.yaml

# Default target
.PHONY: help
help:
	@echo "Ansible Log Monitor - OpenShift Deployment"
	@echo "=========================================="
	@echo ""
	@echo "Available Commands:"
	@echo "  install                    Deploy the application (prompts for OpenAI credentials)"
	@echo "  upgrade                    Upgrade the application (prompts for OpenAI credentials)"
	@echo "  uninstall                  Remove the application"
	@echo "  namespace                  Create/switch to namespace"
	@echo "  port-forward-ui           Forward UI service to localhost:7860"
	@echo "  port-forward-backend      Forward Backend service to localhost:8000"
	@echo "  port-forward-annotation   Forward Annotation Interface to localhost:7861"
	@echo "  port-forward-grafana      Forward Grafana to localhost:3000"
	@echo ""
	@echo "Usage Examples:"
	@echo "  make install                              # Will prompt for credentials"
	@echo "  make install NAMESPACE=ansible-logs      # Will prompt for credentials"
	@echo "  make upgrade                              # Will prompt for credentials"
	@echo "  make uninstall"
	@echo "  make port-forward-ui"
	@echo ""
	@echo "Configuration Variables:"
	@echo "  NAMESPACE                  OpenShift namespace (default: current project)"
	@echo "  OPENAI_API_TOKEN          MaaS API Token (will prompt if not provided)"
	@echo "  OPENAI_API_ENDPOINT       AI API endpoint (will prompt if not provided, defaults to llama-4-scout)"
	@echo "  OPENAI_MODEL              AI model to use (will prompt if not provided, default: llama-4-scout-17b-16e-w4a16)"
	@echo "  OPENAI_TEMPERATURE        AI temperature setting (will prompt if not provided, default: 0.7)"
	@echo ""
	@echo "Note: All commands use the current OpenShift project by default."
	@echo "      Use NAMESPACE=your-namespace to override."
	@echo "      The install/upgrade commands will create a 'model-secret' with OpenAI configuration."



# Create namespace/project and deploy (NAMESPACE optional, defaults to current project)
.PHONY: namespace
namespace:
	$(call check_namespace)
	@echo "Setting up project $(NAMESPACE)..."
	@oc new-project $(NAMESPACE) 2>/dev/null || oc project $(NAMESPACE) 2>/dev/null || true
	@oc label namespace $(NAMESPACE) app=ansible-log-monitor --overwrite
	@echo "Project $(NAMESPACE) ready"

# Install the application (NAMESPACE optional, defaults to current project)
.PHONY: install
install: namespace
	$(call prompt_openai_credentials)
	@echo "🚀 Installing Ansible Log Monitor in namespace $(NAMESPACE)..."
	@oc project $(NAMESPACE)
	@helm install alm ./ansible-log-monitor -n $(NAMESPACE) $(env_args)
	@echo "✅ Helm install complete"
	@echo ""
	@echo "⏳ Waiting for aap-mock to be ready..."
	@$(MAKE) -s wait-for-aap-mock
	@echo ""
	@echo "📂 Loading logs to aap-mock and starting replay..."
	@$(MAKE) -s load-logs
	@echo ""
	@echo "⏳ Waiting for alloy to be ready..."
	@$(MAKE) -s wait-for-alloy
	@echo ""
	@echo "📂 Loading logs to alloy..."
	@$(MAKE) -s load-logs-to-alloy
	@echo ""
	@echo "🎉 Installation complete! View logs in Grafana with query: {job=\"ansible_logs\"}"

# Uninstall the deployment and clean up (NAMESPACE optional, defaults to current project)
.PHONY: uninstall
uninstall:
	$(call check_namespace)
	@echo "Uninstalling Ansible Log Monitor from namespace $(NAMESPACE)..."
	@helm uninstall alm -n $(NAMESPACE)
	@echo "Removing model-secret if it exists..."
	@oc delete secret model-secret -n $(NAMESPACE) --ignore-not-found
	@echo "Removing all PVCs..."
	@oc delete pvc --all -n $(NAMESPACE) 2>/dev/null || true
	@echo "✅ Ansible Log Monitor uninstalled completely (all data removed)"

# Upgrade 
.PHONY: upgrade
upgrade:
	@echo "Upgrading Ansible Log Monitor in namespace $(NAMESPACE)..."
	@helm upgrade alm ./ansible-log-monitor -n $(NAMESPACE) $(env_args)
	@echo "Ansible Log Monitor upgraded"

# Restart
.PHONY: restart
restart: uninstall install
	$(call check_namespace)
	@echo "Restarting Ansible Log Monitor in namespace $(NAMESPACE)..."
	@echo "Ansible Log Monitor restarted"

# Port forwarding helpers (NAMESPACE optional, defaults to current project)
.PHONY: port-forward-ui
port-forward-ui:
	$(call check_namespace)
	@echo "🔗 Forwarding UI service to localhost:7860..."
	@oc port-forward -n $(NAMESPACE) svc/ui 7860:7860

.PHONY: port-forward-backend
port-forward-backend:
	$(call check_namespace)
	@echo "🔗 Forwarding Backend service to localhost:8000..."
	@oc port-forward -n $(NAMESPACE) svc/backend 8000:8000

.PHONY: port-forward-annotation
port-forward-annotation:
	$(call check_namespace)
	@echo "🔗 Forwarding Annotation Interface to localhost:7861..."
	@oc port-forward -n $(NAMESPACE) svc/annotation-interface 7861:7861

.PHONY: port-forward-grafana
port-forward-grafana:
	$(call check_namespace)
	@echo "🔗 Forwarding Grafana to localhost:3000..."
	@oc port-forward -n $(NAMESPACE) svc/grafana 3000:3000

# =============================================================================
# AAP Mock Log Loading
# =============================================================================
# Loads all logs from data/logs/ directory (including subdirectories)

.PHONY: wait-for-aap-mock
wait-for-aap-mock: ## ⏳ Wait for aap-mock pod to be ready
	$(call check_namespace)
	@for i in $$(seq 1 60); do \
		POD=$$(oc get pod -n $(NAMESPACE) -l app.kubernetes.io/name=aap-mock -o jsonpath='{.items[0].metadata.name}' 2>/dev/null); \
		if [ -n "$$POD" ]; then \
			STATUS=$$(oc get pod -n $(NAMESPACE) $$POD -o jsonpath='{.status.phase}' 2>/dev/null); \
			READY=$$(oc get pod -n $(NAMESPACE) $$POD -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null); \
			if [ "$$STATUS" = "Running" ] && [ "$$READY" = "True" ]; then \
				echo "   ✅ aap-mock is ready! (pod: $$POD)"; \
				exit 0; \
			fi; \
		fi; \
		echo "   ... waiting ($$i/60)"; \
		sleep 5; \
	done; \
	echo "   ❌ Timeout waiting for aap-mock"; \
	exit 1

.PHONY: load-logs
load-logs: ## 📂 Load logs from data/logs/ to aap-mock and start replay
	$(call check_namespace)
	@echo "📂 Loading logs to aap-mock..."
	@POD=$$(oc get pod -n $(NAMESPACE) -l app.kubernetes.io/name=aap-mock -o jsonpath='{.items[0].metadata.name}' 2>/dev/null); \
	if [ -z "$$POD" ]; then \
		echo "❌ Error: aap-mock pod not found. Is it running?"; \
		echo "   Run 'make install' first."; \
		exit 1; \
	fi; \
	echo "   Found pod: $$POD"; \
	echo "   Source: ../../data/logs/"; \
	if [ ! -d "../../data/logs" ]; then \
		echo "   ⚠️  Directory ../../data/logs not found"; \
		exit 1; \
	fi; \
	echo "   Preparing log files..."; \
	mkdir -p /tmp/aap-logs-staging; \
	rm -rf /tmp/aap-logs-staging/* 2>/dev/null || true; \
	cp ../../data/logs/*/*.txt ../../data/logs/*/*.log /tmp/aap-logs-staging/ 2>/dev/null; \
	cp ../../data/logs/*.txt ../../data/logs/*.log /tmp/aap-logs-staging/ 2>/dev/null; \
	FILE_COUNT=$$(ls -1 /tmp/aap-logs-staging/ 2>/dev/null | wc -l | tr -d ' '); \
	echo "   Copying $$FILE_COUNT files to pod..."; \
	oc cp -n $(NAMESPACE) /tmp/aap-logs-staging/. "$$POD:/app/sample-logs/"; \
	rm -rf /tmp/aap-logs-staging; \
	echo "   ✅ $$FILE_COUNT log files loaded to /app/sample-logs/"
	@echo "📂 Files loaded! Triggering refresh and replay..."
	@$(MAKE) -s trigger-refresh
	@$(MAKE) -s trigger-replay

.PHONY: trigger-refresh
trigger-refresh: ## 🔄 Trigger aap-mock to refresh auto-loaded files
	$(call check_namespace)
	@echo "🔄 Refreshing auto-loaded files..."
	@ROUTE=$$(oc get route -n $(NAMESPACE) -l app.kubernetes.io/name=aap-mock -o jsonpath='{.items[0].spec.host}' 2>/dev/null); \
	if [ -z "$$ROUTE" ]; then \
		echo "❌ Error: aap-mock route not found"; \
		exit 1; \
	fi; \
	curl -sf -X POST "http://$$ROUTE/api/auto-loaded/refresh" | head -c 300 || true; \
	echo ""; \
	echo "✅ Refresh complete!"

.PHONY: trigger-replay
trigger-replay: ## ▶️ Manually trigger log replay
	$(call check_namespace)
	@echo "▶️  Triggering log replay..."
	@ROUTE=$$(oc get route -n $(NAMESPACE) -l app.kubernetes.io/name=aap-mock -o jsonpath='{.items[0].spec.host}' 2>/dev/null); \
	if [ -z "$$ROUTE" ]; then \
		echo "❌ Error: aap-mock route not found"; \
		exit 1; \
	fi; \
	curl -sf -X POST "http://$$ROUTE/api/logs/replay" \
		-H "Content-Type: application/json" \
		-d '{"source": "auto-loaded", "id_or_path": "all", "loop": true, "rate_lines_per_sec": 100}' | head -c 300; \
	echo ""; \
	echo "✅ Replay started!"

.PHONY: stop-replay
stop-replay: ## ⏹️ Stop log replay
	$(call check_namespace)
	@echo "⏹️  Stopping log replay..."
	@ROUTE=$$(oc get route -n $(NAMESPACE) -l app.kubernetes.io/name=aap-mock -o jsonpath='{.items[0].spec.host}' 2>/dev/null); \
	if [ -z "$$ROUTE" ]; then \
		echo "❌ Error: aap-mock route not found"; \
		exit 1; \
	fi; \
	curl -sf -X POST "http://$$ROUTE/api/replay/stop" || true; \
	echo "✅ Replay stopped"

# =============================================================================
# Alloy Log Loading
# =============================================================================
# Loads all logs from data/logs/ directory (including subdirectories) to Alloy
# This is a workaround to collect logs from aap-mock container (work in progress)
# TODO: Remove this when we have a way to collect logs from aap-mock container (work in progress)

.PHONY: wait-for-alloy
wait-for-alloy: ## ⏳ Wait for alloy pod to be ready
	$(call check_namespace)
	@for i in $$(seq 1 60); do \
		POD=$$(oc get pod -n $(NAMESPACE) -l app.kubernetes.io/name=alloy -o jsonpath='{.items[0].metadata.name}' 2>/dev/null); \
		if [ -n "$$POD" ]; then \
			STATUS=$$(oc get pod -n $(NAMESPACE) $$POD -o jsonpath='{.status.phase}' 2>/dev/null); \
			READY=$$(oc get pod -n $(NAMESPACE) $$POD -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null); \
			if [ "$$STATUS" = "Running" ] && [ "$$READY" = "True" ]; then \
				echo "   ✅ alloy is ready! (pod: $$POD)"; \
				exit 0; \
			fi; \
		fi; \
		echo "   ... waiting ($$i/60)"; \
		sleep 5; \
	done; \
	echo "   ❌ Timeout waiting for alloy"; \
	exit 1

.PHONY: load-logs-to-alloy
load-logs-to-alloy: ## 📂 Load logs from data/logs/ to alloy pod
	$(call check_namespace)
	@echo "📂 Loading logs to alloy..."
	@POD=$$(oc get pod -n $(NAMESPACE) -l app.kubernetes.io/name=alloy -o jsonpath='{.items[0].metadata.name}' 2>/dev/null); \
	if [ -z "$$POD" ]; then \
		echo "❌ Error: alloy pod not found. Is it running?"; \
		echo "   Run 'make cluster/install' first."; \
		exit 1; \
	fi; \
	echo "   Found pod: $$POD"; \
	echo "   Source: ../../data/logs/"; \
	if [ ! -d "../../data/logs" ]; then \
		echo "   ⚠️  Directory ../../data/logs not found"; \
		exit 1; \
	fi; \
	echo "   Copying logs to pod (preserving directory structure)..."; \
	oc cp -n $(NAMESPACE) ../../data/logs/. "$$POD:/var/log/ansible_logs/"; \
	echo "   Verifying copied files..."; \
	COPIED_COUNT=$$(oc exec -n $(NAMESPACE) $$POD -- find /var/log/ansible_logs -type f \( -name "*.txt" -o -name "*.log" \) 2>/dev/null | wc -l | tr -d ' '); \
	echo "   $$COPIED_COUNT log files synced to /var/log/ansible_logs/"
