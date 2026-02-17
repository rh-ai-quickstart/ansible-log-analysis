.PHONY: build push help

# Auto-detect container runtime (podman or docker)
CONTAINER_CMD := $(shell command -v podman 2>/dev/null || command -v docker 2>/dev/null)

# Image configuration
IMAGE_REGISTRY := quay.io/rh-ai-quickstart
IMAGE_NAME := alm-aap-log-collector
IMAGE_TAG := latest
FULL_IMAGE := $(IMAGE_REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)

help: ## Show this help message
	@echo "AAP Log Collector - Container Build Targets"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'
	@echo ""
	@echo "Using container runtime: $(CONTAINER_CMD)"

build: ## Build the container image
	@echo "Building $(FULL_IMAGE) with $(CONTAINER_CMD)..."
	$(CONTAINER_CMD) build -t $(FULL_IMAGE) .
	@echo "✅ Image built successfully: $(FULL_IMAGE)"

push: build ## Build and push the image to registry
	@echo "Pushing $(FULL_IMAGE) to registry..."
	$(CONTAINER_CMD) push $(FULL_IMAGE)
	@echo "✅ Image pushed successfully: $(FULL_IMAGE)"

run: build ## Build and run the container locally (for testing)
	@echo "Make sure test-logs directory exists and is empty..."
	@if [ -d "$(PWD)/test-logs" ]; then rm -rf $(PWD)/test-logs; fi
	@mkdir -p $(PWD)/test-logs
	@echo "Running $(FULL_IMAGE) locally..."
	@echo "Note: Make sure AAP Mock is running and accessible"
	$(CONTAINER_CMD) run --rm \
		-e AAP_API_URL=http://localhost:8082 \
		-e CLUSTER_NAME=test-cluster \
		-e LOG_LEVEL=DEBUG \
		-v $(PWD)/test-logs:/var/log/ansible_logs:z \
		$(FULL_IMAGE)
