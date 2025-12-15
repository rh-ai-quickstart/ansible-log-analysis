.PHONY: deploy start stop status help postgres phoenix loki-stack aap-mock replay stop-replay backend ui annotation health logs restart clean stop-postgres stop-phoenix stop-loki-stack stop-aap-mock embedding stop-embedding wait-for-embedding test-embedding test-rag
.DEFAULT_GOAL := help

include ../../.env
export $(shell sed 's/=.*//' ../../.env | xargs)

COMPOSE_CMD := $(shell command -v docker-compose 2>/dev/null || command -v podman-compose 2>/dev/null)

# Auto-detect container runtime socket (Docker or Podman)
# Podman rootless: ${XDG_RUNTIME_DIR}/podman/podman.sock
# Podman root: /run/podman/podman.sock
# Docker: /var/run/docker.sock
CONTAINER_SOCK := $(shell \
	if [ -S "$${XDG_RUNTIME_DIR}/podman/podman.sock" ]; then \
		echo "$${XDG_RUNTIME_DIR}/podman/podman.sock"; \
	elif [ -S "/run/podman/podman.sock" ]; then \
		echo "/run/podman/podman.sock"; \
	else \
		echo "/var/run/docker.sock"; \
	fi)
export CONTAINER_SOCK

train: run-whole-training-pipeline
training: run-whole-training-pipeline

uninstall: stop

install: start
deploy: start

help: ## Show this help message
	@echo "🚀 Ansible Log Monitor - Local Deployment"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'
	@echo ""
	@echo "📚 RAG Setup:"
	@echo "  1. Place PDF files in data/knowledge_base/"
	@echo "  2. Run 'make run-whole-training-pipeline' to build RAG index"
	@echo "  3. Ensure RAG_ENABLED=true in .env file"
	@echo ""


start: stop ## 🚀 Start all services locally
	@echo "🚀 Starting local deployment of Ansible Log Monitor..."
	@echo "🔌 Cleaning up any existing processes..."
	@$(MAKE) -s postgres
	@echo "   Waiting for PostgreSQL to be ready..."
	@$(MAKE) -s phoenix
	@$(MAKE) -s loki-stack
	@$(MAKE) -s embedding
	@echo "   ⚠️  Note: Embedding service may take 3-5 minutes to load the model"
	@echo "   You can check status with: make test-embedding"
	@$(MAKE) -s aap-mock
	@$(MAKE) -s backend
	@$(MAKE) -s ui
	@$(MAKE) -s annotation
	@echo ""
	@echo "🎉 All services are running!"
	@sleep 8
	@$(MAKE) -s status
	@echo ""
	@echo "Press Ctrl+C to stop all services, or run 'make stop'"
	@bash -c 'trap "exit" SIGINT SIGTERM; while true; do sleep 5; done'

postgres: ## 📊 Start PostgreSQL database
	@echo "   Starting PostgreSQL database..."
	@$(COMPOSE_CMD) -f compose.yaml up -d postgres

phoenix: ## 📊 Start Phoenix
	@echo "   Starting Phoenix..."
	@$(COMPOSE_CMD) -f compose.yaml up -d phoenix

loki-stack: ## 📊 Start Loki stack (Loki, Loki MCP Server, Grafana, Promtail)
	@echo "   Starting Loki stack (Loki, Loki MCP Server, Grafana, Promtail)..."
	@$(COMPOSE_CMD) -f compose.yaml up -d loki loki-mcp-server grafana promtail

embedding: ## 🤖 Start Embedding Service (TEI)
	@echo "   Starting Embedding Service (TEI)..."
	@$(COMPOSE_CMD) -f compose.yaml up -d alm-embedding

wait-for-embedding: ## ⏳ Wait for embedding service to be ready (optional, for manual use)
	@echo "   Waiting for embedding service to be ready (this may take 3-5 minutes for model loading)..."
	@timeout=300; \
	elapsed=0; \
	interval=60; \
	while [ $$elapsed -lt $$timeout ]; do \
		if curl -s -f http://localhost:8080/health >/dev/null 2>&1; then \
			echo "   ✓ Embedding service is ready!"; \
			exit 0; \
		fi; \
		echo "   ⏳ Still waiting... ($$elapsed/$$timeout seconds)"; \
		sleep $$interval; \
		elapsed=$$((elapsed + interval)); \
	done; \
	echo "   ⚠️  Warning: Embedding service did not become ready within $$timeout seconds"; \
	echo "   The service may still be loading the model. Check logs with: docker-compose logs alm-embedding"; \
	exit 0

prepare-logs: ## 📂 Flatten log files from data/logs/ into staging directory
	@echo "   Preparing log files..."
	@mkdir -p .staging/sample-logs
	@rm -rf .staging/sample-logs/* 2>/dev/null || true
	@echo "   Copying log files from ../../data/logs/..."
	@cp ../../data/logs/*/*.txt ../../data/logs/*/*.log .staging/sample-logs/ 2>/dev/null; \
	cp ../../data/logs/*.txt ../../data/logs/*.log .staging/sample-logs/ 2>/dev/null; \
	FILE_COUNT=$$(ls -1 .staging/sample-logs/ 2>/dev/null | wc -l | tr -d ' '); \
	echo "   ✅ $$FILE_COUNT log files prepared"

aap-mock: prepare-logs ## 🎭 Start AAP Mock (Log Generator) and auto-start replay
	@echo "   Starting AAP Mock (Log Generator)..."
	@$(COMPOSE_CMD) -f compose.yaml up -d aap-mock
	@echo "   Waiting for AAP Mock to load 500 files and start server..."
	@echo "   (Files load BEFORE server starts - this takes ~45-60 seconds)"
	@for i in $$(seq 1 45); do \
		FILE_COUNT=$$(curl -s http://localhost:8082/api/auto-loaded 2>/dev/null | grep -o '"count":[0-9]*' | grep -o '[0-9]*' || echo ""); \
		if [ -n "$$FILE_COUNT" ] && [ "$$FILE_COUNT" -gt 0 ] 2>/dev/null; then \
			echo "   ✓ Server ready with $$FILE_COUNT files loaded!"; \
			break; \
		fi; \
		if [ $$i -eq 45 ]; then \
			echo "   ⚠️  Timeout (90s) - proceeding anyway..."; \
		else \
			echo "   ... loading files ($$i/45) - server not ready yet"; \
			sleep 2; \
		fi; \
	done
	@echo "   Starting log replay (loop mode)..."
	@curl -s -X POST "http://localhost:8082/api/logs/replay" \
		-H "Content-Type: application/json" \
		-d '{"source": "auto-loaded", "id_or_path": "all", "loop": true, "rate_lines_per_sec": 100}' | head -c 200 || true
	@echo ""
	@echo "   ✅ AAP Mock running with replay started!"

replay: ## 🔄 Start log replay on AAP Mock (loop mode)
	@echo "   Starting log replay..."
	@sleep 2
	@curl -s -X POST "http://localhost:8082/api/logs/replay" \
		-H "Content-Type: application/json" \
		-d '{"source": "auto-loaded", "id_or_path": "all", "loop": true, "rate_lines_per_sec": 100}' | head -c 200
	@echo ""
	@echo "   ✅ Replay started! View logs at http://localhost:3000 with query: {job=\"aap-mock\"}"

stop-replay: ## 🛑 Stop log replay
	@echo "   Stopping log replay..."
	@curl -s -X POST "http://localhost:8082/api/replay/stop" || true
	@echo "   ✅ Replay stopped"

backend: ## ⚙️ Start Backend (FastAPI)
	@echo "   Starting Backend API..."
	@cd ../.. && uv run uvicorn alm.main_fastapi:app --reload &

ui: ## 🖥️ Start UI (Gradio)
	@echo "   Starting UI Interface..."
	@cd ../../ui && uv run gradio app.py &

annotation: ## 🏷️ Start Annotation UI (Gradio)  
	@echo "   Starting Annotation UI..."
	@cd ../.. && uv run gradio annotation_interface/app.py &


stop: ## 🛑 Stop all services
	@echo "🛑 Stopping all services..."
	@echo "🔧 Using: $(COMPOSE_CMD)"
	@$(MAKE) -s stop-aap-mock
	@$(MAKE) -s stop-loki-stack
	@$(MAKE) -s stop-postgres
	@$(MAKE) -s stop-phoenix
	@$(MAKE) -s stop-embedding
	@$(MAKE) -s kill-ports
	@echo "👋 All services stopped"

stop-postgres: ## 🛑 Stop PostgreSQL database
	@$(COMPOSE_CMD) -f compose.yaml stop postgres || true
	@$(COMPOSE_CMD) -f compose.yaml rm -f postgres || true
	@echo "   ✓ PostgreSQL stopped"

stop-phoenix: ## 🛑 Stop Phoenix
	@$(COMPOSE_CMD) -f compose.yaml stop phoenix || true
	@$(COMPOSE_CMD) -f compose.yaml rm -f phoenix || true
	@echo "   ✓ Phoenix stopped"

stop-loki-stack: ## 🛑 Stop Loki stack (Loki, Loki MCP Server, Grafana, Promtail)
	@$(COMPOSE_CMD) -f compose.yaml stop loki-mcp-server grafana promtail loki || true
	@$(COMPOSE_CMD) -f compose.yaml rm -f loki-mcp-server grafana promtail loki || true
	@echo "   ✓ Loki stack stopped"

stop-aap-mock: ## 🛑 Stop AAP Mock
	@$(COMPOSE_CMD) -f compose.yaml stop aap-mock || true
	@$(COMPOSE_CMD) -f compose.yaml rm -f aap-mock || true
	@echo "   ✓ AAP Mock stopped"

stop-embedding: ## 🛑 Stop Embedding Service
	@$(COMPOSE_CMD) -f compose.yaml down alm-embedding || true
	@echo "   ✓ Embedding service stopped"

kill-ports: ## 🔌 Kill processes using required ports
	@if [ "$$(uname)" = "Darwin" ]; then \
		lsof -ti :7860 | xargs kill -9 2>/dev/null || true; \
	else \
		fuser -k 7860/tcp 2>/dev/null || true; \
	fi
	@echo "   ✓ UI 7860 killed"
	@if [ "$$(uname)" = "Darwin" ]; then \
		lsof -ti :7861 | xargs kill -9 2>/dev/null || true; \
	else \
		fuser -k 7861/tcp 2>/dev/null || true; \
	fi
	@echo "   ✓ Annotation 7861 killed"
	@if [ "$$(uname)" = "Darwin" ]; then \
		lsof -ti :8000 | xargs kill -9 2>/dev/null || true; \
	else \
		fuser -k 8000/tcp 2>/dev/null || true; \
	fi
	@echo "   ✓ Backend 8000 killed"
	@if [ "$$(uname)" = "Darwin" ]; then \
		lsof -ti :6006 | xargs kill -9 2>/dev/null || true; \
	else \
		fuser -k 6006/tcp 2>/dev/null || true; \
	fi
	@echo "   ✓ Phoenix 6006 killed"
	@if [ "$$(uname)" = "Darwin" ]; then \
		lsof -ti :4317 | xargs kill -9 2>/dev/null || true; \
	else \
		fuser -k 4317/tcp 2>/dev/null || true; \
	fi
	@echo "   ✓ Phoenix 4317 killed"
	@if [ "$$(uname)" = "Darwin" ]; then \
		lsof -ti :3100 | xargs kill -9 2>/dev/null || true; \
	else \
		fuser -k 3100/tcp 2>/dev/null || true; \
	fi
	@echo "   ✓ Loki 3100 killed"
	@if [ "$$(uname)" = "Darwin" ]; then \
		lsof -ti :8080 | xargs kill -9 2>/dev/null || true; \
	else \
		fuser -k 8080/tcp 2>/dev/null || true; \
	fi
	@echo "   ✓ Embedding service 8080 killed"
	@if [ "$$(uname)" = "Darwin" ]; then \
		lsof -ti :8081 | xargs kill -9 2>/dev/null || true; \
	else \
		fuser -k 8081/tcp 2>/dev/null || true; \
	fi
	@echo "   ✓ Loki MCP Server 8081 killed"
	@if [ "$$(uname)" = "Darwin" ]; then \
		lsof -ti :3000 | xargs kill -9 2>/dev/null || true; \
	else \
		fuser -k 3000/tcp 2>/dev/null || true; \
	fi
	@echo "   ✓ Grafana 3000 killed"
	@if [ "$$(uname)" = "Darwin" ]; then \
		lsof -ti :8082 | xargs kill -9 2>/dev/null || true; \
	else \
		fuser -k 8082/tcp 2>/dev/null || true; \
	fi
	@echo "   ✓ AAP Mock 8082 killed"

status: ## 📊 Show status of all services
	@echo "📊 Service Status:"
	@echo "   📊 PostgreSQL: $(shell $(COMPOSE_CMD) ps 2>/dev/null | grep 'postgres' | grep -q 'Up' && echo 'Running' || echo 'Stopped') - localhost:5432"
	@echo "   🐍 Phoenix: $(shell $(COMPOSE_CMD) ps 2>/dev/null | grep 'phoenix' | grep -q 'Up' && echo 'Running' || echo 'Stopped') - http://localhost:6006"
	@echo "   📊 Loki: $(shell $(COMPOSE_CMD) ps 2>/dev/null | grep 'loki' | grep -v 'loki-mcp' | grep -q 'Up' && echo 'Running' || echo 'Stopped') - http://localhost:3100"
	@echo "   🔍 Loki MCP Server: $(shell $(COMPOSE_CMD) ps 2>/dev/null | grep 'loki-mcp-server' | grep -q 'Up' && echo 'Running' || echo 'Stopped') - http://localhost:8081"
	@echo "   🤖 Embedding Service: $(shell $(COMPOSE_CMD) ps 2>/dev/null | grep 'alm-embedding' | grep -q 'Up' && echo 'Running' || echo 'Stopped') - http://localhost:8080"
	@echo "   📈 Grafana: $(shell $(COMPOSE_CMD) ps 2>/dev/null | grep 'grafana' | grep -q 'Up' && echo 'Running' || echo 'Stopped') - http://localhost:3000"
	@echo "   📝 Promtail: $(shell $(COMPOSE_CMD) ps 2>/dev/null | grep 'promtail' | grep -q 'Up' && echo 'Running' || echo 'Stopped')"
	@echo "   🎭 AAP Mock: $(shell $(COMPOSE_CMD) ps 2>/dev/null | grep 'aap-mock' | grep -q 'Up' && echo 'Running' || echo 'Stopped') - http://localhost:8082"
	@if lsof -i :8000 >/dev/null 2>&1; then \
		echo "   ⚙️  Backend API: Running - http://localhost:8000"; \
	else \
		echo "   ⚙️  Backend API: Stopped - http://localhost:8000"; \
	fi
	@if lsof -i :7860 >/dev/null 2>&1; then \
		echo "   🖥️  UI Interface: Running - http://localhost:7860"; \
	else \
		echo "   🖥️  UI Interface: Stopped - http://localhost:7860"; \
	fi
	@if lsof -i :7861 >/dev/null 2>&1; then \
		echo "   🏷️  Annotation UI: Running - http://localhost:7861"; \
	else \
		echo "   🏷️  Annotation UI: Stopped - http://localhost:7861"; \
	fi
	@if curl -s http://localhost:6006 >/dev/null 2>&1; then \
		echo "   🐍  Phoenix: Running - http://localhost:6006"; \
	else \
		echo "   🐍  Phoenix: Stopped - http://localhost:6006"; \
	fi

run-whole-training-pipeline: ## 🔍 Run whole training pipeline (builds RAG index)
	@echo "🔍 Running whole training pipeline..."
	@echo "📚 This will build the RAG index from PDFs in data/knowledge_base/"
	@if [ ! -d "../../data/knowledge_base" ] || [ -z "$$(ls -A ../../data/knowledge_base/*.pdf 2>/dev/null)" ]; then \
		echo "⚠️  WARNING: No PDF files found in data/knowledge_base/"; \
		echo "   RAG index will not be created. Add PDF files and run again."; \
	fi
	@( cd ../.. && uv run init_pipeline.py )

# Add new target to check RAG status
rag-status: ## 📊 Check RAG index status
	@echo "📊 RAG Index Status:"
	@if [ -f "../../data/ansible_errors.index" ] && [ -f "../../data/error_metadata.pkl" ]; then \
		echo "   ✅ RAG index exists"; \
		ls -lh ../../data/ansible_errors.index ../../data/error_metadata.pkl; \
	else \
		echo "   ❌ RAG index not found"; \
		echo "   Run 'make run-whole-training-pipeline' to build it"; \
	fi
	@if [ -d "../../data/knowledge_base" ]; then \
		pdf_count=$$(ls -1 ../../data/knowledge_base/*.pdf 2>/dev/null | wc -l); \
		echo "   📄 PDF files in knowledge_base: $$pdf_count"; \
	else \
		echo "   ⚠️  Knowledge base directory not found"; \
	fi

test-embedding: ## 🧪 Test embedding service
	@echo "🧪 Testing Embedding Service..."
	@echo ""
	@echo "1. Health Check:"
	@if curl -s -f http://localhost:8080/health >/dev/null 2>&1; then \
		echo "   ✅ Service is healthy"; \
	else \
		echo "   ❌ Service is not responding"; \
		exit 1; \
	fi
	@echo ""
	@echo "2. Generating test embedding:"
	@response=$$(curl -s -X POST http://localhost:8080/v1/embeddings \
		-H "Content-Type: application/json" \
		-d '{"model": "nomic-ai/nomic-embed-text-v1.5", "input": ["test query"]}'); \
	if echo "$$response" | grep -q '"data"'; then \
		echo "   ✅ Embedding generated successfully"; \
		echo "   Response preview: $$(echo $$response | head -c 200)..."; \
	else \
		echo "   ❌ Failed to generate embedding"; \
		echo "   Response: $$response"; \
		exit 1; \
	fi
	@echo ""
	@echo "✅ Embedding service test passed!"

test-rag: ## 🧪 Test RAG system (requires RAG index)
	@echo "🧪 Testing RAG System..."
	@if [ ! -f "../../data/ansible_errors.index" ]; then \
		echo "❌ RAG index not found. Run 'make run-whole-training-pipeline' first."; \
		exit 1; \
	fi
	@echo "Running RAG test script..."
	@(cd ../.. && uv run python tests/rag/test_embeddings.py)

health: ## 🔍 Check health of running services
	@echo "🔍 Health Checks:"
	@if curl -s http://localhost:8000/health >/dev/null 2>&1; then \
		echo "   ⚙️  Backend http://localhost:8000: Healthy"; \
	else \
		echo "   ⚙️  Backend http://localhost:8000: Unhealthy"; \
	fi
	@if curl -s http://localhost:7860 >/dev/null 2>&1; then \
		echo "   🖥️  UI http://localhost:7860: Healthy"; \
	else \
		echo "   🖥️  UI http://localhost:7860: Unhealthy"; \
	fi
	@if curl -s http://localhost:7861 >/dev/null 2>&1; then \
		echo "   🏷️  Annotation UI http://localhost:7861: Healthy"; \
	else \
		echo "   🏷️  Annotation UI http://localhost:7861: Unhealthy"; \
	fi
	@if curl -s http://localhost:6006 >/dev/null 2>&1; then \
		echo "   🐍  Phoenix: Running - http://localhost:6006"; \
	else \
		echo "   🐍  Phoenix: Stopped - http://localhost:6006"; \
	fi
	@if curl -s http://localhost:3100/ready >/dev/null 2>&1; then \
		echo "   📊  Loki http://localhost:3100: Healthy"; \
	else \
		echo "   📊  Loki http://localhost:3100: Unhealthy"; \
	fi
	@if curl -s http://localhost:8081 >/dev/null 2>&1; then \
		echo "   🔍  Loki MCP Server http://localhost:8081: Healthy"; \
	else \
		echo "   🔍  Loki MCP Server http://localhost:8081: Unhealthy"; \
	fi
	@if curl -s http://localhost:8080/health >/dev/null 2>&1; then \
		echo "   🤖  Embedding Service http://localhost:8080: Healthy"; \
	else \
		echo "   🤖  Embedding Service http://localhost:8080: Unhealthy (may still be loading model)"; \
	fi
	@if curl -s http://localhost:3000 >/dev/null 2>&1; then \
		echo "   📈  Grafana http://localhost:3000: Healthy"; \
	else \
		echo "   📈  Grafana http://localhost:3000: Unhealthy"; \
	fi
	@if curl -s http://localhost:8082/healthz >/dev/null 2>&1; then \
		echo "   🎭  AAP Mock http://localhost:8082: Healthy"; \
	else \
		echo "   🎭  AAP Mock http://localhost:8082: Unhealthy"; \
	fi
restart: stop start ## 🔄 Restart all services

clean: ## 🧹 Clean PID files and logs
	@echo "🧹 Cleaning up PID files and logs..."
	@docker-compose down -v
	@echo "   ✓ Cleanup completed"
