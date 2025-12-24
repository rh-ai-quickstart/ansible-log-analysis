# This makefile routes targets to local or helm specific makefiles
.PHONY: all local helm help rag-status test-rag

# ifneq (,$(wildcard .env))
# # ifneq (,$(filter local,$(MAKECMDGOALS)))
# include .env
# # endif
# endif

.EXPORT_ALL_VARIABLES:

all: ## Show usage instructions
	@echo "Ansible Log Monitor"
	@echo "========================="
	@echo ""
	@echo "Usage:"
	@echo "  make local/<target>   - Run local development targets"
	@echo "  make cluster/<target>    - Run helm deployment targets"
	@echo ""
	@echo "Examples:"
	@echo "  make local/dev        - Start local development environment"
	@echo "  make local/help       - Show local development help"
	@echo "  make cluster/install     - Install via helm (requires NAMESPACE)"
	@echo "  make cluster/help        - Show helm deployment help"
	@echo ""
	@echo "For target-specific help:"
	@echo "  make local/help"
	@echo "  make cluster/help"

help: all ## Show help (alias for all)

local/%: ## Route local targets to deploy/local/Makefile
	@$(MAKE) -C deploy/local $*

cluster/%: ## Route deploy targets to deploy/helm/Makefile
	@$(MAKE) -C deploy/helm $*

# Convenience targets for common local commands
rag-status: local/rag-status ## Check RAG service status
test-rag: local/test-rag ## Test RAG service
