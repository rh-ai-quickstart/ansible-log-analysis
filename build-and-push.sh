#!/usr/bin/env bash

set -euo pipefail

echo "🚀 Starting build & push pipeline..."

# ---- RAG service ----
echo "📦 Building alm-rag:latest"
podman build \
  -f services/rag/Containerfile \
  -t quay.io/rh-ai-quickstart/alm-rag:latest \
  .

echo "⬆️  Pushing alm-rag:latest"
podman push quay.io/rh-ai-quickstart/alm-rag:latest


# ---- Text Embeddings Inference (TEI) ----
echo "📦 Building alm-rag:tei-rag-v1"
pushd services/text-embeddings-inference > /dev/null

podman build \
  -f Dockerfile \
  -t quay.io/rh-ai-quickstart/alm-rag:tei-rag-v1 \
  .

echo "⬆️  Pushing alm-rag:tei-rag-v1"
podman push quay.io/rh-ai-quickstart/alm-rag:tei-rag-v1

popd > /dev/null


# ---- Backend ----
echo "📦 Building alm-backend:latest"
podman build \
  -f Containerfile \
  -t quay.io/rh-ai-quickstart/alm-backend:latest \
  .

echo "⬆️  Pushing alm-backend:latest"
podman push quay.io/rh-ai-quickstart/alm-backend:latest


echo "✅ All images built and pushed successfully!"
