from typing import List, Optional, Tuple
from alm.llm import stream_with_fallback
from sklearn.base import ClusterMixin
import os
from sentence_transformers import SentenceTransformer
from sklearn.cluster import (
    DBSCAN,
    MeanShift,
    AgglomerativeClustering,
    HDBSCAN,
    estimate_bandwidth,
)
from sklearn.metrics.pairwise import cosine_distances
import joblib
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from alm.agents.output_scheme import (
    SummarySchema,
    ClassifySchema,
    RouterStepByStepSolutionSchema,
)
import numpy as np
from alm.utils.minio import upload_model_to_minio
import requests

from alm.agents.prompts.prompts import (
    log_summary_system_message,
    log_summary_user_message,
    log_category_user_message,
    log_category_system_message,
    suggest_step_by_step_solution_user_message,
    suggest_step_by_step_solution_with_context_user_message,
    suggest_step_by_step_solution_system_message,
    router_step_by_step_solution_user_message,
    router_step_by_step_solution_system_message,
)
from alm.utils.logger import get_logger

logger = get_logger(__name__)


# Can be improve by using eval-optimizer.
async def summarize_log(log, llm: ChatOpenAI):
    llm_summary = llm.with_structured_output(SummarySchema)
    log_summary = await llm_summary.ainvoke(
        [
            {"role": "system", "content": log_summary_system_message},
            {
                "role": "user",
                "content": log_summary_user_message.format(error_log=log),
            },
        ]
    )
    return log_summary.summary


async def classify_log(log_summary, llm: ChatOpenAI):
    llm_categorize = llm.with_structured_output(ClassifySchema)
    log_category = await llm_categorize.ainvoke(
        [
            {"role": "system", "content": log_category_system_message},
            {
                "role": "user",
                "content": log_category_user_message.format(log_summary=log_summary),
            },
        ]
    )
    return log_category.category


async def router_step_by_step_solution(log_summary: str, llm: ChatOpenAI):
    llm_router_step_by_step_solution = llm.with_structured_output(
        RouterStepByStepSolutionSchema
    )
    router_step_by_step_solution = await llm_router_step_by_step_solution.ainvoke(
        [
            {
                "role": "system",
                "content": router_step_by_step_solution_system_message,
            },
            {
                "role": "user",
                "content": router_step_by_step_solution_user_message.format(
                    log_summary=log_summary
                ),
            },
        ]
    )
    return router_step_by_step_solution.suggestion


async def suggest_step_by_step_solution(
    log_summary: str,
    log: str,
    llm: ChatOpenAI,
    context: Optional[str] = None,
    streaming: bool = False,
):
    if context:
        user_msg = suggest_step_by_step_solution_with_context_user_message.format(
            context=context,
            log=log,
            log_summary=log_summary,
        )
    else:
        user_msg = suggest_step_by_step_solution_user_message.format(
            log=log,
            log_summary=log_summary,
        )

    messages = [
        {
            "role": "system",
            "content": suggest_step_by_step_solution_system_message,
        },
        {
            "role": "user",
            "content": user_msg,
        },
    ]
    if streaming:
        return await stream_with_fallback(llm, messages)
    else:
        return (await llm.ainvoke(messages)).content


def _embed_logs(logs: List[str]):
    # Check if remote embeddings API is configured
    api_key = os.getenv("EMBEDDINGS_LLM_API_KEY")
    base_url = os.getenv("EMBEDDINGS_LLM_URL")
    embedding_model_name = os.getenv("EMBEDDINGS_LLM_MODEL_NAME")
    texts = [summary[-50:] for summary in logs]

    if api_key and base_url and embedding_model_name:
        # Use remote OpenAI-compatible embeddings API
        return _embed_logs_remote(texts, api_key, base_url, embedding_model_name)
    else:
        # Use local SentenceTransformer model
        return _embed_logs_local(texts)


def _embed_logs_remote(
    texts: List[str], api_key: str, base_url: str, embedding_model_name: str
):
    logger.debug("Using remote embeddings API")

    embeddings_client = OpenAIEmbeddings(
        api_key=api_key,
        base_url=base_url,
        model=embedding_model_name,
    )

    # Generate embeddings
    embeddings = embeddings_client.embed_documents(texts)

    # Convert to numpy array
    embeddings = np.array(embeddings)
    logger.debug("finished embeddings")
    return embeddings


def _embed_logs_local(texts: List[str]):
    logger.debug("Using local SentenceTransformer model")
    model_name = os.getenv("SENTENCE_TRANSFORMER_MODEL_NAME")
    encoder = SentenceTransformer(model_name)
    embeddings = encoder.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        # batch_size=10,
    )
    logger.debug("finished embeddings")
    return embeddings


def _cluster_logs(embeddings: np.ndarray) -> Tuple[ClusterMixin, np.ndarray]:
    algorithm = os.getenv("CLUSTERING_ALGORITHM")
    if algorithm.lower() == "dbscan":
        # DBSCAN - Good for finding clusters of varying shapes and handling noise
        # Uses cosine distance for text similarity
        distance_matrix = cosine_distances(embeddings)
        cluster_model = DBSCAN(eps=0.3, min_samples=2, metric="precomputed")
        cluster_labels = cluster_model.fit_predict(distance_matrix)

    elif algorithm.lower() == "hdbscan":
        cluster_model = HDBSCAN(min_cluster_size=2, metric="cosine")
        cluster_labels = cluster_model.fit_predict(embeddings)

    elif algorithm.lower() == "meanshift":
        # Mean Shift - Automatically determines number of clusters
        bandwidth = estimate_bandwidth(embeddings, quantile=0.05)
        cluster_model = MeanShift(bandwidth=bandwidth)  # Auto-estimate bandwidth
        cluster_labels = cluster_model.fit_predict(embeddings)

    elif algorithm.lower() == "agglomerative":
        # Agglomerative Clustering with distance threshold
        # Automatically determines number of clusters based on distance threshold
        cluster_model = AgglomerativeClustering(
            n_clusters=None, distance_threshold=0.5, linkage="average", metric="cosine"
        )
        cluster_labels = cluster_model.fit_predict(embeddings)

    else:
        raise ValueError(
            f"Unsupported algorithm: {algorithm}. Choose from 'dbscan', 'meanshift', 'agglomerative'"
        )
    return cluster_model, cluster_labels


def infer_cluster_log(log: str):
    embeddings = _embed_logs([log])
    if os.getenv("CLUSTERING_HOST"):
        logger.debug(f"inferring cluster log from {os.getenv('CLUSTERING_HOST')}")
        response = requests.post(
            f"http://{os.getenv('CLUSTERING_HOST')}:{os.getenv('CLUSTERING_PORT')}/cluster",
            json={"embeddings": embeddings.tolist()},
        )
        label_as_int = response.json()["labels"][0]
    else:
        logger.debug(
            f"loading cluster model from local file {os.getenv('TMP_CLUSTER_MODEL_PATH')}"
        )
        cluster_model = joblib.load(os.getenv("TMP_CLUSTER_MODEL_PATH"))
        cluster_label = cluster_model.predict(embeddings)
        label_as_int = cluster_label.tolist()[0]
    return str(label_as_int)


def _handle_outlaier_cluster(cluster_labels: np.ndarray):
    clusters = np.unique(cluster_labels)
    max_cluster = clusters.max()
    # each cluster that is -1 replace it to be in cluster by himself

    # Find indices where cluster_labels is -1 (outliers/noise points)
    outlier_indices = np.where(cluster_labels == -1)[0]
    logger.debug(f"number of outliers: {len(outlier_indices)}")
    # Assign each outlier its own unique cluster ID
    next_cluster_id = max_cluster + 1
    for idx in outlier_indices:
        cluster_labels[idx] = next_cluster_id
        next_cluster_id += 1

    return cluster_labels


def train_embed_and_cluster_logs(
    logs: List[str],
    save_cluster_model: bool = True,
) -> List[str]:
    """
    Cluster log summaries using sentence embeddings and various clustering algorithms.

    Args:
        log_summaries: List of log summary strings

    Returns:
        List of cluster labels for each log summary (same order as input)
    """
    if not logs:
        return []
    import pandas as pd

    # Embed logs
    embeddings = _embed_logs(logs)
    pd.DataFrame(
        zip([embedding.tolist() for embedding in embeddings], logs),
        columns=["embedding", "log"],
    ).to_csv("embeddings.csv", index=False)
    # Train clustering model
    cluster_model, cluster_labels = _cluster_logs(embeddings)

    # handle outlaier cluster
    cluster_labels = _handle_outlaier_cluster(cluster_labels)

    if save_cluster_model:
        if os.getenv("MINIO_BUCKET_NAME"):
            upload_model_to_minio(
                cluster_model, os.getenv("MINIO_BUCKET_NAME"), "clustering_model.joblib"
            )
        else:
            joblib.dump(cluster_model, os.getenv("TMP_CLUSTER_MODEL_PATH"))

    return cluster_labels.astype(str).tolist()
