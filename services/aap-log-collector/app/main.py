"""AAP Log Collector - Polls AAP Mock API and writes job logs to shared volume."""

import requests
import time
import logging
from pathlib import Path
from typing import Set, List, Dict, Any

from .config import Config, setup_logging


# Final job states that indicate a job is complete
FINAL_STATES = {"successful", "failed"}

# Global state: Track processed job IDs (in-memory)
processed_job_ids: Set[int] = set()

logger = logging.getLogger(__name__)


def fetch_all_jobs(api_url: str, page_size: int = 100) -> List[Dict[str, Any]]:
    """
    Fetch all jobs from AAP Mock API, handling pagination.

    Args:
        api_url: Base URL of AAP Mock API
        page_size: Number of jobs per page (1-200)

    Returns:
        List of all job objects with status included
    """
    all_jobs = []
    page = 1

    while True:
        try:
            logger.debug(f"Fetching page {page} with page_size {page_size}")
            response = requests.get(
                f"{api_url}/api/v2/jobs/",
                params={"page": page, "page_size": page_size},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            all_jobs.extend(results)
            logger.debug(f"Fetched {len(results)} jobs from page {page}")

            # Check if there's a next page
            if not data.get("next"):
                logger.debug(f"No more pages. Total jobs fetched: {len(all_jobs)}")
                break

            page += 1

        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching page {page} from {api_url}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching jobs from {api_url}: {e}")
            raise

    return all_jobs


def fetch_job_logs(api_url: str, job_id: int) -> str:
    """
    Fetch logs/stdout for a specific job.

    Args:
        api_url: Base URL of AAP Mock API
        job_id: Job ID to fetch logs for

    Returns:
        Job log content as string
    """
    try:
        logger.debug(f"Fetching logs for job {job_id}")
        response = requests.get(f"{api_url}/api/v2/jobs/{job_id}/stdout/", timeout=60)
        response.raise_for_status()
        # Parse JSON response and extract content field
        # This automatically unescapes \n characters to proper newlines
        data = response.json()
        return data.get("content", "")
    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching logs for job {job_id}")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching logs for job {job_id}: {e}")
        raise


def write_log_file(path: Path, content: str):
    """
    Write log content to file atomically, creating directories as needed.

    Args:
        path: Path to write the log file
        content: Log content to write
    """
    # Create parent directories if they don't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write: write to temp file, then rename
    # This prevents partial writes if interrupted
    temp_path = path.with_suffix(".tmp")
    try:
        temp_path.write_text(content, encoding="utf-8")
        temp_path.replace(path)
        logger.debug(f"Successfully wrote log file: {path}")
    except Exception as e:
        logger.error(f"Failed to write log file {path}: {e}")
        # Clean up temp file if it exists
        if temp_path.exists():
            temp_path.unlink()
        raise


def process_jobs(config: Config) -> int:
    """
    Process one cycle of job collection.

    Returns:
        Number of jobs processed in this cycle
    """
    # 1. Fetch all jobs from AAP Mock (with pagination)
    # Note: /api/v2/jobs/ already returns all job details including status
    all_jobs = fetch_all_jobs(config.aap_api_url, config.page_size)
    logger.info(f"Fetched {len(all_jobs)} total jobs from AAP Mock")

    # 2. Filter jobs that need processing:
    #    - Not already written (check in-memory set)
    #    - Have final status (successful or failed, NOT running)
    jobs_to_process = [
        job
        for job in all_jobs
        if job["id"] not in processed_job_ids and job.get("status") in FINAL_STATES
    ]

    logger.info(f"Found {len(jobs_to_process)} new jobs with final status to process")

    processed_count = 0

    # 3. Process each job
    for job in jobs_to_process:
        job_id = job["id"]
        job_status = job.get("status", "unknown")

        try:
            # 4. Fetch job logs from /api/v2/jobs/{id}/stdout/
            logs = fetch_job_logs(config.aap_api_url, job_id)

            # 5. Write to file: /var/log/ansible_logs/{cluster_name}/job-{job_id}.txt
            output_path = (
                Path(config.output_dir) / config.cluster_name / f"job-{job_id}.txt"
            )
            write_log_file(output_path, logs)

            # 6. Mark as processed
            processed_job_ids.add(job_id)
            processed_count += 1

            logger.info(f"Successfully processed job {job_id} (status: {job_status})")

        except Exception as e:
            logger.error(f"Failed to process job {job_id}: {e}", exc_info=True)
            # Continue with next job (don't fail entire batch)
            continue

    return processed_count


def main():
    """Main entry point for the log collector service."""
    # Load configuration from environment variables
    config = Config.from_env()

    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        return 1

    # Setup logging
    setup_logging(config.log_level)

    logger.info("AAP Log Collector starting...")
    logger.info("Configuration:")
    logger.info(f"  AAP API URL: {config.aap_api_url}")
    logger.info(f"  Output directory: {config.output_dir}")
    logger.info(f"  Cluster name: {config.cluster_name}")
    logger.info(f"  Poll interval: {config.poll_interval}s")
    logger.info(f"  Page size: {config.page_size}")
    logger.info(f"  Log level: {config.log_level}")

    # Main polling loop
    while True:
        try:
            logger.debug("Starting new poll cycle")
            processed_count = process_jobs(config)

            if processed_count > 0:
                logger.info(f"Processed {processed_count} jobs in this cycle")
            else:
                logger.debug("No new jobs to process in this cycle")

            # Sleep for poll interval before next cycle
            logger.debug(f"Sleeping for {config.poll_interval} seconds")
            time.sleep(config.poll_interval)

        except KeyboardInterrupt:
            logger.info("Received shutdown signal, exiting...")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            logger.info(
                f"Continuing after error, will retry in {config.poll_interval} seconds"
            )
            time.sleep(config.poll_interval)

    logger.info("AAP Log Collector stopped")
    return 0


if __name__ == "__main__":
    exit(main())
