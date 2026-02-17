"""Configuration management for AAP Log Collector."""

import os
import logging
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for the log collector service."""

    aap_api_url: str
    output_dir: str
    cluster_name: str
    poll_interval: int
    log_level: str
    page_size: int = 100  # Number of jobs to fetch per API page

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            aap_api_url=os.getenv("AAP_API_URL", "http://alm-aap-mock:8080"),
            output_dir=os.getenv("OUTPUT_DIR", "/var/log/ansible_logs"),
            cluster_name=os.getenv("CLUSTER_NAME", "default-cluster"),
            poll_interval=int(os.getenv("POLL_INTERVAL", "300")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            page_size=int(os.getenv("PAGE_SIZE", "100")),
        )

    def validate(self):
        """Validate configuration values."""
        if not self.aap_api_url:
            raise ValueError("AAP_API_URL is required")

        if not self.output_dir:
            raise ValueError("OUTPUT_DIR is required")

        if not self.cluster_name:
            raise ValueError("CLUSTER_NAME is required")

        if self.poll_interval <= 0:
            raise ValueError("POLL_INTERVAL must be positive")

        if self.page_size < 1 or self.page_size > 200:
            raise ValueError("PAGE_SIZE must be between 1 and 200")

        # Validate log level
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of: {', '.join(valid_levels)}")


def setup_logging(log_level: str):
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
