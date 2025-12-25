import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from alm.models import GrafanaAlert, LogLabels, DetectedLevel
from alm.patterns.ingestion import (
    TESTING_LOG_ERROR,
    TESTING_LOG_FATAL,
    TESTING_LOG_FAILED,
)
from alm.utils.logger import get_logger

logger = get_logger(__name__)


def _filter_matches_end_with_ignoring(matches: list[re.Match]) -> list[re.Match]:
    """Filter matches that end with 'ignoring'."""
    return [
        match
        for match in matches
        if not match.groupdict().get("logmessage", "").endswith("ignoring")
    ]


def shrink_long_logs(log: str) -> str:
    """Shrink long logs."""
    # used for very long logs that have a lot of redundent data
    # This should be improve later to be more accurate
    # TODO: improve this
    # NOTE: see the 20_000 char trim
    return log[:5_000]


def load_alert_from_filesystem(path: str) -> Optional[GrafanaAlert]:
    """Mock the Grafana alerting system."""
    with open(path, "r") as file:
        content = file.read()
    matches = _filter_matches_end_with_ignoring(
        list(re.finditer(TESTING_LOG_ERROR, content, re.MULTILINE))
    )

    if not matches:
        matches = _filter_matches_end_with_ignoring(
            re.finditer(TESTING_LOG_FATAL, content, re.MULTILINE)
        )
        if not matches:
            if not matches:
                matches = _filter_matches_end_with_ignoring(
                    re.finditer(TESTING_LOG_FAILED, content, re.MULTILINE)
                )
            return None

    # Get the last match
    last_match = matches[-1]
    groups = last_match.groupdict()

    # Create GrafanaAlert instance with extracted data
    alert = GrafanaAlert(
        timestamp=(
            datetime.strptime(groups.get("timestamp"), "%A %d %B %Y  %H:%M:%S")
            if groups.get("timestamp")
            else datetime.now()
        ),
        logMessage=shrink_long_logs(
            groups.get("logmessage", "")
        ),  # Full matched text as the log message
        log_labels=LogLabels(
            detected_level=DetectedLevel.ERROR
            if groups.get("status", "error") == "error"
            else DetectedLevel.WARN,
            filename=Path(path).name,
            job=groups.get("job", "").strip(),
            service_name=groups.get("host", "").strip(),
        ).model_dump(),
    )

    return alert


def ingest_alerts(directory: str) -> list[GrafanaAlert]:
    """Ingest alerts from a directory."""
    alerts = []
    error_count = 0
    success_count = 0
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            try:
                alerts.append(load_alert_from_filesystem(os.path.join(directory, file)))
                success_count += 1
            except Exception:
                error_count += 1
    logger.info("%d errors and %d successes", error_count, success_count)
    logger.info("alerts: %d", len(alerts))
    return alerts
