import pandas as pd
from alm.models import DetectedLevel
import re
from alm.utils.logger import get_logger

logger = get_logger(__name__)


def select_error_logs(multi_line_logs: str) -> list[str]:
    lines = pd.Series(multi_line_logs.split("\n\n"))
    lines = lines[
        lines.str.contains(
            r"fatal: \[[^\]]+\]: |error: \[[^\]]+\]: |failed: \[[^\]]+\] "
        )
    ]
    lines = lines[~lines.str.contains(r"...ignoring")]
    return lines.tolist()


def detect_error_level(log: str) -> DetectedLevel:
    status_regex = r"(?P<status>failed|fatal|error):\s+\["
    match = re.search(status_regex, log)
    if match:
        return DetectedLevel(match.group("status"))
    return DetectedLevel.UNKNOWN


def get_log_message(log: str) -> str:
    regex = r"(fatal|error|failed): \[(?P<host>[^\]]+)\]:? (([A-Z]+!)|(\(.*\))) => \{(?P<logmessage>[\s\S]*?)\}"
    # regex = r'(fatal|error|failed): \[(?P<host>[^\]]+)\]:? ([A-Z]+!)|(\(.*\)) => \{(?P<logmessage>[\s\S]*)\}'
    #  regex = r'(fatal|error|failed): \[(?P<host>[^\]]+)\]:? FAILED! => \{(?P<logmessage>[\s\S]*)\}'
    matches = list(re.finditer(regex, log))
    match = matches[-1] if matches else None
    if match:
        logmessage = match.group("logmessage")
        if logmessage is None:
            logger.error(f"Failed to get log message: {log}")
        return logmessage
    logger.error(f"Failed to fix dictionary in log line: {log}")
    return log


def slice_log_message(log: str) -> str:
    try:
        return log.strip()[:5_000]
    except Exception as e:
        logger.error(f"Failed to slice log message: {e}")
        return log


def filter_ingoring(log: str) -> bool:  # TODO remove me when db filtering is working
    if "...ignoring" in log:
        return True
    return False
