from abc import ABC, abstractmethod
import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Dict
import os

import httpx
from alm.ingestion.transformations import filter_ingoring, pre_proccess_log
from alm.models import LogLabels, LogEntry
from alm.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader(ABC):
    @abstractmethod
    async def _load(self) -> List[Dict]:
        """Load raw data from the source."""
        pass

    @abstractmethod
    def _transform(self, raw_data: List[Dict]) -> List[LogEntry]:
        """Transform raw data into LogEntry objects."""
        pass

    async def load_and_transform(self) -> List[LogEntry]:
        """Load and transform data in a single operation."""
        return self._transform(await self._load())


class LokiDataLoader(DataLoader):
    def __init__(
        self,
        query: str = '{status=~"fatal|failed"}',
        start_time: datetime = datetime.now() - timedelta(hours=1),
        end_time: datetime = datetime.now(),
        limit: int = 2000,
        max_retries: int = 12 * 2,
    ):
        self.query = query
        self.end_time = end_time
        self.start_time = start_time
        self.limit = limit
        self.max_retries = max_retries

    async def _load(self) -> Dict:
        """
        Query Loki database for logs matching the configured LogQL query.

        Returns:
            Dictionary containing the raw query results from Loki.
        """
        # Convert to nanosecond timestamps (Loki uses nanoseconds)
        start_ns = int(self.start_time.timestamp() * 1e9)
        end_ns = int(self.end_time.timestamp() * 1e9)

        endpoint = f"{os.getenv('LOKI_URL')}/loki/api/v1/query_range"

        params = {
            "query": self.query,
            "start": start_ns,
            "end": end_ns,
            "limit": self.limit,
        }
        timeout = httpx.Timeout(30.0, connect=10.0)

        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.get(endpoint, params=params)
                    response.raise_for_status()
                return response.json()
            except (
                httpx.TimeoutException,
                httpx.RequestError,
                httpx.HTTPStatusError,
            ) as e:
                last_error = e
                logger.warning(
                    "Loki request failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries + 1,
                    e,
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(5 * attempt)
            except Exception as e:
                last_error = e
                logger.error(
                    "Loki request failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries + 1,
                    e,
                )
                raise e
        raise last_error  # type: ignore

    def _transform(self, raw_data: Dict) -> List[LogEntry]:
        """
        Transform raw Loki response into LogEntry objects.

        Args:
            raw_data: Raw response dictionary from Loki API.

        Returns:
            List of LogEntry objects.
        """
        if raw_data.get("status") != "success":
            return []

        streams = raw_data.get("data", {}).get("result", [])
        if not streams:
            return []

        log_entries = []
        for stream in streams:
            labels = stream.get("stream", {})
            values = stream.get("values", [])

            if not values:
                continue

            database_timestamp_str, log_line = values[0]

            if filter_ingoring(
                log_line
            ):  # TODO remoe me after filtering it right in the ingestion
                continue
            timestamp = (
                datetime.strptime(
                    labels.get("real_timestamp"), "%A %d %B %Y  %H:%M:%S %z"
                ).replace(tzinfo=None)
                if labels.get("real_timestamp")
                else None
            )
            database_timestamp = datetime.fromtimestamp(
                int(database_timestamp_str) / 1e9, tz=timezone.utc
            )

            labels["database_timestamp"] = database_timestamp

            log_labels = LogLabels(**labels)
            log_entry = LogEntry(
                timestamp=timestamp,
                log_labels=log_labels,
                message=pre_proccess_log(log_line),
            )
            log_entries.append(log_entry)

        return log_entries
