"""
=============================================================================
🐘 PROJECT ZOGAN — PERSISTENT ALERT DELIVERY QUEUE (PHASE 9)
=============================================================================

This module provides an offline-first, file-backed delivery queue for
remote incident notifications:
  - Persists failed / pending notifications across process restarts
  - Bounded exponential backoff retry scheduling
  - Event deduplication via stable event IDs
  - Thread-safe operations with atomic disk updates
  - Zero memory leaks and configurable bounded queue size
=============================================================================
"""

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import config
from alerts.models import (
    DELIVERY_FAILED,
    DELIVERY_PENDING,
    DELIVERY_RETRYING,
    DELIVERY_SENT,
    AlertEvent,
)


@dataclass
class QueueItem:
    """Represents an alert event awaiting remote dispatch."""

    event: AlertEvent
    enqueued_at: float
    attempts: int
    last_attempt_time: Optional[float]
    next_retry_time: float
    status: str
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event.to_dict(exclude_none=True),
            "enqueued_at": self.enqueued_at,
            "attempts": self.attempts,
            "last_attempt_time": self.last_attempt_time,
            "next_retry_time": self.next_retry_time,
            "status": self.status,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueueItem":
        ev = AlertEvent.from_dict(data.get("event", {}))
        return cls(
            event=ev,
            enqueued_at=data.get("enqueued_at", time.time()),
            attempts=data.get("attempts", 1),
            last_attempt_time=data.get("last_attempt_time"),
            next_retry_time=data.get("next_retry_time", 0.0),
            status=data.get("status", DELIVERY_PENDING),
            error=data.get("error"),
        )


class PersistentAlertQueue:
    """
    File-backed persistent queue for managing alert retries across network outages
    and system restarts.
    """

    def __init__(
        self,
        queue_file: Optional[Union[str, Path]] = None,
        max_size: Optional[int] = None,
        retry_limit: Optional[int] = None,
        base_delay_seconds: Optional[float] = None,
        backoff_factor: Optional[float] = None,
    ):
        q_path = (
            queue_file if queue_file is not None else getattr(config, "ALERT_QUEUE_FILE", "logs/delivery_queue.jsonl")
        )
        self.queue_file = Path(q_path)
        self.max_size = max_size if max_size is not None else getattr(config, "ALERT_QUEUE_MAX_SIZE", 100)
        self.retry_limit = retry_limit if retry_limit is not None else getattr(config, "ALERT_RETRY_LIMIT", 3)
        self.base_delay = (
            base_delay_seconds
            if base_delay_seconds is not None
            else getattr(config, "ALERT_RETRY_BASE_DELAY_SECONDS", 2.0)
        )
        self.backoff_factor = (
            backoff_factor if backoff_factor is not None else getattr(config, "ALERT_RETRY_BACKOFF_FACTOR", 2.0)
        )

        self._lock = threading.RLock()
        self._items: Dict[str, QueueItem] = {}
        self._load()

    def _ensure_dir(self) -> None:
        try:
            self.queue_file.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    def _load(self) -> None:
        """Loads uncompleted pending items from the queue file."""
        with self._lock:
            self._items.clear()
            if not self.queue_file.exists():
                return

            try:
                with open(self.queue_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line_str = line.strip()
                        if not line_str:
                            continue
                        try:
                            data = json.loads(line_str)
                            item = QueueItem.from_dict(data)
                            self._items[item.event.event_id] = item
                        except Exception:
                            continue
            except Exception as e:
                print(f"[QUEUE WARNING] Failed to load offline queue from {self.queue_file}: {e}")

    def _save(self) -> None:
        """Atomically persists queue items to disk."""
        with self._lock:
            self._ensure_dir()
            tmp_path = self.queue_file.with_suffix(".tmp")
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    for item in self._items.values():
                        f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
                if tmp_path.exists():
                    # Atomic replace
                    os.replace(tmp_path, self.queue_file)
            except Exception as e:
                print(f"[QUEUE ERROR] Failed to save offline queue to {self.queue_file}: {e}")
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass

    def enqueue(
        self,
        event: AlertEvent,
        error: Optional[str] = None,
        now: Optional[float] = None,
    ) -> QueueItem:
        """
        Enqueues an alert event for deferred delivery or retry.
        Guarantees deduplication: if the event_id is already present,
        updates the existing item rather than creating duplicate entries.
        """
        current_time = time.time() if now is None else now

        with self._lock:
            ev_id = event.event_id
            if ev_id in self._items and self._items[ev_id].status not in (DELIVERY_SENT, DELIVERY_FAILED):
                # Update existing pending item without creating duplicate
                item = self._items[ev_id]
                item.error = error or item.error
                item.last_attempt_time = current_time
                self._save()
                return item

            # Calculate initial retry delay
            initial_delay = self.base_delay
            item = QueueItem(
                event=event,
                enqueued_at=current_time,
                attempts=max(1, event.delivery_attempts or 1),
                last_attempt_time=current_time,
                next_retry_time=current_time + initial_delay,
                status=DELIVERY_PENDING,
                error=error,
            )

            # Update event delivery metadata
            event.delivery_status = item.status
            event.delivery_attempts = item.attempts
            event.last_delivery_attempt = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(current_time))
            event.delivery_error = error

            # Check queue capacity bounds
            if len(self._items) >= self.max_size:
                self._prune_excess()

            self._items[ev_id] = item
            self._save()
            return item

    def _prune_excess(self) -> None:
        """Prunes completed or oldest items to keep queue size within bounds."""
        with self._lock:
            # 1. Prune already SENT or FAILED items first
            completed_ids = [
                eid for eid, item in self._items.items() if item.status in (DELIVERY_SENT, DELIVERY_FAILED)
            ]
            for eid in completed_ids:
                del self._items[eid]
                if len(self._items) < self.max_size:
                    return

            # 2. If still full, prune oldest items
            sorted_items = sorted(self._items.items(), key=lambda kv: kv[1].enqueued_at)
            overflow = len(self._items) - self.max_size + 1
            for eid, _ in sorted_items[:overflow]:
                del self._items[eid]

    def get(self, event_id: str) -> Optional[QueueItem]:
        """Retrieves a queue item by its event_id."""
        with self._lock:
            return self._items.get(event_id)

    @property
    def items(self) -> Dict[str, QueueItem]:
        """Returns a snapshot of the current queue items mapping."""
        with self._lock:
            return dict(self._items)

    def calculate_next_retry(self, attempts: int, now: Optional[float] = None) -> float:
        """Calculates next retry timestamp based on exponential backoff policy."""
        current_time = time.time() if now is None else now
        delay = self.base_delay * (self.backoff_factor ** max(0, attempts - 1))
        return current_time + delay

    def get_pending(self) -> List[QueueItem]:
        """Returns all items currently awaiting delivery."""
        with self._lock:
            return [item for item in self._items.values() if item.status in (DELIVERY_PENDING, DELIVERY_RETRYING)]

    def get_ready_for_retry(self, now: Optional[float] = None) -> List[QueueItem]:
        """Returns pending items whose backoff delay has expired and are ready for retry."""
        current_time = time.time() if now is None else now
        with self._lock:
            return [
                item
                for item in self._items.values()
                if item.status in (DELIVERY_PENDING, DELIVERY_RETRYING) and item.next_retry_time <= current_time
            ]

    def mark_sent(self, event_id: str, now: Optional[float] = None) -> None:
        """Marks a queue item as successfully sent."""
        current_time = time.time() if now is None else now
        with self._lock:
            if event_id in self._items:
                item = self._items[event_id]
                item.status = DELIVERY_SENT
                item.last_attempt_time = current_time
                item.error = None
                item.event.delivery_status = DELIVERY_SENT
                item.event.delivery_error = None
                self._save()

    def mark_attempt_failed(
        self,
        event_id: str,
        error: str,
        now: Optional[float] = None,
    ) -> None:
        """
        Records a failed delivery attempt and schedules the next exponential backoff retry.
        If retry_limit is reached, transitions the item to FAILED.
        """
        current_time = time.time() if now is None else now
        with self._lock:
            if event_id in self._items:
                item = self._items[event_id]
                item.attempts += 1
                item.last_attempt_time = current_time
                item.error = error
                item.event.delivery_attempts = item.attempts
                item.event.delivery_error = error
                item.event.last_delivery_attempt = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(current_time))

                if item.attempts >= self.retry_limit:
                    item.status = DELIVERY_FAILED
                    item.event.delivery_status = DELIVERY_FAILED
                else:
                    item.status = DELIVERY_RETRYING
                    item.event.delivery_status = DELIVERY_RETRYING
                    # Exponential backoff: base_delay * (factor ^ (attempts - 1))
                    backoff_delay = self.base_delay * (self.backoff_factor ** (item.attempts - 1))
                    item.next_retry_time = current_time + backoff_delay

                self._save()

    @property
    def pending_count(self) -> int:
        """Count of items awaiting delivery or retry."""
        with self._lock:
            return sum(1 for item in self._items.values() if item.status in (DELIVERY_PENDING, DELIVERY_RETRYING))

    @property
    def failed_count(self) -> int:
        """Count of items permanently failed after exhausting retry limit."""
        with self._lock:
            return sum(1 for item in self._items.values() if item.status == DELIVERY_FAILED)

    @property
    def sent_count(self) -> int:
        """Count of items successfully delivered."""
        with self._lock:
            return sum(1 for item in self._items.values() if item.status == DELIVERY_SENT)

    @property
    def total_count(self) -> int:
        """Total items currently in queue index."""
        with self._lock:
            return len(self._items)

    def purge_completed(self) -> int:
        """Removes all SENT items from the queue and compacts the disk file."""
        with self._lock:
            to_remove = [eid for eid, item in self._items.items() if item.status == DELIVERY_SENT]
            for eid in to_remove:
                del self._items[eid]
            if to_remove:
                self._save()
            return len(to_remove)

    def clear(self) -> None:
        """Clears all in-memory items and empties the persistent file."""
        with self._lock:
            self._items.clear()
            self._save()
