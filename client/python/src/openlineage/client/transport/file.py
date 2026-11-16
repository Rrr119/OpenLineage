# Copyright 2018-2026 contributors to the OpenLineage project
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import io
import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse
from urllib.request import url2pathname

from openlineage.client.serde import Serde
from openlineage.client.transport.transport import Config, Transport
from openlineage.client.utils import config_value_to_str, import_from_string

if TYPE_CHECKING:
    from openlineage.client.client import Event

log = logging.getLogger(__name__)

# Try to import fsspec for file-like transport support
try:
    import fsspec  # type: ignore[import-not-found]

    FSSPEC_AVAILABLE = True
except ImportError:
    FSSPEC_AVAILABLE = False


@dataclass
class BatchConfig:
    """Configuration for buffering and batching events before writing them.

    Batching is disabled by default to preserve the historical one-write-per-emit
    behavior. When enabled, events are buffered in memory and flushed together as a
    single newline-delimited JSON payload.

    Attributes:
        enabled (bool): Whether batching is enabled (default: False).
        max_events (int): Maximum number of events buffered before an automatic flush
            (default: 100). A value <= 1 flushes on every emitted event.
        max_wait_millis (int): Maximum age, in milliseconds, of the oldest buffered event
            before an automatic background flush (default: 1000). A value <= 0 disables
            time-based flushing, leaving size-based flushing and explicit ``close`` only.
    """

    enabled: bool = False
    max_events: int = 100
    max_wait_millis: int = 1000

    @classmethod
    def from_dict(cls, params: dict[str, Any] | None) -> BatchConfig:
        params = params or {}
        return cls(
            enabled=bool(params.get("enabled", False)),
            max_events=int(params.get("max_events", 100)),
            max_wait_millis=int(params.get("max_wait_millis", 1000)),
        )


@dataclass
class RetryConfig:
    """Configuration for retrying failed file/object-store writes.

    Writes can fail transiently, especially against remote object stores accessed through
    fsspec. Failed writes are retried with exponential backoff. Events that still fail
    after all attempts are logged and dropped rather than propagated to the caller, so
    lineage collection never breaks the instrumented job.

    Attributes:
        enabled (bool): Whether retries are enabled (default: True).
        total (int): Maximum number of retry attempts after the initial write (default: 3).
        backoff_factor (float): Base backoff in seconds; the delay before attempt ``n`` is
            ``backoff_factor * 2 ** n`` with a small random jitter (default: 0.1).
        max_backoff (float): Upper bound, in seconds, for a single backoff sleep
            (default: 5.0).
    """

    enabled: bool = True
    total: int = 3
    backoff_factor: float = 0.1
    max_backoff: float = 5.0

    @classmethod
    def from_dict(cls, params: dict[str, Any] | None) -> RetryConfig:
        params = params or {}
        return cls(
            enabled=bool(params.get("enabled", True)),
            total=int(params.get("total", 3)),
            backoff_factor=float(params.get("backoff_factor", 0.1)),
            max_backoff=float(params.get("max_backoff", 5.0)),
        )


@dataclass
class FileConfig(Config):
    """Configuration for file-based OpenLineage event transport.

    Supports local files and remote filesystems via fsspec (optional dependency).
    The transport auto-detects protocols from URL schemes (s3://, gs://, etc.).

    Attributes:
        log_file_path (str): Path to the log file. Can be local or URL with protocol.
        append (bool): Whether to append to existing file (default: False, creates timestamped files).
            Note: Many cloud/object stores (e.g. GCS) do not support true append mode via fsspec.
            In such cases, opening with append may raise an error or be emulated by rewriting the object.
        storage_options (dict): Options passed to the filesystem (e.g., credentials).
        filesystem (str): Optional filesystem provider specification:
            - Class path (e.g., "s3fs.S3FileSystem")
            - Factory function path (e.g., "mymodule.get_filesystem")
            - Instance attribute path (e.g., "mymodule.fs_instance")
        fs_kwargs (dict): Keyword arguments for constructing/calling the filesystem provider.
        debug_mode (bool): Enable debug mode (default: False). Only effective when append=False. When enabled:
            - Filenames include short timestamp (2-digit year), job/dataset name sanitized, and event type.
              For RunEvent: uses job name and event type (e.g., START, COMPLETE).
              For JobEvent: uses job name with "unknown_event_type" as event type.
              For DatasetEvent: uses dataset name with "unknown_event_type" as event type.
              Example: "log_file_path/ol-250115T143052.123456-my_job.name-START.json"
            - JSON output is prettified with indentation for easier reading
        batch (BatchConfig): Optional batching configuration. When enabled, events are buffered
            and written together to reduce the number of file/object-store operations.
        retry (RetryConfig): Optional retry configuration applied to every write attempt.
    """

    log_file_path: str
    append: bool = False
    storage_options: dict[str, Any] | None = None
    # Unified: explicit filesystem provider (class, callable factory, or instance attribute),
    # dotted import path
    filesystem: str | None = None
    # Keyword arguments for constructing/calling the filesystem provider
    fs_kwargs: dict[str, Any] | None = None
    debug_mode: bool = False
    batch: BatchConfig | None = None
    retry: RetryConfig | None = None

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> FileConfig:
        if "log_file_path" not in params:
            msg = "`log_file_path` key not passed to FileConfig"
            raise RuntimeError(msg)

        config = cls(
            log_file_path=config_value_to_str(params["log_file_path"]),
            append=params.get("append", False),
            storage_options=params.get("storage_options"),
            filesystem=params.get("filesystem"),
            fs_kwargs=params.get("fs_kwargs"),
            debug_mode=params.get("debug_mode", False),
            batch=BatchConfig.from_dict(params.get("batch")),
            retry=RetryConfig.from_dict(params.get("retry")),
        )

        # Validate configuration
        config._validate()
        return config

    def _validate(self) -> None:
        """Validate configuration and check fsspec availability if needed."""
        # Check if fsspec features are requested but not available
        if self._needs_fsspec() and not FSSPEC_AVAILABLE:
            msg = (
                "An explicit filesystem was requested or remote protocol detected but fsspec is not "
                "available. Please install fsspec to use remote filesystems."
            )
            raise RuntimeError(msg)

        # Warn about conflicting configurations
        if self.filesystem and self.storage_options:
            log.warning("'storage_options' ignored when using explicit filesystem. Use fs_kwargs instead.")

    def _needs_fsspec(self) -> bool:
        """Determine if fsspec is needed based on configuration."""
        # Need fsspec if explicit filesystem is configured
        if self.filesystem:
            return True

        # Need fsspec if the path contains a remote protocol
        return "://" in self.log_file_path and not self.log_file_path.startswith("file://")


class FileHandler(ABC):
    """Abstract base class for file handling implementations."""

    @abstractmethod
    def open_file(self, file_path: str, mode: str) -> Any:
        """Open a file-like handle."""
        pass


class LocalFileHandler(FileHandler):
    """Handler for local filesystem operations."""

    @staticmethod
    def _to_local_path(file_path: str) -> str:
        parsed = urlparse(file_path)
        if parsed.scheme != "file":
            return file_path

        if parsed.netloc and parsed.netloc != "localhost":
            return url2pathname(f"//{parsed.netloc}{parsed.path}")
        return url2pathname(parsed.path)

    def open_file(self, file_path: str, mode: str) -> Any:
        """Open a file using built-in open for local filesystem."""
        return open(self._to_local_path(file_path), mode)


class FsspecFileHandler(FileHandler):
    """Handler for fsspec-based filesystem operations."""

    def __init__(self, config: FileConfig) -> None:
        self.config = config
        self._fs = self._create_filesystem()

    def _create_filesystem(self) -> Any | None:
        """Create and return an fsspec filesystem instance if configured, else None."""
        if self.config.filesystem:
            target = import_from_string(self.config.filesystem)
            kwargs = self.config.fs_kwargs or {}
            # If it's a class, instantiate. If it's a callable factory, call it. If it's an instance, use it.
            if inspect.isclass(target):
                fs = target(**kwargs)
            elif callable(target):
                fs = target(**kwargs)
            else:
                fs = target
            if not hasattr(fs, "open"):
                raise RuntimeError(
                    f"'{self.config.filesystem}' did not produce a filesystem implementing an 'open' method"
                )
            return fs
        # Fall back to protocol-based lazy opening (handled in open_file)
        return None

    def open_file(self, file_path: str, mode: str) -> Any:
        """Open a file-like handle using fsspec.

        - If an explicit filesystem instance is configured, use its .open.
        - Else use fsspec.open with auto-detection from URL.
        """
        if self._fs is not None:
            return self._fs.open(file_path, mode)

        storage_options = self.config.storage_options or {}
        try:
            # Let fsspec auto-detect protocol from URL (e.g., s3://, gs://, etc.)
            return fsspec.open(file_path, mode=mode, **storage_options)
        except (ValueError, ImportError) as e:
            # If fsspec can't handle the URL (e.g., local path with no protocol),
            # fall back to built-in open
            log.debug("fsspec could not handle path '%s', falling back to built-in open: %s", file_path, e)
            return open(file_path, mode)


class FileTransport(Transport):
    """File-based transport for OpenLineage events.

    Writes events to local or remote files. Supports automatic protocol detection
    for remote filesystems via fsspec (s3://, gs://, az://, etc.).

    Three filesystem configuration approaches:
    1. Auto-detection: Specify only log_file_path with protocol URL
    2. Explicit filesystem: Use 'filesystem' parameter with class/factory/instance
    3. Storage options: Use 'storage_options' for credentials and configuration

    Optionally, events can be buffered and written in batches (see ``BatchConfig``) and
    transient write failures can be retried with exponential backoff (see ``RetryConfig``).
    """

    kind = "file"
    config_class = FileConfig

    def __init__(self, config: FileConfig) -> None:
        self.config = config
        self._file_handler = self._create_file_handler()
        self._buffer: list[tuple[Event, int]] = []
        self._buffer_lock = threading.RLock()
        self._buffer_event = threading.Event()
        self._closed = False
        self._worker: threading.Thread | None = None
        self._dropped_events = 0
        if self.config.batch is None:
            self.config.batch = BatchConfig()
        if self.config.retry is None:
            self.config.retry = RetryConfig()
        if self.config.batch.enabled and self.config.batch.max_wait_millis > 0:
            self._worker = threading.Thread(
                target=self._flush_loop, name="OpenLineageFileTransport", daemon=True
            )
            self._worker.start()
        log.debug(
            "Constructing OpenLineage transport that will send events "
            "to file(s) using the following config: %s",
            self.config,
        )

    def _flush_loop(self) -> None:
        """Background loop that flushes buffered events once they are old enough."""
        wait_seconds = self.config.batch.max_wait_millis / 1000.0
        while not self._closed:
            self._buffer_event.wait(timeout=wait_seconds)
            self._buffer_event.clear()
            if self._closed:
                break
            with self._buffer_lock:
                if not self._buffer:
                    continue
                oldest_age = (time.monotonic_ns() - self._buffer[0][1]) / 1_000_000
                if oldest_age >= self.config.batch.max_wait_millis:
                    self._flush_locked()

    @staticmethod
    def _serialize(event: Event, debug_mode: bool) -> str:
        kwargs = {"indent": 2} if debug_mode else {}
        return Serde.to_json(event, **kwargs)

    def _write_with_retry(self, file_path: str, mode: str, payload: str) -> None:
        """Write a payload to the target, retrying transient failures with backoff."""
        retry = self.config.retry
        attempts = retry.total + 1 if retry.enabled else 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                with self._file_handler.open_file(file_path, mode) as handle:
                    handle.write(payload)
                return
            except (PermissionError, io.UnsupportedOperation) as error:
                # Wrong mode or missing permissions are not transient: do not retry.
                msg = f"Log file `{file_path}` is not writeable"
                raise RuntimeError(msg) from error
            except Exception as error:
                last_error = error
                if attempt == attempts - 1:
                    break
                sleep_seconds = min(retry.max_backoff, retry.backoff_factor * (2**attempt))
                sleep_seconds *= 1 + random.uniform(0, 0.1)
                log.warning(
                    "Failed to write to log file `%s` (attempt %d/%d): %s. Retrying in %.3fs",
                    file_path,
                    attempt + 1,
                    attempts,
                    error,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
        msg = f"Failed to write to log file `{file_path}`: {last_error}"
        raise RuntimeError(msg)

    def _flush_locked(self) -> None:
        """Write all currently buffered events. Caller must hold ``_buffer_lock``."""
        if not self._buffer:
            return
        events = [event for event, _ in self._buffer]
        self._buffer = []
        if self.config.append:
            try:
                payload = "".join(f"{self._serialize(event, self.config.debug_mode)}\n" for event in events)
                self._write_with_retry(self.config.log_file_path, "a", payload)
                return
            except Exception:
                log.exception(
                    "Dropping %d OpenLineage event(s) that could not be written to `%s`",
                    len(events),
                    self.config.log_file_path,
                )
                self._dropped_events += len(events)
                return

        for event in events:
            try:
                file_path = self._get_file_path(event)
                payload = f"{self._serialize(event, self.config.debug_mode)}\n"
                self._write_with_retry(file_path, "w", payload)
            except Exception:
                self._dropped_events += 1
                log.exception(
                    "Dropping an OpenLineage event that could not be written to `%s`",
                    self.config.log_file_path,
                )

    def flush(self) -> None:
        """Force a write of all buffered events. No-op when batching is disabled."""
        with self._buffer_lock:
            self._flush_locked()
        self._buffer_event.set()

    def _create_file_handler(self) -> FileHandler:
        """Create the appropriate file handler based on configuration and availability."""
        if self.config._needs_fsspec() and FSSPEC_AVAILABLE:
            return FsspecFileHandler(self.config)
        else:
            return LocalFileHandler()

    def _get_file_path(self, event: Event) -> str:
        """Get the file path, adding timestamp if not in append mode."""
        if self.config.append:
            return self.config.log_file_path
        if not self.config.debug_mode:
            time_str = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
            return f"{self.config.log_file_path}-{time_str}.json"

        # Debug mode
        time_str = datetime.now().strftime("%y%m%dT%H%M%S.%f")

        # Extract job name if present
        job_name = "unknown_job_name"
        if hasattr(event, "job") and hasattr(event.job, "name"):
            job_name = event.job.name
        elif hasattr(event, "dataset") and hasattr(event.dataset, "name"):
            # DatasetEvent doesn't have job, use dataset name
            job_name = event.dataset.name

        # Extract event type if present (only RunEvent has it)
        event_type = "unknown_event_type"
        if hasattr(event, "eventType"):
            event_type = event.eventType.value if hasattr(event.eventType, "value") else str(event.eventType)  # type: ignore[union-attr]

        # Sanitize names for filesystem (replace problematic characters)
        job_name_safe = "".join(c if c.isalnum() or c in ("_", ".", "-") else "__" for c in job_name)

        return f"{self.config.log_file_path}-{time_str}-{job_name_safe}-{event_type}.json"

    def emit(self, event: Event) -> None:
        if self._closed:
            log.warning("FileTransport is closed; dropping emitted event")
            return

        if not self.config.batch.enabled:
            log_file_path = self._get_file_path(event)
            mode = "a" if self.config.append else "w"
            log.debug("Openlineage event will be emitted to file: `%s`", log_file_path)
            self._write_with_retry(log_file_path, mode, f"{self._serialize(event, self.config.debug_mode)}\n")
            return

        with self._buffer_lock:
            if self._closed:
                log.warning("FileTransport is closed; dropping emitted event")
                return
            self._buffer.append((event, time.monotonic_ns()))
            should_flush = len(self._buffer) >= max(1, self.config.batch.max_events)
            if should_flush:
                self._flush_locked()
        if should_flush:
            self._buffer_event.set()
        else:
            log.debug("Openlineage event buffered for batched file emission")

    def close(self, timeout: float = -1) -> bool:
        """Flush buffered events and stop the background flush thread.

        Returns False when one or more events were dropped because their writes failed
        even after retries, True otherwise.
        """
        if self._closed:
            return self._dropped_events == 0
        with self._buffer_lock:
            self._closed = True
            self._flush_locked()
        self._buffer_event.set()
        if self._worker is not None:
            self._worker.join(timeout=None if timeout < 0 else timeout)
        return self._dropped_events == 0

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            log.debug("Error while closing FileTransport", exc_info=True)
