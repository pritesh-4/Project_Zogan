# Project Zogan — Phase 9: Offline-First Operation & Failure Recovery

This document details the architectural reliability layer implemented in Phase 9 for **Project Zogan**.

The core objective is to ensure that temporary hardware, network, software, or remote API failures do not disrupt monitoring or result in lost incident data.

---

## 1. Fundamental Principle: Local-First Event Persistence

> **LOCAL EVENT DATA MUST NEVER DEPEND ON REMOTE NETWORK AVAILABILITY.**

When a confirmed elephant encounter occurs:
1. **Create the Event**: A structured `AlertEvent` is constructed with spatial risk analysis, track telemetry, and a stable UUID (`event_id`).
2. **Persist Locally First**: The event is synchronously written to local disk storage (`logs/alerts.jsonl`) via atomic append.
3. **Attempt Remote Notification**: Only after local storage is guaranteed does the system attempt remote notifications (such as Telegram).

```
   Detection & Tracking
           ↓
   Risk Assessment Engine
           ↓
      Event Creation
           ↓
[1] LOCAL PERSISTENCE FIRST ──> logs/alerts.jsonl (Zero Network Dependency)
           ↓
[2] REMOTE TELEGRAM ATTEMPT ──> Timeout: 3.0s (Non-blocking)
           ↓
       (Failure?)
           ↓
[3] PERSISTENT OFFLINE QUEUE ──> logs/delivery_queue.jsonl
           ↓
[4] BOUNDED RETRY WORKER   ──> Exponential Backoff (1s, 2s, 4s...)
```

If the internet is severed, Telegram servers are down, or API limits are exhausted:
- **Zero events are lost.**
- The incident is permanently recorded locally.
- The event is scheduled in the offline queue for deferred delivery.

---

## 2. Alert Delivery State Model

Delivery state is tracked separately from threat assessment. An event being `HIGH RISK` is completely orthogonal to whether remote notification succeeded.

The following delivery states are supported on `AlertEvent`:

| State | Description |
|---|---|
| `PENDING` | Created and persisted locally; awaiting initial remote delivery or enqueued in offline queue. |
| `SENT` | Successfully delivered to remote endpoint (e.g. Telegram confirmed HTTP 200). |
| `RETRYING` | Remote delivery attempt failed; awaiting next scheduled backoff retry. |
| `FAILED` | Retry limit exhausted without successful delivery; marked permanently failed. |
| `NOT_CONFIGURED` | Remote service is not configured (e.g. missing environment credentials). |
| `SKIPPED` | Event risk level is below remote notification threshold (e.g. `LOW` or `MEDIUM`). |

Each `AlertEvent` maintains:
- `delivery_status`: One of the states above.
- `delivery_attempts`: Count of attempted deliveries.
- `last_delivery_attempt`: ISO-8601 timestamp of most recent attempt.
- `delivery_error`: Sanitized error message if delivery failed.

---

## 3. Persistent Offline Delivery Queue

The offline queue (`alerts.queue.PersistentAlertQueue`) manages deferred deliveries:

- **Disk Persistence**: Stored as JSON Lines in `logs/delivery_queue.jsonl`.
- **Process Restart Survival**: Survives process crashes and restarts. When Zogan starts up, any pending items are restored into memory.
- **Thread Safety**: Synchronized using a reentrant lock (`threading.RLock`) with atomic temporary file swapping (`.tmp` -> `os.replace`).
- **Stable Identity Deduplication**: Events are indexed by their unique `event_id`. Re-enqueuing an existing event updates its error and attempt metadata without generating duplicate queue entries (`EVENT-001`, `EVENT-002`).
- **Bounded Capacity**: Automatically enforces `ALERT_QUEUE_MAX_SIZE` (default: 100), pruning completed (`SENT`/`FAILED`) items first, followed by oldest entries to prevent unbounded disk growth.

---

## 4. Bounded Retry Policy with Exponential Backoff

To prevent infinite loops, busy-waiting, and API flooding, remote retries adhere to a strictly bounded exponential backoff schedule:

- **Formula**:
  $$\text{Delay} = \text{base\_delay} \times (\text{backoff\_factor})^{(\text{attempts} - 1)}$$
- **Defaults**:
  - `ALERT_RETRY_LIMIT = 3`
  - `ALERT_RETRY_BASE_DELAY_SECONDS = 2.0`
  - `ALERT_RETRY_BACKOFF_FACTOR = 2.0`
- **Schedule**:
  - Attempt 1 failure $\to$ Next retry in 2.0s
  - Attempt 2 failure $\to$ Next retry in 4.0s
  - Attempt 3 failure $\to$ Marked `FAILED` (bounded, no further retries)

---

## 5. Non-Blocking Execution Guarantee

Remote alert delivery must **never** stall the video detection loop. Waiting 10 seconds for a Telegram request would cause hundreds of camera frames to be dropped, compromising safety.

Zogan guarantees non-blocking operation via two mechanisms:
1. **Fast Delivery Timeouts**: Synchronous network calls are capped at `ALERT_DELIVERY_TIMEOUT_SECONDS = 3.0`.
2. **Asynchronous Dispatch Worker**: In `scripts/run_camera.py`, alert dispatches use `async_delivery=True`, submitting remote network requests to a background `ThreadPoolExecutor`. The video frame capture and risk processing continue immediately without interruption.
3. **Background Queue Draining**: Queue retries are checked periodically (every 30 frames) and executed asynchronously via `drain_delivery_queue_async()`.

---

## 6. Camera Input Recovery

Live video capture is prone to temporary USB disconnections, network RTSP drops, or driver resets:

1. **Failure Detection**: When `camera.read()` returns `False`, the stream is flagged.
2. **Degraded State**: Health monitoring logs the drop and marks input status as `DEGRADED`.
3. **Controlled Recovery Loop**: If `CAMERA_RECONNECT_ENABLED = True`, Zogan releases the handle and attempts reconnection up to `CAMERA_MAX_RECONNECT_ATTEMPTS = 3` times, waiting `CAMERA_RECONNECT_DELAY_SECONDS = 1.0` between attempts.
4. **Recovery**: If the stream successfully yields a valid frame, camera health recovers to `ONLINE` and execution resumes seamlessly.
5. **Safe Halting**: If all retries fail, input is marked `OFFLINE`, a critical notice is logged, and the application halts cleanly rather than hanging in an infinite busy-loop.

*Note: For recorded video replay files (`.mp4`), end-of-file is detected naturally without initiating reconnection attempts.*

---

## 7. Detector Safe Failure

If the YOLO neural network encounters an inference exception (e.g., CUDA OOM, corrupted frame buffer, or memory allocation error):

1. **Consecutive Failure Tracking**: Zogan tracks consecutive inference failures and total failures.
2. **Subsystem Degradation**: A single exception flags the detector as `DEGRADED` in health monitoring.
3. **Critical Threshold**: If consecutive failures reach `DETECTOR_MAX_CONSECUTIVE_FAILURES = 3`, detector status transitions to `FAILED` and overall system health becomes `OFFLINE`.
4. **Safe Failure Semantics (No False "All Clear")**:
   - `consecutive_elephant_frames` is **NOT** reset to 0 (which would falsely report "Monitoring - No Elephant Detected").
   - The system does **NOT** invent a `LOW` risk assessment.
   - The HUD explicitly displays `DETECTOR DEGRADED` or `DETECTOR FAILED` in high-visibility colors.

---

## 8. System Health & Observability Integration

Phase 9 integrates network and queue observability directly into the Phase 8 `SystemHealthMonitor`:

- **Natural Connectivity Detection**: Zogan avoids wasteful periodic internet pings. Network connectivity is inferred naturally from actual delivery results (`NETWORK_AVAILABLE` on success, `NETWORK_UNAVAILABLE` on failure).
- **Subsystem Metrics**:
  - `network_status`: `AVAILABLE` / `UNAVAILABLE` / `UNKNOWN`
  - `alert_delivery_status`: `HEALTHY` / `DEGRADED` / `FAILED`
  - `pending_alert_count`: Integer count of queued events
- **HUD Indicator**: The live video overlay dynamically displays offline status and pending alerts:
  ```
  SYS: DEGRADED (ONLINE) | NET:OFF | Q:3 | FPS: 28.4
  ```
  This immediately informs operators: *Local detection and tracking are operating normally, but remote alerts are currently queued for retry.*

---

## 9. Configuration Reference

All settings reside centrally in `config/settings.py` (with override support via environment variables):

```python
# -----------------------------------------------------------------------------
# OFFLINE QUEUE & RETRY POLICY (PHASE 9)
# -----------------------------------------------------------------------------
ALERT_QUEUE_FILE: str = "logs/delivery_queue.jsonl"
OFFLINE_QUEUE_ENABLED: bool = True
ALERT_QUEUE_MAX_SIZE: int = 100
ALERT_RETRY_LIMIT: int = 3
ALERT_RETRY_BASE_DELAY_SECONDS: float = 2.0
ALERT_RETRY_BACKOFF_FACTOR: float = 2.0
ALERT_DELIVERY_TIMEOUT_SECONDS: float = 3.0

# -----------------------------------------------------------------------------
# CAMERA RECOVERY POLICY (PHASE 9)
# -----------------------------------------------------------------------------
CAMERA_RECONNECT_ENABLED: bool = True
CAMERA_MAX_RECONNECT_ATTEMPTS: int = 3
CAMERA_RECONNECT_DELAY_SECONDS: float = 1.0

# -----------------------------------------------------------------------------
# DETECTOR RECOVERY POLICY (PHASE 9)
# -----------------------------------------------------------------------------
DETECTOR_MAX_CONSECUTIVE_FAILURES: int = 3
```

---

## 10. Automated Testing Coverage

The Phase 9 test suite (`tests/test_phase9.py`) validates all 20 required resilience scenarios without requiring physical cameras, internet connectivity, or GPU hardware:

1. `test_1_event_saved_before_alert_delivery`: Verified local disk write strictly precedes remote call.
2. `test_2_successful_delivery`: Full delivery cycle with state `SENT`.
3. `test_3_network_failure`: Network timeouts preserve local data and enqueue for retry.
4. `test_4_telegram_api_failure`: API error handling and status tracking.
5. `test_5_missing_telegram_credentials`: Graceful non-crashing fallback with `NOT_CONFIGURED`.
6. `test_6_retry_mechanism`: Queue draining delivers pending items when connection returns.
7. `test_7_retry_limit_bounded`: Attempts strictly capped by `ALERT_RETRY_LIMIT`.
8. `test_8_exponential_backoff_behavior`: Mathematical verification of exponential delay growth.
9. `test_9_queue_file_persistence`: File-backed JSON Lines disk inspection.
10. `test_10_queue_recovery_after_restart`: Persistence across simulated process restarts.
11. `test_11_event_deduplication_stable_id`: Stable UUID preservation preventing duplicate alerts.
12. `test_12_queue_cleanup_and_bounds`: Queue capacity limiting and pruning.
13. `test_13_camera_temporary_failure_degraded`: Frame drop handling in health monitor.
14. `test_14_camera_recovery_healthy`: Stream reconnection restoring `ONLINE` health.
15. `test_15_detector_temporary_failure_safe`: Recoverable inference exception isolation.
16. `test_16_detector_repeated_failure_limit`: Repeated failure transitioning to `FAILED`.
17. `test_17_system_health_integration_offline_queue`: Degraded state tracking during network outages.
18. `test_18_no_event_loss_across_failures`: 100% event retention under network blackout.
19. `test_19_no_false_low_risk_after_detector_failure`: Safe failure semantics.
20. `test_20_detection_pipeline_continues_while_remote_delivery_fails`: Non-blocking async dispatch.

Total automated test suite: **144 tests passing**.

---

## 11. Known Limitations & Future Improvements

1. **Single-Node Queue**: The queue is currently local to the edge device. In multi-camera edge clusters, a shared decentralized queue (e.g. SQLite WAL or MQTT edge broker) could coordinate alerts across physical devices.
2. **Cellular/Satellite Latency**: In extreme wilderness deployments with satellite backhaul, base delay and backoff factors may be adjusted to accommodate high-latency, intermittent uplink bursts.
