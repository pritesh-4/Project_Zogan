# System Health Monitoring & Observability (Phase 8)

> **Architectural Specification & Engineering Reference**  
> *Project Zogan — AI-Assisted Wildlife Monitoring Prototype*

---

## 1. Executive Summary & Core Objective

In production wildlife monitoring deployments along forest-village fringes, an automated vision system must not only detect animals, but also **continuously report the operational integrity of its own sensing and inference pipeline**.

Silent failures—such as an unplugged camera, frozen video frame, inference crash, or collapsing frame rates—can create a false sense of security where an operator assumes an area is safe simply because no alerts are being received.

Phase 8 introduces a lightweight, zero-overhead **Observability & System Health Layer** (`monitoring/`) designed to:
1. Continuously monitor camera input stream freshness and frame reception.
2. Calculate stable rolling-window FPS throughput without single-frame jitter.
3. Track the lifecycle and error state of every pipeline stage (Input, Detector, Tracker, Risk Engine).
4. Synthesize multi-signal telemetry into an explainable, deterministic overall system state (`HEALTHY`, `DEGRADED`, `OFFLINE`, `UNKNOWN`).
5. Provide compact runtime HUD status and state-transition logging without cluttering the screen or spamming logs.
6. Strictly separate **System Health** (*"Is Zogan running correctly?"*) from **Threat Risk** (*"How dangerous is the elephant situation?"*).

---

## 2. System Health vs. Threat Risk: Orthogonal Separation

A fundamental design rule of Project Zogan is that **System Health** and **Elephant Threat Risk** are completely independent dimensions:

```
┌────────────────────────────────────────────────────────────────────────┐
│                          ORTHOGONAL DOMAINS                            │
├──────────────────────────────────┬─────────────────────────────────────┤
│        SYSTEM HEALTH             │             THREAT RISK             │
│   "Is the pipeline working?"     │    "How serious is the animal?"     │
├──────────────────────────────────┼─────────────────────────────────────┤
│ • Evaluates hardware & software  │ • Evaluates wildlife detections     │
│ • Camera, FPS, model inference   │ • Proximity, persistence, herd size │
│ • States: HEALTHY, DEGRADED,     │ • Levels: LOW, MEDIUM, HIGH,        │
│   OFFLINE, UNKNOWN               │   CRITICAL                          │
└──────────────────────────────────┴─────────────────────────────────────┘
```

### Operational Matrices

| Pipeline State | Detection Evidence | System Health | Threat Risk | Interpretation |
|:---|:---|:---:|:---:|:---|
| Normal capture, 30 FPS | Zero elephants in frame | **`HEALTHY`** | **`LOW`** | **Nominal operation.** Camera is actively scanning; no elephants present. |
| Normal capture, 30 FPS | Elephant inside protected zone | **`HEALTHY`** | **`CRITICAL`** | **Active incident.** System is working properly and sounding alarm. |
| Camera disconnected / frozen | Unable to process imagery | **`OFFLINE`** | **`N/A`** | **System outage.** Operator alerted of pipeline breakdown; safety status unknown. |
| Slow FPS (4 FPS) / inference drops | Elephant detected in background | **`DEGRADED`** | **`MEDIUM`** | **Degraded alert.** Detection verified but pipeline is experiencing compute bottleneck. |

> **Critical Safety Safeguard**: The absence of elephants is **never** interpreted as a system failure. Conversely, a technical malfunction is **never** interpreted as elephant presence or danger.

---

## 3. Architecture & Data Flow

Health monitoring operates non-intrusively alongside the primary video capture and inference loop:

```
Camera / Video Stream
         │
         ├──► [1] record_frame_read(success) ────► RollingFPS & Freshness Tracker
         ▼
YOLO Elephant Detector
         │
         ├──► [2] record_detector_status() ──────► Detector Health Probe
         ▼
ByteTrack Multi-Object Tracker
         │
         ├──► [3] record_tracker_status() ───────► Tracker Health Probe
         ▼
Spatial & Geofence Risk Engine 2.0
         │
         ├──► [4] record_risk_engine_status() ───► Risk Engine Health Probe
         ▼
[5] Central SystemHealthMonitor (monitoring/health.py)
         │  Synthesizes signals, checks transitions, generates reasons
         │
         ├──► Runtime HUD (ai/renderer.py) ──► Displays "SYS: HEALTHY | FPS: 28.4"
         ├──► Transition Logger ─────────────► Logs state changes only (e.g. HEALTHY -> DEGRADED)
         └──► HealthSnapshot (.to_dict()) ───► Available for JSON export, logs & diagnostics
```

---

## 4. State Definitions & Transitions

### 4.1 Overall System States

| Status | Meaning | Conditions | HUD Indicator Color |
|:---:|:---|:---|:---:|
| **`HEALTHY`** | All subsystems operational at expected performance levels. | Camera streaming, frame age $< 1.5\text{s}$, $\text{FPS} \ge 12.0$, all subsystems `READY`, 0 active consecutive errors. | **Green** `(0, 255, 0)` |
| **`DEGRADED`** | System is monitoring but experiencing performance degradation or partial non-fatal faults. | Throughput $< 12.0\text{ FPS}$, frame age $1.5\text{--}5.0\text{s}$, $1\text{--}4$ dropped frames, or non-fatal subsystem error. | **Yellow** `(0, 215, 255)` |
| **`OFFLINE`** | Monitoring has ceased or suffered a fatal failure. | Camera failed to open, frame age $> 5.0\text{s}$, $\ge 20$ consecutive read drops, fatal detector crash, or $\ge 10$ consecutive errors. | **Red** `(0, 0, 255)` |
| **`UNKNOWN`** | Pipeline is starting up or awaiting initial frames. | Stream opened, zero frames read yet (`total_frames == 0`). | **White / Gray** `(200, 200, 200)` |

### 4.2 Subsystem Statuses

- **Input / Camera Stream**: `ONLINE`, `DEGRADED`, `LOST`, `OFFLINE`, `UNKNOWN`.
- **Inference & Algorithmic Modules** (`Detector`, `Tracker`, `Risk Engine`): `READY`, `DEGRADED`, `FAILED`, `UNKNOWN`.

---

## 5. Metrics & Measurement Specifications

### 5.1 Rolling-Window FPS Calculation

Rather than computing frames-per-second over a single delta time ($\Delta t = t_i - t_{i-1}$), which introduces extreme volatility from operating system scheduling, `RollingFPS` maintains a sliding FIFO deque of the last $N$ frame timestamps (default: 30 frames):

$$\text{FPS}_{\text{current}} = \frac{K - 1}{t_K - t_1}$$

where $K = \text{len}(\text{deque})$ and $t_K - t_1$ is the total elapsed time across the window.
- Eliminates single-frame jitter.
- Tracks `min_observed_fps` and `max_observed_fps` over the session.
- Provides immediate detection of sustained thermal throttling or compute starvation.

### 5.2 Frame Freshness Latency

Frame freshness measures the age of the most recently acquired video frame:

$$\Delta t_{\text{freshness}} = t_{\text{now}} - t_{\text{last\_valid\_frame}}$$

- $\Delta t_{\text{freshness}} < 1.5\text{s}$: **`ONLINE`**
- $1.5\text{s} \le \Delta t_{\text{freshness}} < 5.0\text{s}$: **`DEGRADED`**
- $\Delta t_{\text{freshness}} \ge 5.0\text{s}$: **`OFFLINE`**

### 5.3 Bounded Memory Error Tracking

To prevent memory leaks during long-running field deployments:
- Total errors and consecutive errors are tracked as scalar counters.
- Historical error logs are held in a bounded ring buffer (`deque(maxlen=20)`).
- Error messages are sanitized to redact sensitive environment tokens, passwords, or keys.
- Successful frame processing decays consecutive error counts.

---

## 6. Health Snapshot Data Model

The health state is encapsulated in an immutable, serializable `HealthSnapshot` dataclass:

```python
@dataclass
class HealthSnapshot:
    overall_status: str  # HEALTHY, DEGRADED, OFFLINE, UNKNOWN
    input_status: str  # ONLINE, DEGRADED, LOST, OFFLINE, UNKNOWN
    detector_status: str  # READY, DEGRADED, FAILED, UNKNOWN
    tracker_status: str  # READY, DEGRADED, FAILED, UNKNOWN
    risk_engine_status: str  # READY, DEGRADED, FAILED, UNKNOWN
    current_fps: float  # Rolling window FPS
    average_fps: float  # Cumulative processed frames / total uptime
    last_frame_age_seconds: float  # Latency since last frame
    uptime_seconds: float  # Seconds since pipeline initialization
    total_frames: int  # Total read attempts
    processed_frames: int  # Frames successfully analyzed
    dropped_frames: int  # Failed read attempts
    error_count: int  # Total recorded errors
    consecutive_errors: int  # Active unbroken error sequence
    last_error: Optional[str]  # Sanitized message of last error
    last_error_time: Optional[str]  # UTC timestamp of last error
    reasons: list[str]  # Factual explanations for degraded/offline
    timestamp: str  # ISO-8601 UTC timestamp
```

### JSON Serialization Example

```json
{
  "overall_status": "HEALTHY",
  "input_status": "ONLINE",
  "detector_status": "READY",
  "tracker_status": "READY",
  "risk_engine_status": "READY",
  "current_fps": 29.84,
  "average_fps": 28.91,
  "last_frame_age_seconds": 0.033,
  "uptime_seconds": 124.5,
  "total_frames": 3600,
  "processed_frames": 3600,
  "dropped_frames": 0,
  "error_count": 0,
  "consecutive_errors": 0,
  "last_error": null,
  "last_error_time": null,
  "reasons": [
    "All monitored subsystems operational"
  ],
  "timestamp": "2026-10-04T10:30:00Z"
}
```

---

## 7. Runtime Heads-Up Display (HUD) Integration

The compact system health indicator is integrated into Row 2 of the video overlay (`ai/renderer.py`):

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STATUS: MONITORING (No Elephant Detected)                            MODEL: elephant_v1│
│ Target: ELEPHANT | Min Conf: 70% | Press 'Q' to exit           SYS: HEALTHY | FPS: 29.8│
└────────────────────────────────────────────────────────────────────────────────────────┘
```

When degraded or offline:
- **`SYS: DEGRADED (ONLINE) | FPS: 7.2`** in yellow.
- **`SYS: OFFLINE (LOST) | FPS: 0.0`** in red.

---

## 8. Configuration Reference

System health monitoring is configured through `config/settings.py`:

| Parameter | Default Value | Description |
|:---|:---:|:---|
| `HEALTH_MONITOR_ENABLED` | `True` | Master switch for runtime observability subsystem. |
| `HEALTH_FPS_MINIMUM_HEALTHY` | `12.0` | FPS threshold below which system transitions to `DEGRADED`. |
| `HEALTH_FPS_EXPECTED` | `30.0` | Expected baseline capture frame rate. |
| `HEALTH_FPS_WINDOW_SIZE` | `30` | Number of recent frames included in rolling average. |
| `HEALTH_FRESHNESS_DEGRADED_SECONDS` | `1.5` | Latency threshold for `DEGRADED` input stream. |
| `HEALTH_FRESHNESS_OFFLINE_SECONDS` | `5.0` | Latency threshold for `OFFLINE` stream disconnect. |
| `HEALTH_MAX_CONSECUTIVE_FRAME_FAILURES` | `5` | Dropped read attempts before marking stream `DEGRADED`. |
| `HEALTH_MAX_CONSECUTIVE_FRAME_FAILURES_OFFLINE`| `20` | Dropped read attempts before marking stream `OFFLINE`. |
| `HEALTH_MAX_CONSECUTIVE_ERRORS_DEGRADED` | `3` | Pipeline errors before transitioning to `DEGRADED`. |
| `HEALTH_MAX_CONSECUTIVE_ERRORS_OFFLINE` | `10` | Pipeline errors before transitioning to `OFFLINE`. |
| `HEALTH_MAX_ERROR_HISTORY` | `20` | Maximum error objects retained in memory buffer. |
| `HEALTH_HEARTBEAT_TIMEOUT_SECONDS` | `5.0` | Maximum duration before watchdog flags stalled loop. |
| `HEALTH_LOG_TRANSITIONS_ONLY` | `True` | Suppresses per-frame spam; logs state changes only. |

---

## 9. Verification & Failure Injection Testing

Reliability is tested deterministically using test doubles (`FakeClock`, `FakeCamera`, `FaultyDetector`, `FaultyTracker`, `FaultyRiskEngine`) in `monitoring/test_doubles.py`:

- **Zero Physical Hardware**: Tests do not require physical cameras, USB webcams, or monitors.
- **Deterministic Time**: `FakeClock` advances simulated timestamps instantly without `time.sleep()`, eliminating wall-clock timing flakiness.
- **Zero Network / Telegram Dependencies**: Health evaluation is strictly self-contained.
- **23 Comprehensive Scenarios**: All scenarios pass in `< 1.0` second (`tests/test_phase8.py`).
