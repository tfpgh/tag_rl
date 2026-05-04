from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from online.game.runtime import TagGameRuntime


INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Tag RL Live Game</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #111318;
      --panel: #1a1f27;
      --panel-2: #202632;
      --border: #313846;
      --text: #eef2f7;
      --muted: #9aa6b2;
      --accent: #6ec1ff;
      --good: #40c98b;
      --warn: #f0b35f;
      --bad: #e36d6d;
    }
    * { box-sizing: border-box; }
    body {
      font-family: ui-sans-serif, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1.55fr) minmax(320px, 1fr);
      gap: 16px;
      padding: 16px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.18);
    }
    .video-panel {
      position: sticky;
      top: 16px;
    }
    img {
      width: 100%;
      border-radius: 10px;
      display: block;
      background: #0b0d11;
      border: 1px solid #272d38;
    }
    .controls-header, .section-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 12px;
    }
    .section-title {
      font-size: 14px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
    }
    .buttons {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 14px;
    }
    button {
      background: #293241;
      color: var(--text);
      border: 1px solid #3f4b5e;
      border-radius: 8px;
      padding: 10px 14px;
      cursor: pointer;
      font: inherit;
    }
    button:hover { border-color: #607089; }
    button.danger { background: #6b2020; border-color: #9e3636; }
    .summary-grid, .robot-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }
    .card {
      background: var(--panel-2);
      border: 1px solid #2f3745;
      border-radius: 10px;
      padding: 12px;
    }
    .card h3 {
      margin: 0 0 10px;
      font-size: 14px;
    }
    .metric {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 6px 0;
      border-top: 1px solid rgba(255, 255, 255, 0.05);
      font-size: 14px;
    }
    .metric:first-of-type { border-top: 0; padding-top: 0; }
    .metric-label { color: var(--muted); }
    .metric-value {
      text-align: right;
      font-variant-numeric: tabular-nums;
      font-feature-settings: 'tnum';
    }
    .hero {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .hero .card { padding: 14px; }
    .hero-label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 8px;
    }
    .hero-value {
      font-size: 20px;
      font-weight: 700;
      line-height: 1.2;
    }
    .hero-subtext {
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
    }
    .badge-row {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      background: #273041;
      color: var(--text);
      border: 1px solid #3a4558;
      text-transform: capitalize;
    }
    .badge.good { background: rgba(64, 201, 139, 0.12); color: #9ee8c2; border-color: rgba(64, 201, 139, 0.35); }
    .badge.warn { background: rgba(240, 179, 95, 0.12); color: #ffd498; border-color: rgba(240, 179, 95, 0.35); }
    .badge.bad { background: rgba(227, 109, 109, 0.12); color: #ffb1b1; border-color: rgba(227, 109, 109, 0.35); }
    .progress {
      height: 10px;
      border-radius: 999px;
      background: #0f1319;
      border: 1px solid #2f3745;
      overflow: hidden;
      margin-top: 8px;
    }
    .progress-bar {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #4e89ff, #64d2ff);
    }
    .status-line {
      font-size: 14px;
      color: var(--muted);
      margin-bottom: 12px;
      min-height: 20px;
    }
    .event-list {
      display: grid;
      gap: 8px;
      margin: 0;
      padding: 0;
      list-style: none;
    }
    .event-item {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 8px 10px;
      border-radius: 8px;
      background: #171c24;
      border: 1px solid #2b3340;
      font-size: 13px;
      font-variant-numeric: tabular-nums;
    }
    details {
      margin-top: 12px;
      border-top: 1px solid rgba(255, 255, 255, 0.08);
      padding-top: 12px;
    }
    summary {
      cursor: pointer;
      color: var(--muted);
      font-weight: 600;
    }
    pre {
      margin: 10px 0 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      line-height: 1.4;
      color: #d7dee7;
      background: #11151c;
      border: 1px solid #2c3442;
      border-radius: 10px;
      padding: 12px;
      max-height: 320px;
      overflow: auto;
    }
    .empty {
      color: var(--muted);
      font-style: italic;
    }
    @media (max-width: 1080px) {
      .layout { grid-template-columns: 1fr; }
      .video-panel { position: static; }
    }
    @media (max-width: 720px) {
      .hero, .summary-grid, .robot-grid { grid-template-columns: 1fr; }
      .layout { padding: 12px; gap: 12px; }
      .panel { padding: 12px; }
      .controls-header, .section-header { align-items: flex-start; flex-direction: column; }
    }
  </style>
</head>
<body>
  <div class=\"layout\">
      <div class=\"panel video-panel\">
        <img src=\"/video.mjpg\" alt=\"Live video\" />
      </div>
    <div class=\"panel\">
        <div class=\"controls-header\">
          <div id=\"connectionStatus\" class=\"status-line\">Connecting to telemetry...</div>
          <div class=\"badge-row\">
            <span id=\"modeBadge\" class=\"badge\">mode</span>
            <span id=\"phaseBadge\" class=\"badge\">phase</span>
          <span id=\"readyBadge\" class=\"badge\">ready</span>
        </div>
      </div>
      <div class=\"buttons\">
        <button onclick=\"sendAction('arm')\">Arm</button>
        <button onclick=\"sendAction('start')\">Start Game</button>
        <button onclick=\"sendAction('stop')\">Stop</button>
        <button onclick=\"sendAction('reset')\">Reset</button>
        <button class=\"danger\" onclick=\"sendAction('estop')\">E-Stop</button>
        <button onclick=\"sendAction('clear_estop')\">Clear E-Stop</button>
        <button onclick=\"sendAction('disarm')\">Disarm</button>
      </div>
      <div class=\"buttons\">
        <button onclick=\"sendAction('record_off')\">Record Off</button>
        <button onclick=\"sendAction('record_train')\">Record Train</button>
        <button onclick=\"sendAction('record_eval')\">Record Eval</button>
        <button onclick=\"sendAction('record_showcase')\">Record Showcase</button>
      </div>

      <div class=\"hero\">
        <div class=\"card\">
          <div class=\"hero-label\">Match Progress</div>
          <div id=\"elapsedValue\" class=\"hero-value\">--</div>
          <div id=\"stepsValue\" class=\"hero-subtext\">Waiting for telemetry</div>
          <div class=\"progress\"><div id=\"progressBar\" class=\"progress-bar\"></div></div>
        </div>
        <div class=\"card\">
          <div class=\"hero-label\">Tracker FPS</div>
          <div id=\"trackerFpsValue\" class=\"hero-value\">--</div>
          <div id=\"trackerMeta\" class=\"hero-subtext\">Capture -- ms, loop -- ms</div>
        </div>
        <div class=\"card\">
          <div class=\"hero-label\">Frame Age</div>
          <div id=\"frameAgeValue\" class=\"hero-value\">--</div>
          <div id=\"runtimeStatus\" class=\"hero-subtext\">Waiting for telemetry</div>
        </div>
      </div>

      <div class=\"summary-grid\">
        <div class=\"card\">
          <h3>Runtime</h3>
          <div class=\"metric\"><span class=\"metric-label\">Status</span><span id=\"runtimeState\" class=\"metric-value\">--</span></div>
          <div class=\"metric\"><span class=\"metric-label\">Ready Reason</span><span id=\"readyReason\" class=\"metric-value\">--</span></div>
          <div class=\"metric\"><span class=\"metric-label\">Control Rate</span><span id=\"controlRate\" class=\"metric-value\">--</span></div>
          <div class=\"metric\"><span class=\"metric-label\">Render Rate</span><span id=\"renderRate\" class=\"metric-value\">--</span></div>
          <div class=\"metric\"><span class=\"metric-label\">Inference</span><span id=\"inferenceTime\" class=\"metric-value\">--</span></div>
          <div class=\"metric\"><span class=\"metric-label\">Recording</span><span id=\"recordingValue\" class=\"metric-value\">--</span></div>
        </div>
        <div class=\"card\">
          <h3>Game</h3>
          <div class=\"metric\"><span class=\"metric-label\">Duration</span><span id=\"durationValue\" class=\"metric-value\">--</span></div>
          <div class=\"metric\"><span class=\"metric-label\">Tag Distance</span><span id=\"tagDistanceValue\" class=\"metric-value\">--</span></div>
          <div class=\"metric\"><span class=\"metric-label\">Visible Tags</span><span id=\"visibleTagsValue\" class=\"metric-value\">--</span></div>
          <div class=\"metric\"><span class=\"metric-label\">Calibration</span><span id=\"calibrationValue\" class=\"metric-value\">--</span></div>
          <div class=\"metric\"><span class=\"metric-label\">Action Scale</span><span id=\"actionScaleValue\" class=\"metric-value\">--</span></div>
        </div>
      </div>

      <div class=\"section-header\">
        <div class=\"section-title\">Robots</div>
      </div>
      <div class=\"robot-grid\">
        <div class=\"card\">
          <h3>Chaser</h3>
          <div class=\"metric\"><span class=\"metric-label\">Pose</span><span id=\"chaserPose\" class=\"metric-value\">--</span></div>
          <div class=\"metric\"><span class=\"metric-label\">Raw Command</span><span id=\"chaserRaw\" class=\"metric-value\">--</span></div>
          <div class=\"metric\"><span class=\"metric-label\">Sent Command</span><span id=\"chaserSent\" class=\"metric-value\">--</span></div>
          <div class=\"metric\"><span class=\"metric-label\">Policy Summary</span><span id=\"chaserPolicy\" class=\"metric-value\">--</span></div>
        </div>
        <div class=\"card\">
          <h3>Evader</h3>
          <div class=\"metric\"><span class=\"metric-label\">Pose</span><span id=\"evaderPose\" class=\"metric-value\">--</span></div>
          <div class=\"metric\"><span class=\"metric-label\">Raw Command</span><span id=\"evaderRaw\" class=\"metric-value\">--</span></div>
          <div class=\"metric\"><span class=\"metric-label\">Sent Command</span><span id=\"evaderSent\" class=\"metric-value\">--</span></div>
          <div class=\"metric\"><span class=\"metric-label\">Policy Summary</span><span id=\"evaderPolicy\" class=\"metric-value\">--</span></div>
        </div>
      </div>

      <div class=\"section-header\">
        <div class=\"section-title\">Recent Events</div>
      </div>
      <ul id=\"eventsList\" class=\"event-list\">
        <li class=\"event-item empty\">Waiting for telemetry...</li>
      </ul>

      <details>
        <summary>Raw JSON</summary>
        <pre id=\"snapshot\">Waiting for telemetry...</pre>
      </details>
    </div>
  </div>
  <script>
    const snapshot = document.getElementById('snapshot');
    const connectionStatus = document.getElementById('connectionStatus');
    const modeBadge = document.getElementById('modeBadge');
    const phaseBadge = document.getElementById('phaseBadge');
    const readyBadge = document.getElementById('readyBadge');
    const progressBar = document.getElementById('progressBar');
    const eventsList = document.getElementById('eventsList');

    const elements = {
      elapsedValue: document.getElementById('elapsedValue'),
      stepsValue: document.getElementById('stepsValue'),
      trackerFpsValue: document.getElementById('trackerFpsValue'),
      trackerMeta: document.getElementById('trackerMeta'),
      frameAgeValue: document.getElementById('frameAgeValue'),
      runtimeStatus: document.getElementById('runtimeStatus'),
      runtimeState: document.getElementById('runtimeState'),
      readyReason: document.getElementById('readyReason'),
      controlRate: document.getElementById('controlRate'),
      renderRate: document.getElementById('renderRate'),
      inferenceTime: document.getElementById('inferenceTime'),
      recordingValue: document.getElementById('recordingValue'),
      durationValue: document.getElementById('durationValue'),
      tagDistanceValue: document.getElementById('tagDistanceValue'),
      visibleTagsValue: document.getElementById('visibleTagsValue'),
      calibrationValue: document.getElementById('calibrationValue'),
      actionScaleValue: document.getElementById('actionScaleValue'),
      chaserPose: document.getElementById('chaserPose'),
      chaserRaw: document.getElementById('chaserRaw'),
      chaserSent: document.getElementById('chaserSent'),
      chaserPolicy: document.getElementById('chaserPolicy'),
      evaderPose: document.getElementById('evaderPose'),
      evaderRaw: document.getElementById('evaderRaw'),
      evaderSent: document.getElementById('evaderSent'),
      evaderPolicy: document.getElementById('evaderPolicy'),
    };

    function fmtNumber(value, digits = 2, suffix = '') {
      if (value === null || value === undefined || Number.isNaN(value)) {
        return '--';
      }
      return `${Number(value).toFixed(digits)}${suffix}`;
    }

    function fmtPercent(value) {
      if (value === null || value === undefined || Number.isNaN(value)) {
        return '--';
      }
      return `${(Number(value) * 100).toFixed(0)}%`;
    }

    function fmtPose(pose) {
      if (!pose) {
        return 'Not visible';
      }
      const deg = Number(pose.yaw_rad) * 180 / Math.PI;
      return `${fmtNumber(pose.x_m, 3, ' m')}, ${fmtNumber(pose.y_m, 3, ' m')}, ${fmtNumber(deg, 0, ' deg')}`;
    }

    function fmtCommand(command) {
      if (!command) {
        return '--';
      }
      return `L ${fmtNumber(command.left, 2)} / R ${fmtNumber(command.right, 2)}`;
    }

    function fmtPolicy(policy) {
      if (!policy || !policy.observation_ready) {
        return 'Observation not ready';
      }
      return `progress ${fmtPercent(policy.episode_progress)}, ray ${fmtNumber(policy.mean_ray_distance, 2)} m, hit ${fmtPercent(policy.agent_hit_fraction)}`;
    }

    function fmtTime(timestamp) {
      if (!timestamp) {
        return '--';
      }
      return new Date(timestamp * 1000).toLocaleTimeString();
    }

    function badgeClass(kind) {
      if (kind === 'good') return 'badge good';
      if (kind === 'warn') return 'badge warn';
      if (kind === 'bad') return 'badge bad';
      return 'badge';
    }

    function setBadge(element, label, kind) {
      element.textContent = label;
      element.className = badgeClass(kind);
    }

    function renderEvents(events) {
      if (!events || !events.length) {
        eventsList.innerHTML = '<li class="event-item empty">No events yet</li>';
        return;
      }
      eventsList.innerHTML = events.slice(0, 8).map((event) => {
        return `<li class="event-item"><span>${event.message}</span><span>${fmtTime(event.timestamp)}</span></li>`;
      }).join('');
    }

    function renderSnapshot(data) {
      const runtime = data.runtime || {};
      const game = data.game || {};
      const tracker = data.tracker || {};
      const robots = data.robots || {};
      const policy = data.policy || {};

      snapshot.textContent = JSON.stringify(data, null, 2);

      setBadge(modeBadge, `Mode: ${runtime.mode || '--'}`, runtime.mode === 'estop' ? 'bad' : runtime.mode === 'running' ? 'good' : runtime.mode === 'armed' ? 'warn' : '');
      setBadge(phaseBadge, `Phase: ${runtime.phase || '--'}`, runtime.phase === 'active' ? 'good' : runtime.phase === 'tagged' || runtime.phase === 'time_up' || runtime.phase === 'stopped' ? 'warn' : '');
      setBadge(readyBadge, runtime.ready ? 'Ready' : 'Not Ready', runtime.ready ? 'good' : 'bad');

      elements.elapsedValue.textContent = fmtNumber(game.elapsed_s, 1, ' s');
      elements.stepsValue.textContent = `${game.step_count ?? '--'} / ${game.max_steps ?? '--'} steps`; 
      elements.trackerFpsValue.textContent = fmtNumber(tracker.fps, 1);
      elements.trackerMeta.textContent = `Capture ${fmtNumber(tracker.capture_ms, 1, ' ms')}, loop ${fmtNumber(tracker.loop_ms, 1, ' ms')}`;
      elements.frameAgeValue.textContent = fmtNumber(runtime.frame_age_s, 3, ' s');
      elements.runtimeStatus.textContent = runtime.status || '--';

      elements.runtimeState.textContent = `${runtime.mode || '--'} / ${runtime.phase || '--'}`;
      elements.readyReason.textContent = runtime.ready_reason || '--';
      elements.controlRate.textContent = `${fmtNumber(runtime.control_actual_hz, 1)} Hz (target ${fmtNumber(runtime.control_target_hz, 1)} Hz)`;
      elements.renderRate.textContent = `${fmtNumber(runtime.render_actual_hz, 1)} Hz (target ${fmtNumber(runtime.render_target_hz, 1)} Hz)`;
      elements.inferenceTime.textContent = `${fmtNumber(runtime.observation_build_ms, 1, ' ms')} build, ${fmtNumber(runtime.inference_ms, 1, ' ms')} infer, ${fmtNumber(runtime.send_ms, 1, ' ms')} send`;
      elements.recordingValue.textContent = `${runtime.recording_split || 'off'}${runtime.recording_active ? ' (active)' : ''}`;

      elements.durationValue.textContent = fmtNumber(game.duration_s, 1, ' s');
      elements.tagDistanceValue.textContent = fmtNumber(game.tag_distance_m, 3, ' m');
      elements.visibleTagsValue.textContent = `${tracker.visible_tags ?? '--'}`;
      elements.calibrationValue.textContent = tracker.calibration_valid ? 'Valid' : 'Incomplete';
      elements.actionScaleValue.textContent = fmtNumber(runtime.action_output_scale, 2);

      elements.chaserPose.textContent = fmtPose(robots.chaser_pose);
      elements.chaserRaw.textContent = fmtCommand(robots.chaser_command_raw);
      elements.chaserSent.textContent = fmtCommand(robots.chaser_command_sent);
      elements.chaserPolicy.textContent = fmtPolicy(policy.chaser);

      elements.evaderPose.textContent = fmtPose(robots.evader_pose);
      elements.evaderRaw.textContent = fmtCommand(robots.evader_command_raw);
      elements.evaderSent.textContent = fmtCommand(robots.evader_command_sent);
      elements.evaderPolicy.textContent = fmtPolicy(policy.evader);

      progressBar.style.width = `${Math.max(0, Math.min(100, Number(game.episode_progress || 0) * 100))}%`;
      renderEvents(data.events || []);
    }

    async function sendAction(action) {
      connectionStatus.textContent = `Sending ${action}...`;
      try {
        const response = await fetch(`/api/control/${action}`, { method: 'POST' });
        if (!response.ok) {
          const payload = await response.json().catch(() => ({}));
          throw new Error(payload.detail || `HTTP ${response.status}`);
        }
        connectionStatus.textContent = `Sent ${action}`;
      } catch (error) {
        connectionStatus.textContent = `Control error: ${error.message}`;
      }
    }

    function connectWebSocket() {
      const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
      const ws = new WebSocket(`${protocol}://${location.host}/ws`);

      ws.onopen = () => {
        connectionStatus.textContent = 'Telemetry connected';
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        renderSnapshot(data);
      };

      ws.onerror = () => {
        connectionStatus.textContent = 'Telemetry connection error';
      };

      ws.onclose = () => {
        connectionStatus.textContent = 'Telemetry disconnected, retrying...';
        window.setTimeout(connectWebSocket, 1000);
      };
    }

    connectWebSocket();
  </script>
</body>
</html>
"""


def _mjpeg_stream(runtime: TagGameRuntime) -> Iterator[bytes]:
    boundary = b"--frame\r\n"
    while True:
        frame = runtime.get_jpeg_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        yield boundary
        yield b"Content-Type: image/jpeg\r\n\r\n"
        yield frame
        yield b"\r\n"


def build_dashboard_app(runtime: TagGameRuntime) -> FastAPI:
    app = FastAPI(title="Tag RL Live Game")

    @app.get("/")
    def index() -> HTMLResponse:
        return HTMLResponse(INDEX_HTML)

    @app.get("/snapshot")
    def snapshot() -> JSONResponse:
        return JSONResponse(runtime.get_snapshot())

    @app.post("/api/control/{action}")
    def control(action: str) -> JSONResponse:
        try:
            runtime.handle_action(action)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse({"ok": True, "action": action})

    @app.get("/video.mjpg")
    def video() -> StreamingResponse:
        return StreamingResponse(
            _mjpeg_stream(runtime),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                await websocket.send_json(runtime.get_snapshot())
                await asyncio.sleep(1.0 / runtime.config.telemetry_hz)
        except Exception:
            pass

    return app
