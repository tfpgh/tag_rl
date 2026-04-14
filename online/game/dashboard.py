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
  <title>Tag RL Live Game</title>
  <style>
    body { font-family: ui-sans-serif, sans-serif; background: #111318; color: #eef2f7; margin: 0; }
    .layout { display: grid; grid-template-columns: 1.5fr 1fr; gap: 16px; padding: 16px; }
    .panel { background: #1a1f27; border: 1px solid #313846; border-radius: 10px; padding: 12px; }
    img { width: 100%; border-radius: 8px; display: block; }
    .buttons { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
    button { background: #293241; color: #eef2f7; border: 1px solid #3f4b5e; border-radius: 8px; padding: 10px 14px; cursor: pointer; }
    button.danger { background: #6b2020; border-color: #9e3636; }
    pre { margin: 0; white-space: pre-wrap; word-break: break-word; font-size: 12px; line-height: 1.35; }
  </style>
</head>
<body>
  <div class=\"layout\">
    <div class=\"panel\">
      <img src=\"/video.mjpg\" alt=\"Live video\" />
    </div>
    <div class=\"panel\">
      <div class=\"buttons\">
        <button onclick=\"sendAction('arm')\">Arm</button>
        <button onclick=\"sendAction('start')\">Start Game</button>
        <button onclick=\"sendAction('stop')\">Stop</button>
        <button onclick=\"sendAction('reset')\">Reset</button>
        <button class=\"danger\" onclick=\"sendAction('estop')\">E-Stop</button>
        <button onclick=\"sendAction('clear_estop')\">Clear E-Stop</button>
        <button onclick=\"sendAction('disarm')\">Disarm</button>
      </div>
      <pre id=\"snapshot\">Waiting for telemetry...</pre>
    </div>
  </div>
  <script>
    async function sendAction(action) {
      await fetch(`/api/control/${action}`, { method: 'POST' });
    }
    const snapshot = document.getElementById('snapshot');
    const ws = new WebSocket(`ws://${location.host}/ws`);
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      snapshot.textContent = JSON.stringify(data, null, 2);
    };
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
