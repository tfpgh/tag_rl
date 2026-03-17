from __future__ import annotations

import asyncio
import mimetypes
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from online.config import DemoConfig
from online.runtime_state import RuntimeState


def create_app(config: DemoConfig, state: RuntimeState) -> FastAPI:
    app = FastAPI(title="tag-demo")
    dist_dir = config.gui.frontend_dir / "dist"
    assets_dir = dist_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/api/state")
    async def get_state() -> dict:
        return state.snapshot().to_dict()

    @app.post("/api/control/enable")
    async def enable_control() -> dict[str, bool]:
        state.mutate_snapshot(
            lambda snapshot: setattr(snapshot.operator, "control_enabled", True)
        )
        return {"ok": True}

    @app.post("/api/control/disable")
    async def disable_control() -> dict[str, bool]:
        def mutate(snapshot) -> None:  # type: ignore[no-untyped-def]
            snapshot.operator.control_enabled = False
            snapshot.operator.emergency_stop = False

        state.mutate_snapshot(mutate)
        return {"ok": True}

    @app.post("/api/control/estop")
    async def estop() -> dict[str, bool]:
        state.mutate_snapshot(
            lambda snapshot: setattr(snapshot.operator, "emergency_stop", True)
        )
        return {"ok": True}

    @app.post("/api/control/reset-estop")
    async def reset_estop() -> dict[str, bool]:
        state.mutate_snapshot(
            lambda snapshot: setattr(snapshot.operator, "emergency_stop", False)
        )
        return {"ok": True}

    @app.post("/api/detection/pause")
    async def pause_detection() -> dict[str, bool]:
        state.mutate_snapshot(
            lambda snapshot: setattr(snapshot.operator, "pause_detection", True)
        )
        return {"ok": True}

    @app.post("/api/detection/resume")
    async def resume_detection() -> dict[str, bool]:
        state.mutate_snapshot(
            lambda snapshot: setattr(snapshot.operator, "pause_detection", False)
        )
        return {"ok": True}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        interval = 1.0 / config.gui.websocket_rate_hz
        try:
            while True:
                await websocket.send_json(state.snapshot().to_dict())
                await asyncio.sleep(interval)
        except WebSocketDisconnect:
            return

    @app.get("/video.mjpeg")
    async def video_feed() -> StreamingResponse:
        interval = 1.0 / config.gui.mjpeg_rate_hz

        async def stream():
            while not state.stop_event().is_set():
                jpeg = state.get_annotated_jpeg()
                if jpeg is None:
                    await asyncio.sleep(interval)
                    continue
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                await asyncio.sleep(interval)

        return StreamingResponse(
            stream(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/{full_path:path}", response_model=None)
    async def serve_frontend(full_path: str) -> Response:
        if not dist_dir.exists():
            return HTMLResponse(
                "<html><body><h1>Frontend not built</h1><p>Run npm install && npm run build in online/gui.</p></body></html>",
                status_code=200,
            )
        if full_path and full_path != "index.html":
            target = dist_dir / full_path
            if target.exists() and target.is_file():
                media_type = (
                    mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                )
                return FileResponse(target, media_type=media_type)
        index_path = dist_dir / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="Frontend index missing")
        return FileResponse(index_path, media_type="text/html")

    return app
