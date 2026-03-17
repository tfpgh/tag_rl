import { useEffect, useMemo, useState } from 'react'
import ArenaMap from './components/ArenaMap'
import type { Snapshot } from './types'

const initialSnapshot: Snapshot = {
  frame: { frame_id: 0, timestamp: 0, width: 0, height: 0, age_s: 0 },
  detections: [],
  calibration: { status: 'uncalibrated', stable_count: 0, last_update_s: 0, source_tag_ids: [], arena_corners_world: [[-1.12, -0.51], [1.12, -0.51], [1.12, 0.51], [-1.12, 0.51]] },
  world: { timestamp: 0, ready: false, frame_id: 0, chaser: null, evader: null, obstacles: [] },
  policy: { timestamp: 0, enabled: false, episode_progress: 0, chaser_action: [0, 0], evader_action: [0, 0], observation_size: 0, reset_count: 0, last_reason: 'idle' },
  chaser_command: { name: 'chaser', timestamp: 0, left: 0, right: 0, packets_sent: 0, watchdog_stop: false },
  evader_command: { name: 'evader', timestamp: 0, left: 0, right: 0, packets_sent: 0, watchdog_stop: false },
  stats: { capture_fps: 0, detection_fps: 0, detection_ms: 0, control_hz: 0, control_ms: 0, frame_age_s: 0, world_age_s: 0, detections: 0, last_error: '' },
  operator: { control_enabled: false, emergency_stop: false, pause_detection: false, show_debug: true },
}

async function post(path: string) {
  await fetch(path, { method: 'POST' })
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

export default function App() {
  const [snapshot, setSnapshot] = useState<Snapshot>(initialSnapshot)
  const [connected, setConnected] = useState(false)

  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const socket = new WebSocket(`${protocol}://${window.location.host}/ws`)
    socket.onopen = () => setConnected(true)
    socket.onclose = () => setConnected(false)
    socket.onmessage = (event) => setSnapshot(JSON.parse(event.data) as Snapshot)
    return () => socket.close()
  }, [])

  const readiness = useMemo(() => {
    if (snapshot.operator.emergency_stop) return 'E-Stop'
    if (!snapshot.world.ready) return 'Waiting for world lock'
    if (!snapshot.operator.control_enabled) return 'Armed but disabled'
    return 'Running'
  }, [snapshot])

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Robots Playing Tag</p>
          <h1>Local Demo Console</h1>
          <p className="hero-copy">Live AprilTag perception, policy telemetry, dual-robot commands, and safety state in one dashboard.</p>
        </div>
        <div className="status-stack">
          <span className={`pill ${connected ? 'online' : 'offline'}`}>{connected ? 'ws connected' : 'ws offline'}</span>
          <span className={`pill ${snapshot.world.ready ? 'online' : 'warn'}`}>{snapshot.calibration.status}</span>
          <span className={`pill ${snapshot.operator.control_enabled ? 'online' : 'offline'}`}>{readiness}</span>
        </div>
      </header>

      <main className="dashboard-grid">
        <section className="panel panel-video">
          <div className="panel-head">
            <h2>Annotated Video</h2>
            <span>{snapshot.frame.width}x{snapshot.frame.height}</span>
          </div>
          <img className="video-feed" src="/video.mjpeg" alt="Live annotated feed" />
        </section>

        <section className="panel">
          <div className="panel-head">
            <h2>Arena Map</h2>
            <span>{snapshot.world.obstacles.length} obstacles</span>
          </div>
          <ArenaMap snapshot={snapshot} />
        </section>

        <section className="panel">
          <div className="panel-head">
            <h2>System Stats</h2>
          </div>
          <div className="metric-grid">
            <Metric label="Capture FPS" value={snapshot.stats.capture_fps.toFixed(1)} />
            <Metric label="Detect FPS" value={snapshot.stats.detection_fps.toFixed(1)} />
            <Metric label="Detect ms" value={snapshot.stats.detection_ms.toFixed(1)} />
            <Metric label="Control Hz" value={snapshot.stats.control_hz.toFixed(1)} />
            <Metric label="Control ms" value={snapshot.stats.control_ms.toFixed(1)} />
            <Metric label="Frame age" value={`${(snapshot.stats.frame_age_s * 1000).toFixed(0)} ms`} />
            <Metric label="World age" value={`${(snapshot.stats.world_age_s * 1000).toFixed(0)} ms`} />
            <Metric label="Detections" value={`${snapshot.stats.detections}`} />
          </div>
          {snapshot.stats.last_error ? <p className="error-banner">{snapshot.stats.last_error}</p> : null}
        </section>

        <section className="panel">
          <div className="panel-head">
            <h2>Policy & Commands</h2>
            <span>obs {snapshot.policy.observation_size}</span>
          </div>
          <div className="metric-grid">
            <Metric label="Episode" value={`${(snapshot.policy.episode_progress * 100).toFixed(0)}%`} />
            <Metric label="Resets" value={`${snapshot.policy.reset_count}`} />
            <Metric label="Reason" value={snapshot.policy.last_reason} />
            <Metric label="Chaser L/R" value={`${snapshot.chaser_command.left.toFixed(2)} / ${snapshot.chaser_command.right.toFixed(2)}`} />
            <Metric label="Evader L/R" value={`${snapshot.evader_command.left.toFixed(2)} / ${snapshot.evader_command.right.toFixed(2)}`} />
            <Metric label="Packets" value={`${snapshot.chaser_command.packets_sent} / ${snapshot.evader_command.packets_sent}`} />
          </div>
        </section>

        <section className="panel">
          <div className="panel-head">
            <h2>Tracking</h2>
          </div>
          <div className="tracking-grid">
            {[snapshot.world.chaser, snapshot.world.evader].map((body) => (
              <div key={body?.tag_id ?? Math.random()} className="tracking-card">
                <h3>{body?.label ?? 'unknown'}</h3>
                <p>tag {body?.tag_id ?? '-'}</p>
                <p>{body?.visible ? 'visible' : 'missing'} / {body?.stale ? 'stale' : 'fresh'}</p>
                <p>age {(body?.age_s ?? 0).toFixed(3)} s</p>
                <p>
                  pose {body?.filtered_pose ? `${body.filtered_pose.x.toFixed(2)}, ${body.filtered_pose.y.toFixed(2)}, ${body.filtered_pose.yaw.toFixed(2)}` : '--'}
                </p>
              </div>
            ))}
          </div>
          <div className="obstacle-list">
            {snapshot.world.obstacles.map((obstacle) => (
              <div key={obstacle.tag_id} className="obstacle-row">
                <span>Obstacle {obstacle.tag_id}</span>
                <span>{obstacle.visible ? 'visible' : 'held'}</span>
                <span>{obstacle.pose.x.toFixed(2)}, {obstacle.pose.y.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </section>

        <section className="panel">
          <div className="panel-head">
            <h2>Controls</h2>
          </div>
          <div className="button-grid">
            <button onClick={() => post('/api/control/enable')}>Enable Control</button>
            <button onClick={() => post('/api/control/disable')}>Disable Control</button>
            <button className="danger" onClick={() => post('/api/control/estop')}>E-Stop</button>
            <button onClick={() => post('/api/control/reset-estop')}>Reset E-Stop</button>
            <button onClick={() => post('/api/detection/pause')}>Pause Detection</button>
            <button onClick={() => post('/api/detection/resume')}>Resume Detection</button>
          </div>
        </section>
      </main>
    </div>
  )
}
