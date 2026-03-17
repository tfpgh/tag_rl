export type Pose2D = {
  x: number
  y: number
  yaw: number
  timestamp: number
}

export type TrackedBody = {
  tag_id: number
  label: string
  visible: boolean
  stale: boolean
  age_s: number
  raw_pose: Pose2D | null
  filtered_pose: Pose2D | null
}

export type TrackedObstacle = {
  tag_id: number
  visible: boolean
  stale: boolean
  age_s: number
  pose: Pose2D
  size_m: number
}

export type Snapshot = {
  frame: {
    frame_id: number
    timestamp: number
    width: number
    height: number
    age_s: number
  }
  detections: Array<{
    tag_id: number
    center_px: [number, number]
    corners_px: Array<[number, number]>
    decision_margin: number
    hamming: number
  }>
  calibration: {
    status: string
    stable_count: number
    last_update_s: number
    source_tag_ids: number[]
    arena_corners_world: Array<[number, number]>
  }
  world: {
    timestamp: number
    ready: boolean
    frame_id: number
    chaser: TrackedBody | null
    evader: TrackedBody | null
    obstacles: TrackedObstacle[]
  }
  policy: {
    timestamp: number
    enabled: boolean
    episode_progress: number
    chaser_action: number[]
    evader_action: number[]
    observation_size: number
    reset_count: number
    last_reason: string
  }
  chaser_command: {
    name: string
    timestamp: number
    left: number
    right: number
    packets_sent: number
    watchdog_stop: boolean
    last_error: string
  }
  evader_command: {
    name: string
    timestamp: number
    left: number
    right: number
    packets_sent: number
    watchdog_stop: boolean
    last_error: string
  }
  stats: {
    capture_fps: number
    detection_fps: number
    detection_ms: number
    control_hz: number
    control_ms: number
    frame_age_s: number
    world_age_s: number
    detections: number
    last_error: string
  }
  operator: {
    control_enabled: boolean
    emergency_stop: boolean
    pause_detection: boolean
    show_debug: boolean
  }
}
