from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ArenaConfig:
    corner_tag_ids: tuple[int, int, int, int] = (0, 1, 2, 3)
    chaser_tag_id: int = 4
    evader_tag_id: int = 5
    obstacle_tag_ids: tuple[int, ...] = (6,)
    tag_size_m: float = 0.1
    tag_center_width_m: float = 2.3384
    tag_center_height_m: float = 1.1192
    obstacle_size_m: float = 0.1

    @property
    def board_width_m(self) -> float:
        return self.tag_center_width_m - self.tag_size_m

    @property
    def board_height_m(self) -> float:
        return self.tag_center_height_m - self.tag_size_m


@dataclass(slots=True)
class CameraConfig:
    device_index: int = 0
    backend: int | None = None
    frame_width: int = 1920
    frame_height: int = 1080
    fps: int = 60
    mjpg: bool = True
    buffer_size: int = 1
    auto_exposure: float | None = 1.0
    exposure: float | None = 17.0


@dataclass(slots=True)
class DetectorConfig:
    family: str = "tagStandard41h12"
    threads: int = 24
    quad_decimate: float = 2.0


@dataclass(slots=True)
class ViewConfig:
    frame_width: int = 1920
    stats_height: int = 35
    buffer_fraction: float = 0.01


@dataclass(slots=True)
class DisplayConfig:
    show_raw_window: bool = False
    show_arena_window: bool = False
    preview_fps: float = 15.0


@dataclass(slots=True)
class TeleopConfig:
    robot_ip: str = "192.168.1.5"
    udp_port: int = 8888
    send_hz: float = 20.0
    linear_step: float = 0.15
    angular_step: float = 0.015
    linear_max: float = 1.0
    angular_max: float = 1.0
    robot_tag_id: int = 4
    run_label: str = "manual_collect"


@dataclass(slots=True)
class TrackerConfig:
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    view: ViewConfig = field(default_factory=ViewConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    calibration_frames: int = 30
    smoothing_alpha: float = 0.85
    use_roi_tracking: bool = True
    roi_padding_scale: float = 3.0
    min_roi_size_px: int = 160
    max_roi_size_px: int = 600
    full_frame_refresh_interval: int = 15
    teleop: TeleopConfig = field(default_factory=TeleopConfig)
