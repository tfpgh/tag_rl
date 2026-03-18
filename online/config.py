from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ArenaConfig:
    corner_tag_ids: tuple[int, int, int, int] = (0, 1, 2, 3)
    chaser_tag_id: int = 4
    evader_tag_id: int = 5
    obstacle_tag_ids: tuple[int, ...] = (6,)
    tag_size_mm: float = 100.0
    tag_center_width_mm: float = 2338.4
    tag_center_height_mm: float = 1119.2
    obstacle_size_mm: float = 100.0

    @property
    def board_width_mm(self) -> float:
        return self.tag_center_width_mm - self.tag_size_mm

    @property
    def board_height_mm(self) -> float:
        return self.tag_center_height_mm - self.tag_size_mm


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
    threads: int = 8
    quad_decimate: float = 2.0


@dataclass(slots=True)
class ViewConfig:
    frame_width: int = 1920
    stats_height: int = 35
    buffer_fraction: float = 0.01


@dataclass(slots=True)
class DisplayConfig:
    show_raw_window: bool = True
    show_arena_window: bool = True
    preview_fps: float = 15.0


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
