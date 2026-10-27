from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PixelPosition:
    x: float
    y: float


@dataclass(slots=True)
class GroundPosition:
    x: float
    z: float


@dataclass(slots=True)
class Observation:
    track_id: str
    timestamp: float
    source: str
    bbox_xyxy: list[float]
    pixel_position: PixelPosition
    relative_depth: float | None = None
    depth_method: str | None = None
    ground_position: GroundPosition | None = None
    ground_coordinate_unit: str | None = None
    position_kind: str = "image_only"
    detection_confidence: float | None = None
    quality_flags: list[str] = field(default_factory=list)
    occluded: bool | None = None

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "timestamp": self.timestamp,
            "source": self.source,
            "bbox_xyxy": self.bbox_xyxy,
            "pixel_position": {
                "x": self.pixel_position.x,
                "y": self.pixel_position.y,
            },
            "relative_depth": self.relative_depth,
            "depth_method": self.depth_method,
            "ground_position": (
                None
                if self.ground_position is None
                else {
                    "x": self.ground_position.x,
                    "z": self.ground_position.z,
                }
            ),
            "ground_coordinate_unit": self.ground_coordinate_unit,
            "position_kind": self.position_kind,
            "detection_confidence": self.detection_confidence,
            "quality_flags": self.quality_flags,
            "occluded": self.occluded,
        }


@dataclass(slots=True)
class RadarTrackSnapshot:
    track_id: str
    timestamp: float
    pixel_position: PixelPosition | None
    ground_position: GroundPosition
    position_kind: str
    quality_flags: list[str]
    confidence: float | None
    relative_depth: float | None = None

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "timestamp": self.timestamp,
            "pixel_position": (
                None
                if self.pixel_position is None
                else {
                    "x": self.pixel_position.x,
                    "y": self.pixel_position.y,
                }
            ),
            "ground_position": {
                "x": self.ground_position.x,
                "z": self.ground_position.z,
            },
            "relative_depth": self.relative_depth,
            "position_kind": self.position_kind,
            "quality_flags": self.quality_flags,
            "confidence": self.confidence,
        }
