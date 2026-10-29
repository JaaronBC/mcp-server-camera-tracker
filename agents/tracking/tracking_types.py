from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PixelPosition:
    x: float
    y: float


@dataclass(slots=True)
class GroundPosition:
    x: float
    z: float


@dataclass(slots=True)
class RadarTrackSnapshot:
    track_id: str
    timestamp: float
    pixel_position: PixelPosition | None
    ground_position: GroundPosition
    ground_coordinate_unit: str | None
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
            "ground_coordinate_unit": self.ground_coordinate_unit,
            "relative_depth": self.relative_depth,
            "position_kind": self.position_kind,
            "quality_flags": self.quality_flags,
            "confidence": self.confidence,
        }