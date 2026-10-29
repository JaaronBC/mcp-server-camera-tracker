# Handles dashboardCalibration behavior.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents.tracking.calibration import load_ground_plane_calibration

from ...registry import mcp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_PATH = PROJECT_ROOT / "captures" / "ground_plane_calibration.json"


def _ensure_json_object(payload: dict[str, Any] | Any) -> dict[str, Any]:
    # Performs ensure json object logic.
    if not isinstance(payload, dict):
        raise ValueError("calibration payload must be a JSON object")
    return payload


def _save_calibration_payload(payload: dict[str, Any]) -> None:
    # Performs save calibration payload logic.
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@mcp.tool()
def set_ground_plane_calibration(calibration: dict[str, Any]) -> dict[str, Any]:
    # Performs set ground plane calibration logic.
    """Save and validate dashboard ground-plane calibration JSON payload."""
    payload = _ensure_json_object(calibration)

    _save_calibration_payload(payload)
    loaded = load_ground_plane_calibration(CALIBRATION_PATH)
    if loaded is None:
        raise ValueError("calibration could not be loaded after save")

    return {
        "saved": True,
        "calibration_path": str(CALIBRATION_PATH),
        "ground_coordinate_unit": loaded.unit,
        "matrix_shape": list(loaded.matrix.shape),
    }


@mcp.tool()
def get_ground_plane_calibration() -> dict[str, Any]:
    # Performs get ground plane calibration logic.
    """Return current dashboard ground-plane calibration file contents."""
    if not CALIBRATION_PATH.is_file():
        return {
            "available": False,
            "calibration_path": str(CALIBRATION_PATH),
            "error": "calibration_missing",
        }

    payload = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {
            "available": False,
            "calibration_path": str(CALIBRATION_PATH),
            "error": "calibration_not_object",
        }

    return {
        "available": True,
        "calibration_path": str(CALIBRATION_PATH),
        "calibration": payload,
    }


@mcp.tool()
def validate_ground_plane_calibration() -> dict[str, Any]:
    # Performs validate ground plane calibration logic.
    """Validate current dashboard calibration file and return summary metadata."""
    if not CALIBRATION_PATH.is_file():
        return {
            "valid": False,
            "calibration_path": str(CALIBRATION_PATH),
            "error": "calibration_missing",
        }

    try:
        loaded = load_ground_plane_calibration(CALIBRATION_PATH)
    except Exception as error:
        return {
            "valid": False,
            "calibration_path": str(CALIBRATION_PATH),
            "error": f"{type(error).__name__}: {error}",
        }

    if loaded is None:
        return {
            "valid": False,
            "calibration_path": str(CALIBRATION_PATH),
            "error": "calibration_missing",
        }

    return {
        "valid": True,
        "calibration_path": str(CALIBRATION_PATH),
        "ground_coordinate_unit": loaded.unit,
        "matrix_shape": list(loaded.matrix.shape),
    }


@mcp.tool()
def clear_ground_plane_calibration() -> dict[str, Any]:
    # Performs clear ground plane calibration logic.
    """Delete dashboard ground-plane calibration file if present."""
    existed = CALIBRATION_PATH.is_file()
    if existed:
        CALIBRATION_PATH.unlink()

    return {
        "cleared": existed,
        "calibration_path": str(CALIBRATION_PATH),
    }
