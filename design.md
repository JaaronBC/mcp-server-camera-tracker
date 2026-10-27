---
name: counter-uas-visual-radar-dashboard
description: Build and extend a live webcam plus ground-plane radar dashboard for a Counter-UAS decision-support capstone, integrating OpenCV, YOLO/ByteTrack, optional depth estimation and vision-language interpretation, and three distinct agents.
---

# Counter-UAS Visual Radar Dashboard Skill

## Mission

Help build a reliable demo that displays a live webcam feed beside a radar-style visualization. The radar should plot tracked objects using a documented ground-plane coordinate system when calibration supports it. The system is for monitoring and human decision support; it must not autonomously initiate physical countermeasures or infer hostile intent from movement alone.

The current milestone prioritizes the visual dashboard and reliable data flow. Detection, depth, advanced analysis, and additional sensor integrations can be improved later. Keep the interface usable with simulated data while the live perception pipeline is under development.

## Required demo outcome

- A live webcam panel and radar panel are visible at the same time.
- YOLO detections and ByteTrack IDs, when available, are reflected in the UI.
- Radar markers update as new observations arrive and show short trajectory trails.
- The camera feed and radar update independently; slow LLM calls must never block frame capture or radar rendering.
- Track details can show ID, timestamp, image position, estimated ground position, motion history, and uncertainty.
- Camera, detector, depth, or tracking failures are shown as status/quality states instead of crashing the dashboard.
- Simulated tracks can be used when live detections are unavailable.
- The dashboard labels estimates honestly and distinguishes image-space coordinates, relative depth, calibrated floor coordinates, and true 3D position.

## Architecture principles

1. **Separate perception, geometry, agent reasoning, and presentation.** Do not put the camera loop, model inference, coordinate transforms, and UI rendering in one monolithic function.
2. **Keep real-time work deterministic.** OpenCV capture, YOLO inference, ByteTrack, calibration transforms, track history, and radar rendering should run without waiting for LLM responses.
3. **Use a shared, typed observation contract.** All agents and UI components should consume structured records rather than parse free-form prose.
4. **Use bounded asynchronous work.** Use queues or latest-frame buffers, timestamps, timeouts, and bounded history. Drop stale frames rather than building an unbounded backlog.
5. **Keep LLMs out of the critical frame loop.** Invoke a vision-language model only for selected frames, ambiguous cases, or periodic summaries. Do not send every frame unless there is a clear performance and cost justification.
6. **Treat model output as an interpretation, not ground truth.** Validate schema, preserve source measurements, record uncertainty, and never allow an LLM to overwrite measured coordinates.
7. **Make components replaceable.** Simulated data, YOLO detections, depth estimation, and future sensor sources should feed the same observation interface.
8. **Human oversight.** Present evidence and uncertainty for operator review. No autonomous countermeasure actions.

## Three-agent design

The coordinator is ordinary application code and routes typed messages. It does not impersonate an agent or replace their independent responsibilities.

### 1. Tracking Agent

**Purpose:** Produce reliable, timestamped observations from the camera/perception pipeline.

Responsibilities:
- Consume observations from OpenCV and YOLO/ByteTrack.
- Preserve stable track IDs when the tracker supports them.
- Attach image-space bounding boxes and image coordinates.
- Optionally attach depth-model estimates and depth-quality indicators.
- Convert image points to ground-plane coordinates only when a valid calibration and valid ground-plane point are available.
- Flag occlusion, missing detections, stale observations, poor image quality, invalid depth, and track discontinuities.
- Respond to bounded requests for additional observation windows.
- Optionally invoke its own vision-language model call for selected ambiguous frames. Supply the original frame and relevant structured evidence, and request a constrained response describing visible context and uncertainty.

The Tracking Agent must not:
- Treat a monocular depth value as meters unless the method has been metrically calibrated and validated.
- Treat raw pixel distance as physical distance.
- project an airborne object's image point onto the floor and label it its true 3D position.
- Allow the vision-language model to invent or replace track IDs, coordinates, timestamps, or confidence measurements.

If the project requirement calls for three LLM-backed agents, Tracking Agent must have its own LLM invocation, system prompt, input/output contract, and tests. The actual frame-by-frame tracking remains algorithmic.

### 2. Observation / Behavior Agent

**Purpose:** Analyze timestamped track history and describe movement patterns.

Responsibilities:
- Consume histories and quality flags, not a continuous raw video stream by default.
- Compute or consume deterministic quantities such as velocity, direction, and trajectory summaries.
- Identify candidate patterns such as loitering, turns, corridor deviation, or missing-history limitations using explicit rules or validated ML methods.
- Produce qualified summaries and uncertainty; distinguish observed facts from interpretations.
- Use a deterministic motion model for numerical future coordinates. An LLM may explain a prediction but must not fabricate coordinates.
- Withdraw or revise a prediction when new evidence contradicts assumptions or the history is insufficient.

The Observation Agent should state when there is insufficient history, invalid timestamps, uncertain coordinate mapping, or an unsuitable coordinate frame.

### 3. Reasoning / Decision-Support Agent

**Purpose:** Coordinate evidence collection and create explainable operator-facing assessments.

Responsibilities:
- Combine Tracking and Observation outputs without conflating their provenance.
- Identify specific evidence gaps and request targeted follow-up observations through the coordinator.
- Use bounded retries, request limits, and timeouts to prevent endless feedback loops.
- Generate assessments with supporting evidence, limitations, and a clear separation between facts and interpretation.
- Use transparent, configured rules or a separately validated model for any numerical score; do not let an LLM invent a risk score.
- Communicate assessment updates to the dashboard and preserve assessment history for replay.

The system should use neutral descriptions such as “requires review” or “unusual relative to the configured scenario.” Movement alone is not evidence of hostile intent.

## Ground-plane radar and coordinate conventions

Choose one convention and use it everywhere. Recommended convention for a fixed camera looking across a flat floor:

- `X`: horizontal floor axis, positive to the right from the chosen origin.
- `Z`: floor axis, positive away from the camera.
- `Y` or `altitude`: reserved for vertical position only when a valid 3D method provides it.
- Radar screen mapping must be explicit, for example `screen_x = center_x + scale * X` and `screen_y = camera_edge_y - scale * Z` if the camera is represented at the bottom of the radar.
- Do not add unexplained flips, sign changes, or rotations in the renderer. Coordinate conversion belongs in one tested function.

A planar homography maps image points to points on the same physical plane. It generally needs at least four non-collinear correspondences and a fixed camera. Validate it using known floor points across the visible area. Use a suitable ground-contact point, such as bottom-center of a person bounding box when feet are visible. The point is only an approximation when feet are occluded or boxes are loose.

A ground-plane homography does not give an airborne object's true 3D position. Monocular depth models often provide relative depth rather than calibrated metric depth. If using depth for airborne objects, document its scale and uncertainty and do not label the result as meters without validation.

Distance calculation is valid only when the coordinate units and frame support it. For two positions in a shared calibrated 2D plane:

`distance = sqrt((x2 - x1)^2 + (z2 - z1)^2)`

If coordinates are pixels or arbitrary units, label the result accordingly. Do not calculate physical separation by applying this formula to raw image pixels and presenting it as meters.

## Suggested data contracts

Use schemas appropriate to the existing codebase (Pydantic, dataclasses, TypeScript interfaces, or equivalent). Keep fields explicit and nullable where measurements may not exist.

### Observation

```json
{
  "track_id": "track-001",
  "timestamp": 0.0,
  "source": "camera",
  "bbox_xyxy": [100, 80, 160, 140],
  "pixel_position": {"x": 130.0, "y": 140.0},
  "relative_depth": null,
  "depth_method": null,
  "ground_position": null,
  "ground_coordinate_unit": null,
  "position_kind": "image_only",
  "detection_confidence": 0.0,
  "quality_flags": [],
  "occluded": null
}
```

The values above are placeholders. `timestamp` should use a consistent time basis; if wall-clock time is also needed, add a separate ISO-8601 field. Do not put a made-up default confidence into real observations. Omit or set unavailable values to `null`.

### Agent assessment

```json
{
  "track_id": "track-001",
  "created_at": "ISO-8601 timestamp",
  "assessment_label": "requires_review",
  "summary": "Short evidence-based explanation",
  "supporting_observation_ids": [],
  "uncertainties": [],
  "source_agent": "reasoning",
  "score": null,
  "score_method": null
}
```

Do not include a score unless the scoring method is defined and validated. Preserve links to source observations so an operator can inspect the evidence.

### Vision-language model output

Request strict JSON with fields such as:
- `visible_context`: concise description of visible scene context.
- `occlusion_or_ambiguity`: what is unclear or obscured.
- `qualitative_relative_position`: only if supported by the frame and supplied evidence.
- `evidence_used`: IDs or descriptions of provided observations.
- `uncertainties`: limitations and conflicts.
- `needs_more_observations`: boolean.

Do not ask the visual model to return authoritative coordinates, metric distances, track IDs, or threat intent from appearance alone. Parse and validate every response; handle malformed or unavailable output gracefully.

## Dashboard requirements

Primary layout:
- Header with system name and camera/tracking status.
- Two primary panels side by side when screen width allows: live webcam and radar. Stack on narrow screens.
- Webcam panel can overlay YOLO bounding boxes and track IDs if available.
- Radar panel displays track markers, short trails, coordinate axes, scale/units, camera/origin indicator, and optional calibration status.
- Small status summary: camera connection, active track count, last observation time, mapping mode.
- Selecting a track shows ID, recent history, pixel position, ground position if available, depth estimate and method if available, and quality flags.
- Display stale/lost/uncertain tracks differently from fresh valid observations, using both visual styling and text labels rather than color alone.
- Do not display sample values as if they were live. Mark simulated data clearly.

## Implementation workflow

1. Inspect the existing repository and run instructions before changing code. Preserve working camera and tracking functionality.
2. Identify the UI framework and existing webcam, tracker, queue, and event interfaces. Reuse existing components and styles where practical.
3. Implement the dashboard first with a simulated observation source and a stable data contract.
4. Implement live webcam rendering independently from the radar update loop.
5. Connect the tracker output to the shared observation schema.
6. Implement and unit-test coordinate conversion separately from radar rendering. Test axes, orientation, units, and known calibration points.
7. Add depth as optional data. Missing depth must not prevent a valid 2D or ground-plane marker from rendering.
8. Add the Tracking, Observation, and Reasoning agent interfaces with separate prompts/invocations if required. Keep LLM calls asynchronous, bounded, and outside the real-time loop.
9. Add logs for agent messages, timestamps, request reasons, response status, latency, and errors. Avoid logging unnecessary raw video or sensitive content.
10. Add a repeatable demo scenario and document how to launch it, calibrate the camera, switch between simulated/live data, and recover from camera failure.

## Testing checklist

### Camera and dashboard
- Camera opens and releases cleanly.
- UI remains responsive when inference or LLM calls are slow.
- Camera disconnect and reconnect are handled.
- Radar continues to render with simulated data if camera access fails.

### Tracking and coordinates
- Track IDs are preserved when supported and new tracks are handled safely.
- Missing, stale, or invalid coordinates do not crash rendering.
- Known floor calibration points map to expected radar positions within a documented tolerance.
- Left/right and near/far axes are verified with test points.
- Unit tests cover coordinate conversion and screen mapping separately.
- Pixel distance, relative depth, ground-plane distance, and 3D distance are not mislabeled.

### Agents
- Each required LLM-backed agent has its own prompt, invocation, typed contract, and tests.
- The coordinator routes messages without collapsing the agents into one prompt.
- Agent output is schema-validated; malformed output has a safe fallback.
- Evidence requests are bounded and do not create infinite loops.
- Predictions are withdrawn when history is insufficient or invalid.
- Every assessment links to evidence and states important uncertainty.

### Demo and evaluation
- A repeatable scenario demonstrates marker updates and trajectory trails.
- Simulated data is clearly labeled.
- Record update latency and end-to-end delay.
- Measure tracking continuity and position error when ground truth is available.
- Evaluate false alerts and assessment quality separately from detection accuracy.

## Engineering response style

When implementing a feature:
1. Briefly explain the planned change and which files/components it affects.
2. Make the smallest coherent change that fits existing project conventions.
3. Provide runnable code rather than pseudocode when asked to implement.
4. Explain how to test it, including expected output and common failure points.
5. Do not claim a feature works until it has been tested or clearly state what remains unverified.
6. Do not replace the entire architecture or introduce dependencies without explaining why.
7. Keep the current demo goal in view: reliable webcam + ground-plane radar visualization first, advanced analysis later.
