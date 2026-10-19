---
name: movement-analysis
description: Interpret raw object tracking data from the real-time camera pipeline.
---

# Movement Analysis Skill

## Purpose

Analyze raw tracking observations produced by YOLO and ByteTrack.
Use the observed data to describe object movement, without assuming
that every movement indicates a threat.

## Workflow

1. Read the movement event returned by the tracking pipeline.
2. Use the available MCP tool to load the associated capture JSON.
3. Inspect the timestamps, track IDs, and center coordinates.
4. Compare positions over time to infer movement.
5. Report the findings in a structured, concise format.

## Interpretation rules

- In image coordinates, x increases to the right.
- In image coordinates, y increases downward.
- Describe direction using changes in x and y over time.
- Estimate average image-plane speed in pixels per second when
  timestamps and positions support the calculation.
- Distinguish net displacement from total path distance.
- Describe changes in direction or apparent speed only when
  enough observations support the inference.
- Do not infer real-world speed, altitude, or distance from
  pixel coordinates alone.
- Do not claim a threat based on movement alone.
- State uncertainty and identify missing information.
- When analyzing a movement event, inspect the observations returned by
  analyze_object_movement. Use elapsed_seconds or timestamp to determine
  the observation duration. Use center_x and center_y to infer direction
  and calculate image-plane speed. Do not claim timestamps are missing
  without checking the returned observations.

## Response format

Present the movement analysis as a Markdown table with one row per track.

Include these columns:

Track ID: The unique tracking ID.

Primary Direction: The primary direction of movement (left, right, up, down, diagonal, or stationary).

Duration (s): The observation duration in seconds.

Average Speed (px/s): The estimated average image-plane speed in pixels per second, when available.

Calculate average speed using the net displacement between the first and last observed positions divided by the elapsed observation time. Do not confuse net average speed with total path speed.

Include every track present in the capture. If a value cannot be reliably calculated, use "Unavailable" rather than inventing a value. After the table, provide at most one short sentence noting important uncertainty or limitations. Do not add headings, calculations, or lengthy explanations.
