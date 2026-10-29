---
name: counter-uas-end-to-end-integration
description: Enforces a connected Counter-UAS application architecture linking camera perception, YOLO/ByteTrack, depth and coordinate estimates, the MCP coordinator, three distinct agents, and the live camera/radar dashboard. Requires explaining why changes matter before editing and carefully removing only confirmed unnecessary files.
---

# Counter-UAS End-to-End Integration Skill

## Mission

Maintain one working Counter-UAS research system in which the live camera, perception pipeline, shared track observations, MCP tools, three agents, and radar dashboard exchange consistent data.

The dashboard is not an independent demo. It must display the same validated track state produced by the real perception pipeline and show relevant assessment events produced by the agent workflow.

The system is for monitoring and human decision support. Do not infer hostile intent from movement alone or autonomously initiate physical countermeasures.

## 1. Explain before changing

Before making any code change, first explain:

- **Problem:** What is missing, broken, duplicated, or incorrect?
- **Why it matters:** How does it affect the end-to-end system?
- **Planned change:** Which exact files, classes, and functions will change?
- **Data-flow impact:** Which producer, consumer, event, or interface will be connected or modified?
- **Verification:** Which tests or launch checks will prove the change works?

Wait for user approval before destructive cleanup, major architecture changes, or changes that may remove working behavior. Do not edit first and explain afterward.

Prefer the smallest coherent change. Do not perform a broad rewrite just to make the code look cleaner.

## 2. Required end-to-end architecture

Maintain the following logical data flow, adapting names and boundaries to the existing repository:

Camera capture
→ YOLO detections
→ ByteTrack identities and track history
→ optional depth estimate and quality metadata
→ image-space coordinates
→ calibrated ground-plane coordinates when valid
→ canonical timestamped observation/track state
→ MCP coordinator and Tracking Agent
→ Observation/Behavior Agent
→ Reasoning/Decision-Support Agent
→ validated assessment events
→ dashboard camera overlay, radar, track details, and assessment display.

The diagram describes logical responsibilities, not a requirement that every component be a separate process. Reuse the current implementation and avoid duplicate pipelines.

The camera overlay and radar must use consistent track IDs and observation timestamps. The dashboard must receive relevant assessment updates, not just raw detections. If the application has a process boundary, explicitly define and test the IPC, API, UDP, or event contract that crosses it.

## 3. Ownership boundaries

### Camera/perception/tracking runtime

Owns:
- Camera capture and release.
- YOLO inference and ByteTrack updates.
- Track IDs, bounding boxes, detection confidence, and timestamped history.
- Depth inference and quality metadata when available.
- Pixel coordinates and their selected-point definition.
- Ground-plane transforms when calibration is valid.
- Emission of canonical observation updates.

Keep the real-time capture and rendering path responsive. Do not wait synchronously for LLM responses.

### MCP server and coordinator

Owns:
- MCP tool entry points and application lifecycle commands.
- Routing validated observation data to the agents.
- Shared track history or a clearly defined interface to it.
- Typed-message and evidence-reference validation.
- Scheduling asynchronous agent work.
- Bounded evidence requests, retry limits, timeouts, and error reporting.
- Publishing validated assessment events to the dashboard-facing interface.

The coordinator is ordinary application code. It is not a fourth reasoning agent and must not replace the independent reasoning assigned to the agents.

### Three agents

**Tracking Agent**
- Uses perception evidence, including YOLO/ByteTrack outputs, image coordinates, available depth estimates, and calibrated ground-plane estimates.
- Uses its own LLM invocation if the required architecture calls for an LLM-backed Tracking Agent.
- Helps interpret ambiguous visual context and association conflicts.
- Proposes interpretations or associations with uncertainty and references to the supplied evidence.
- Must not invent or silently overwrite track IDs, measured coordinates, timestamps, or confidence values.

**Observation/Behavior Agent**
- Receives timestamped track histories, motion measurements, and quality flags.
- Has its own prompt, LLM invocation, structured output contract, and validation when a three-LLM design is required.
- Interprets candidate movement patterns and explains supporting evidence, uncertainty, contradictions, and alternatives.
- Does not invent numeric measurements or infer hostile intent from movement alone.

**Reasoning/Decision-Support Agent**
- Receives Tracking and Observation outputs plus relevant underlying evidence.
- Has its own prompt, LLM invocation, structured output contract, and validation when a three-LLM design is required.
- Cross-checks agent claims against source observations and identifies disagreements.
- Produces an evidence-linked operator assessment and may request targeted additional observation.
- Must not blindly accept upstream interpretations or fabricate risk scores.

Agents must not simply repeat one another. Preserve provenance and source observation IDs so claims can be checked against measurements. Record unresolved disagreements rather than silently hiding them.

## 4. One canonical observation contract

Reuse an existing suitable schema if available. Do not create a second competing schema without explaining why.

The canonical observation should represent, where available:

- `observation_id`
- `track_id`
- `timestamp`
- `source_frame_id` or equivalent source reference
- `bbox_xyxy`
- `pixel_position` and selected-point definition
- `detection_confidence`
- `relative_depth`, `depth_method`, and `depth_quality`
- `ground_position`, coordinate frame, unit, and calibration status/version
- `motion_measurements` and their units
- `quality_flags`
- source/provenance references
- occlusion or stale-data state when known

Use the existing project's schema conventions, such as Pydantic or dataclasses. Make unavailable measurements nullable or explicitly unavailable. Never fill missing measurements with plausible-looking fake values.

## 5. Coordinate and depth integrity

Keep distinct:
- Pixel/image-space coordinates.
- Relative monocular depth.
- Calibrated two-dimensional ground-plane coordinates.
- True three-dimensional position.

Do not label relative depth as meters unless its metric scale has been validated. A planar homography does not recover an airborne object's true 3D position. Do not label a floor projection as the object's actual 3D position. Do not convert raw pixel distance into physical distance without a valid calibrated mapping.

Use one documented, tested coordinate transformation. The radar renderer should not apply unexplained flips, rotations, sign changes, or scale factors.

## 6. Dashboard responsibilities

The dashboard should:
- Render the live camera feed.
- Display YOLO bounding boxes and track IDs from the shared live observations.
- Render radar markers and trajectory trails from the same track IDs and timestamped state.
- Show axes, origin, units/scale, calibration status, and uncertainty.
- Show selected-track details, available depth information, quality flags, and recent history.
- Display validated agent assessments, supporting evidence references, timestamps, and uncertainty.
- Show camera, tracking, depth, calibration, MCP connection, and agent status.
- Distinguish stale, lost, uncertain, simulated, and live data clearly.

The dashboard may request start/stop operations, select tracks, and subscribe to updates. It must not independently own agent orchestration, run a duplicate reasoning cycle, or maintain an unrelated authoritative track history.

Do not require the dashboard to be a separate process unless the existing architecture needs it. Separate logical responsibilities first; process boundaries are an implementation choice.

## 7. MCP, transport, and event flow

Inspect how the repository currently uses MCP tools, UDP, JSON snapshots, event logs, queues, or APIs. Do not assume that components are connected because they use similarly named data.

For any data path, identify:
1. Producer and exact call site.
2. Message/schema and key fields.
3. Transport or storage interface.
4. Consumer and exact call site.
5. Timestamp/track identity handling.
6. Error, stale-data, and reconnection behavior.

A latest-value JSON snapshot can be useful for a prototype but may become inconsistent with another independently written snapshot. UDP can be appropriate for local communication, but messages need explicit timestamps, IDs, schema/version expectations, and stale-data handling. JSONL logs are useful for auditing but are not automatically a live event bus.

Keep existing transports if they work and the contracts can be made reliable. Replace them only when there is a concrete reason. Avoid introducing a broker, database, or new service without explaining its value and operational cost first.

## 8. Required integration audit

Before implementing an integration change, trace a real observation end to end:

1. Where the camera frame is captured.
2. Where YOLO/ByteTrack produce a detection and track ID.
3. Where depth, pixel position, and ground-plane estimates are attached.
4. Where the canonical observation is published or stored.
5. How the Tracking Agent receives it.
6. How Observation and Reasoning receive evidence and return validated results.
7. How MCP agents/coordinator route requests and assessments.
8. How the dashboard consumes the same track update and assessment event.

Report every missing connection explicitly. Do not substitute a mock-only pathway for a missing live integration. Use simulated data only for a clearly labeled fallback or tests.

## 9. Duplicate and unnecessary file cleanup

Keep the project focused, but do not delete files merely because they look unused.

Before suggesting deletion:
- Search imports, call sites, entry points, scripts, tests, configuration references, and documented commands.
- Determine whether the file is part of live tracking, the dashboard, agent workflow, calibration, test fixtures, or demo setup.
- Explain why it is redundant or obsolete and what replaces its behavior.
- Obtain user approval before destructive or uncertain cleanup.
- After approved removal, update references, tests, and documentation.

Prefer removing proven duplicates and dead code. Preserve working functionality, calibration assets, user data, and useful test fixtures. Do not remove the main movement-summary LLM or any other pathway until its role and consumers have been traced and the user has approved the plan when behavior changes.

## 10. Testing and verification

Test the smallest relevant unit and the end-to-end connection.

Where applicable, verify:
- Camera capture and graceful release.
- YOLO/ByteTrack output reaches the canonical observation interface.
- Radar and camera overlay show the same track IDs and timestamps.
- Depth and coordinate units are accurately represented.
- Known calibration points map to expected radar positions.
- Missing/stale data does not crash rendering.
- Agent inputs contain relevant evidence and provenance.
- Each required LLM-backed agent has a distinct invocation and validated output.
- Agent assessments reach the dashboard.
- Evidence requests are bounded.
- Slow or failed LLM calls do not block frame capture/rendering.
- Simulated data is visibly distinct from live data.
- Existing MCP tools and launch commands still work.

Mocked unit tests do not prove the live camera, actual model endpoint, or inter-process transport works. Clearly report which checks used mocks and which were run against live components.

Never claim integration or tests succeeded without observing the actual result.

## 11. Definition of done

- [ ] The purpose and importance of the proposed change were explained before editing.
- [ ] Only the necessary files were changed.
- [ ] One canonical observation path connects live perception to shared track state.
- [ ] Camera overlay and radar use consistent observations.
- [ ] MCP/agents consume relevant timestamped evidence from that same path.
- [ ] Validated agent assessments can reach the dashboard.
- [ ] Coordinate units, depth limitations, and uncertainty are preserved.
- [ ] Required agents have distinct responsibilities and real LLM invocation paths.
- [ ] Real-time capture and display remain responsive.
- [ ] Simulation is clearly labeled and cannot silently masquerade as live data.
- [ ] File deletions were justified and approved when needed.
- [ ] Relevant tests and launch checks were run and honestly reported.
- [ ] Remaining gaps and unverified behavior are listed.

## 12. Required response format

Before changes, report:

**Problem:** What is missing or incorrect?

**Why it matters:** How does it affect the connected pipeline?

**Planned change:** Exact files/functions and their responsibilities.

**Data path:** Producer → schema/transport → consumer.

**Verification:** Tests and launch checks.

After changes, report:

**Files changed/deleted:** Exact paths and reasons.

**Integration proof:** Exact call sites, message fields, and route from perception through agents to dashboard.

**Tests run:** Commands and observed results.

**Unverified items:** Anything requiring a real camera, model endpoint, calibration, or manual check.

Do not start a broad rewrite. Optimize for a small, testable, connected end-to-end system—not a tidy-looking diagram with disconnected components.
