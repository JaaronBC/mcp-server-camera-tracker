---
name: phase2-agentic-reasoning
description: Guides Phase 2 development of the Counter-Drone MCP system, including autonomous multi-agent reasoning, agent collaboration, MCP tool integration, identity verification, and multi-occlusion handling. Use when auditing, planning, implementing, or testing Phase 2 functionality.
---

# Phase 2 — Agentic Reasoning and MCP Integration

## Purpose

Guide the implementation of Phase 2 of the Counter-Drone system without unnecessarily restructuring the existing architecture.

The objective is to enable independent LLM-backed agents to collaborate, request evidence, use MCP tools, and make traceable decisions.

## Core Architecture

Preserve:

- YOLO and ByteTrack detection/tracking.
- Application-level identity registry.
- Tracking, Observation, and Reasoning agents.
- Operator/Orchestration component.
- MCP server and existing tools.
- Ground-plane radar and dashboard.
- Existing identity reassociation safeguards.

## Agent Responsibilities

**Tracking Agent**
- Investigates identity discontinuities and occlusions.
- Requests tracking and visual evidence.
- Proposes evidence-supported identity associations.
- Never directly modifies the identity registry.

**Observation Agent**
- Independently evaluates Tracking Agent proposals.
- Analyzes movement and competing candidates.
- Verifies, rejects, or marks proposals inconclusive.
- Requests additional evidence when justified.

**Reasoning Agent**
- Synthesizes observations and verification results.
- Considers uncertainty and alternative explanations.
- Produces traceable assessments.

**Operator/Orchestration**
- Coordinates investigations and agent requests.
- Handles disagreement and unresolved cases.
- Controls investigation lifecycle.
- Routes verified decisions through existing registry safeguards.

## MCP Tool Integration

Before migrating Python functionality into MCP tools:

1. Inspect existing implementations.
2. Determine whether agent-controlled invocation is beneficial.
3. Prefer bounded, on-demand, structured operations.
4. Keep continuous tracking and rendering processes as services.
5. Reuse existing Python functions through MCP wrappers.
6. Avoid duplicate implementations.
7. Preserve existing callers and lifecycle behavior.
8. Document tool inputs, outputs, permissions, and failure handling.

Do not convert every Python script into an MCP tool.

## Agent Collaboration

- Use structured, evidence-linked messages.
- Preserve correlation IDs and observation timestamps.
- Validate identity snapshot generation/version.
- Distinguish measurements from hypotheses.
- Allow independent disagreement and inconclusive results.
- Prevent agents from fabricating observations.
- Never bypass registry ownership or verification safeguards.

## Multiple Occlusions

Support multiple pending investigations.

Evaluate competing identity candidates and prevent conflicting ownership transfers.

Do not force identity matches when evidence is insufficient.

## Phase Boundaries

**Phase 2:** Agent autonomy, tool access, structured collaboration, correctness, and deterministic integration testing.

**Phase 3:** Latency optimization, throughput, concurrency performance, and provider benchmarking.

**Phase 4:** Trajectory prediction and threat assessment.

Do not expand Phase 2 into unrelated work.

## Development Workflow

For each Phase 2 task:

1. Inspect the relevant source files.
2. Explain the current behavior.
3. Identify the smallest necessary change.
4. Explain why the change improves agent autonomy or collaboration.
5. Identify affected interfaces and dependencies.
6. Present the implementation plan.
7. Wait for approval before significant production changes.
8. Implement only the approved scope.
9. Run focused deterministic tests.
10. Report changes, test results, and remaining limitations.

## Constraints

- Do not modify partner-owned drone detection model or model-selection logic.
- Do not rewrite the identity registry.
- Do not bypass verification safeguards.
- Do not rebuild the dashboard unnecessarily.
- Do not introduce unnecessary agent frameworks.
- Do not delete files without approval.
- Do not run provider-backed LLM tests without approval.
- Do not perform unrelated Git operations.
- Do not optimize latency during Phase 2.

## Completion Requirements

Phase 2 should demonstrate:

- Three independently LLM-backed specialist agents.
- Autonomous evidence gathering and appropriate tool selection.
- Structured agent communication.
- Independent identity verification.
- Reasoning that handles uncertainty and disagreement.
- Safe handling of multiple occlusions.
- MCP integration where beneficial.
- Preservation of real-time tracking independence.
- Passing deterministic integration tests.

Mocked agent tests establish workflow correctness, not actual LLM reasoning quality. Report this distinction explicitly.