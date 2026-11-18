# Coding Agent — Efficient Development Instructions

Work efficiently while maintaining correctness and existing architecture.

## Context

This project is an agentic real-time camera radar system using YOLO, ByteTrack, MCP, and Tracking, Observation, and Reasoning LLM agents.

Current focus: occlusion recovery and short-term trajectory prediction.

Existing components include:

- Persistent identity lifecycle and reassociation safeguards.
- TrackingEstimateProposalEngine.
- Bounded prediction evidence runtime store.
- Tracking decision IPC and MCP evidence tools.

Reuse existing implementations. Do not redesign them without evidence of a problem.

## Working Rules

1. **Inspect narrowly.** Search relevant symbols first. Read only necessary functions and surrounding code. Avoid rereading entire files or the repository.

2. **Avoid redundant research.** Use previous audit findings unless the current code contradicts them.

3. **Make minimal edits.** Change only files necessary for the requested task. Avoid unrelated refactoring, cleanup, or new abstractions.

4. **Explain briefly.** Before editing, state the intended change and reason in 2–4 sentences. Do not produce lengthy architectural reports.

5. **Use focused tests.** Run directly relevant deterministic tests first. Expand testing only when failures or cross-component changes justify it.

6. **Avoid repeated tool calls.** Batch related searches, file reads, and tests where practical. Do not repeatedly inspect unchanged code.

7. **Limit retries.** Investigate a failed command before retrying. Do not repeatedly run the same failing test without a hypothesis.

8. **Preserve safety.** Never bypass identity mutation guards, evidence validation, timestamps, coordinate compatibility, or uncertainty checks.

9. **No expensive execution by default.** Do not invoke LLM providers, cameras, full-system experiments, or long-running tests without explicit approval.

10. **Stop when complete.** Do not continue optimizing, researching, or proposing unrelated changes after acceptance criteria pass.

## Response Format

Keep progress messages minimal.

Final report:

- Changed: files and one-line summary.
- Tests: commands and pass/fail counts.
- Risks: only unresolved issues.
- Next: one recommended task.

Target under 150 words unless important failures require additional explanation.

Token efficiency must never come at the expense of correctness or safety.
