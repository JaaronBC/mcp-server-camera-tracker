Development Agent — Efficient Workflow

Objective

Minimize development-agent credit usage, tool calls, repeated exploration, and unnecessary execution while maintaining correctness.

Investigation

Search only files relevant to the current task.

Reuse previously identified architecture and file paths.

Do not repeatedly read unchanged files.

Prefer targeted searches over repository-wide searches.

Distinguish confirmed defects from hypotheses.

Stop investigating once enough evidence exists to recommend a change.

Implementation

Explain proposed changes before editing.

Modify only the approved files and functions.

Prefer minimal changes over refactoring.

Do not introduce unrelated features.

Verify edits using a focused diff.

Testing

Run the smallest relevant test first.

Do not run broad test suites without approval.

Do not repeat passing tests unless code affecting them changes.

Do not launch the webcam or external models without approval.

Prefer deterministic provider-free tests.

Stop after the first successful targeted validation.

Output

Keep progress messages minimal.

Report only changed files, test results, unresolved issues, and next steps.

Avoid repeating large logs or code already inspected.

Do not commit without approval.

Stop Conditions

Stop when the requested acceptance criteria are met.

Stop and ask before expanding scope.

Do not continue optional improvements automatically.
