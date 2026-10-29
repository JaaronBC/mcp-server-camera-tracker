---
name: human-maintainable-code
description: Refactor or build a codebase so a team of humans can read, navigate, and maintain it. Use this skill whenever the user says a project is hard to read, messy, overcomplicated, or needs to be handed to other people or a team; when asked to clean up, simplify, restructure, or refactor code; when the code has unclear names, giant files, deep nesting, over-abstraction, or duplicated logic; or when starting a new project that several people will own. Use it even if the user never says "maintainability" - any request about making code understandable to humans qualifies.
---

# Human-Maintainable Code

The goal: a new teammate can open this project, understand what it does in 15 minutes, find any piece of behavior in under a minute, and change it without fear.

Agent-written code tends to fail humans in predictable ways: vague names, huge files that do everything, layers of abstraction nobody asked for, and clever tricks instead of obvious code. This skill is the corrective.

## Core principle

**Optimize for the reader, not the writer.** Code is read far more than it is written. When choosing between "short and clever" and "longer and obvious," choose obvious. When choosing between "flexible for the future" and "simple for today," choose simple for today.

## Workflow

Do not start rewriting immediately. Work in this order.

### 1. Survey before touching anything
- Map the project: entry points, main data flow, external dependencies, config.
- List files by size. Anything over ~300 lines is a candidate for splitting.
- Find the pain: run a quick scan for duplicated logic, deeply nested code, functions over ~50 lines, and names like `data`, `temp`, `handler`, `utils`, `manager`, `helper`.
- Note what currently works. Identify tests, or the manual way to check behavior.

### 2. Lock in behavior first
Refactoring must not change what the program does. Before restructuring:
- Run existing tests. If none exist, write a few simple tests (or a script) that capture current outputs for key paths.
- Record how to run the project end-to-end. Verify it runs now.

### 3. Write a short refactor plan and share it
Before big moves, give the user a brief plan: the proposed folder structure, the main renames, which files will be split or merged, and what will be deleted. Let them object. Large renames and moves are expensive to undo.

### 4. Refactor in small, verified steps
One kind of change at a time (rename, then split, then simplify), re-running tests after each. Keep changes in separate commits if git is available, so reviewers can follow them. Never mix a behavior change into a cleanup step. If you spot a bug, note it and fix it separately.

### 5. Document for newcomers
Finish with a README and an architecture overview (see "Documentation" below).

---

## Naming conventions

Names are the most important documentation. A good name removes the need for a comment.

**Rules**
- Name things for what they *are or do* in the domain, not how they're implemented. `unmatched_tracks` beats `list2`.
- Be specific. `parse_detection_results` beats `process`. `retry_count` beats `n`.
- Functions are verbs (`load_config`, `merge_tracks`). Variables and classes are nouns (`track_id`, `FrameBuffer`). Booleans read as questions (`is_active`, `has_pending_frames`).
- Include units when relevant: `timeout_seconds`, `distance_meters`, `frame_width_px`.
- No abbreviations unless universal in the domain (`id`, `url`, `fps` are fine; `trk_mgr_cfg` is not).
- Avoid junk-drawer names: `utils`, `helpers`, `common`, `misc`, `manager`, `handler`, `processor`, `data`, `info`, `stuff`. If you need one, the code probably belongs somewhere more specific.
- One concept, one name. Don't call the same thing `track`, `target`, and `object` in different files. Pick one and use it everywhere.
- Match the language's convention (snake_case in Python, camelCase in JS/Java, PascalCase for types) and stay consistent.

**File and folder names**
- A file name should tell you what's inside without opening it. `track_association.py`, not `core2.py`.
- Folders group by *feature or responsibility*, not by technical type, once the project grows. Prefer `detection/`, `tracking/`, `alerts/` over a single `models/`, `controllers/`, `services/` split that scatters one feature across five folders.
- Never leave `final`, `new`, `old`, `v2`, `copy`, `test2`, or `backup` in names. Use version control for history.

**Renaming safely**
Rename with the tooling (IDE rename, search-and-replace with word boundaries) and re-run tests. Update docs, config, and string references (including things like dynamic imports and config keys) that a refactor tool might miss.

---

## File and function size ("overloading")

A file or function that does many unrelated things is overloaded. Humans can't hold it in their head.

**Targets (guidelines, not laws)**
- Files: one clear responsibility, usually under 300 lines. If you can't describe the file in one sentence without "and", split it.
- Functions: do one thing, usually under 40 lines, ideally fits on one screen. Few parameters (aim for 4 or fewer; if more, group them into a named object).
- Classes: a small, coherent set of responsibilities. A class named `SystemManager` that touches everything should be broken up.
- Nesting: no more than ~3 levels deep. Flatten with early returns (guard clauses) and by extracting helpers.

**How to split**
1. Identify the distinct responsibilities (e.g. reading input, transforming data, drawing output, talking to the network).
2. Move each into its own file/module with a specific name.
3. Keep the original entry point thin: it should read like a table of contents, calling well-named steps in order.
4. Don't split mechanically by line count. Split along real seams, so each piece makes sense on its own.

**Signs a function should be split**
- You need a comment to separate "phases" inside it.
- It has sections that could be tested independently.
- It mixes levels of detail (high-level orchestration next to low-level byte handling).
- It does something *and* reports/logs/saves the result.

**Avoid the opposite failure**: don't shatter code into dozens of tiny one-line functions and files that force readers to jump around constantly. A function is worth extracting when it has a meaningful name and a clear purpose.

---

## Overcomplication

Complexity should be earned. Remove anything that exists "just in case."

**Delete or simplify**
- **Unused code**: dead functions, commented-out blocks, unused imports, stale feature flags, abandoned experiments. Version control remembers them.
- **Speculative abstraction**: interfaces with one implementation, factories that build one thing, plugin systems nobody plugs into, config options nobody changes. Inline them.
- **Needless layers**: wrapper functions that only call another function, classes that only hold one method, inheritance hierarchies where a plain function would do. Prefer composition and plain functions over deep inheritance.
- **Cleverness**: dense one-liners, nested ternaries, metaprogramming, magic decorators, and "tricks." Rewrite as plain, step-by-step code.
- **Duplication**: when the same logic appears three or more times, extract it once with a good name. But two similar-looking pieces of code aren't always the same concept; don't force them together if they'll evolve differently.
- **Magic values**: replace unexplained numbers and strings with named constants or config (`MAX_TRACK_AGE_FRAMES = 30`).
- **Global state and hidden coupling**: pass dependencies in explicitly; avoid modules that silently modify shared state.
- **Over-configuration**: if there are 40 settings, most should be sensible defaults. Group settings in one documented place.

**Simplicity checklist for every function**
- Could a new teammate understand this in one read?
- Is there a simpler way to write it with the same behavior?
- Does every parameter, branch, and abstraction here have a reason I can state?

If you can't state the reason, remove it (after verifying with tests).

**Error handling**: handle errors where you can do something useful about them. Avoid blanket `try/except` that swallows problems silently. Fail loudly with a clear message that says what went wrong and what to check.

---

## Project structure

Aim for a layout where the structure itself explains the system. Example shape (adapt to the project):

```
project-name/
├── README.md              # what it is, how to install, how to run
├── docs/
│   ├── architecture.md    # how the pieces fit together
│   └── contributing.md    # conventions, how to test, how to submit changes
├── src/
│   ├── main.py            # thin entry point
│   ├── config.py          # all settings in one place
│   ├── <feature_a>/       # one folder per major responsibility
│   └── <feature_b>/
├── tests/                 # mirrors src/ structure
├── scripts/               # one-off tools, clearly separated
└── requirements / package manifest
```

Principles:
- One obvious entry point.
- Configuration in one place, not scattered.
- Dependencies point one way (high-level code depends on low-level, not in circles). Remove circular imports.
- Keep experiments, notebooks, and throwaway scripts out of the main source tree.

---

## Comments and documentation

**Comments explain *why*, not *what*.** If a comment restates the code, delete it and improve the name instead. Keep comments for: non-obvious decisions, workarounds, domain rules, links to specs or tickets, and warnings ("order matters here because...").

**Docstrings**: every public function/class gets a short description of purpose, inputs, outputs, and notable errors. One to three lines is often enough.

**README (required)** covers: what the project does and why, how to install, how to run, how to run tests, folder overview, and where to look next.

**Architecture doc (required for multi-person projects)**: a short walkthrough of the main data flow, the key modules and what each owns, and the important design decisions with reasons. A simple diagram helps. Write it for someone who has never seen the code.

**Glossary**: if the domain has specialized terms, define them once in the docs and use them consistently in code.

---

## Consistency and tooling

Humans on a team need the code to look the same everywhere.
- Add a formatter and linter (e.g. Black/Ruff for Python, Prettier/ESLint for JS) and commit the config, so style arguments disappear.
- Add type hints or types where the language supports them. They are executable documentation.
- Pin dependencies and document setup so "it works on my machine" doesn't happen.
- Add a basic test suite and make it one command to run. Tests also teach newcomers how the code is supposed to behave.

---

## Rules for the agent while refactoring

- **Preserve behavior.** Verify before and after. If behavior must change, say so explicitly and do it as a separate step.
- **Don't invent features.** The job is clarity, not new functionality.
- **Prefer deleting to adding.** The best refactor often makes the project smaller.
- **Explain your choices.** When you make a non-obvious structural decision, record why (in the architecture doc or commit message).
- **Ask when the domain is unclear.** If you can't tell what a piece of code is for, ask the user rather than guessing at a name or deleting it.
- **Stop and report** if a refactor turns out riskier than expected (no tests, unclear behavior, tangled dependencies). Propose a smaller step instead.
- **Leave a trail.** Summarize what changed so reviewers and teammates can follow it.

## Final report to the user

When finished, provide:
1. The new project structure (a short tree).
2. A list of the major renames (old name to new name) so the team can update their mental map and any external references.
3. What was deleted or simplified, and why.
4. How to run the project and the tests.
5. Anything left unresolved or worth a human decision (known bugs, unclear code, risky areas).

## Quick self-review checklist

Before declaring done, confirm:
- [ ] Project runs and tests pass; behavior is unchanged.
- [ ] No file has more than one clear job; no giant "does everything" file remains.
- [ ] No function a reader must scroll through to understand.
- [ ] Names are specific, consistent, and free of `utils`/`helper`/`data`/`temp`.
- [ ] No dead code, commented-out blocks, or unused imports.
- [ ] No abstraction exists without a current, real use.
- [ ] Magic numbers are named; settings live in one place.
- [ ] README and architecture doc exist and are accurate.
- [ ] Formatter/linter configured and the code passes.
- [ ] A new teammate could find where to make a typical change in under a minute.
