---
name: sdd-ai-spec
description: >
  Specification-Driven Development (SDD) for AI/ML research projects.
  Generates a structured SPEC.md and TODO.md from a research idea through
  an iterative ambiguity-resolution workflow. Use this skill whenever the
  user types `/sdd-ai-spec`, says "develop an AI project with SDD",
  "write a spec for my research project", "SDD-style planning",
  or any request that involves turning a research idea into a detailed
  implementation specification before coding. Also trigger when the user
  wants to plan an ML experiment pipeline, design a training/evaluation
  system, or structure a research codebase from scratch.
---

# SDD AI Spec — Specification-Driven Development for AI Research

This skill turns a research idea into a detailed, implementation-ready
specification (`SPEC.md`) and a task list (`TODO.md`), through iterative
ambiguity resolution with the user. No code is written in this skill —
coding is handed off to a separate coding skill afterward.

---

## Inputs from the User

Ask the user to provide four things (they don't need to be long):

1. **Problem** — What problem does this research address?
2. **Idea** — What is the core idea or hypothesis?
3. **Method** — What approach do you plan to take?
4. **Evaluation** — How will you know it works?

If the user provides them inline or in a document, extract and confirm
before proceeding.

---

## Workflow Overview

```
User provides Problem / Idea / Method / Evaluation
        │
        ▼
┌─────────────────────────────┐
│  Phase 1: Research-Level    │
│  Ambiguity List v1          │──▶ output: AMBIGUITY.md
│  (method, scope, eval)      │
└──────────┬──────────────────┘
           │  ◀── user edits AMBIGUITY.md and provides it back
           ▼
┌─────────────────────────────┐
│  SPEC v1 + Ambiguity v2     │──▶ output: SPEC.md, AMBIGUITY.md
│  (adds impl-level Qs)       │
└──────────┬──────────────────┘
           │  ◀── user edits AMBIGUITY.md and provides it back
           ▼
┌─────────────────────────────┐
│  SPEC v2 + Ambiguity v3     │──▶ output: SPEC.md, AMBIGUITY.md
│  (convergence check)        │
└──────────┬──────────────────┘
           │  ◀── user confirms or answers
           ▼
┌─────────────────────────────┐
│  Final SPEC.md + TODO.md    │──▶ output: SPEC.md, TODO.md
└─────────────────────────────┘
```

The soft limit is **3 rounds**. After each round, judge whether remaining
ambiguities affect the spec's core structure. If they are only
implementation details, mark them `[deferred]` and finalize.

---

## Phase 1: Ambiguity List (Research-Level)

After receiving the user's four inputs, generate the first ambiguity list
as an `AMBIGUITY.md` file. This round focuses only on research-level questions:

- **Method & Scope** — edge cases, failure modes, assumptions
- **Evaluation** — metrics, baselines, datasets, what counts as success
- **Core Idea** — theoretical grounding, novelty, related work gaps

Do NOT ask implementation questions (architecture details, library choices,
VRAM) in this round — those come in Phase 2.

### ⚠️ CRITICAL: File-Based Workflow — Do NOT Ask Questions in Chat

**NEVER** ask ambiguity questions directly in the conversation. Instead:

1. Write ALL questions into the `AMBIGUITY.md` file with answer placeholders
2. Save the file and present it to the user
3. Tell the user: "Please edit AMBIGUITY.md to fill in your answers, then
   provide the file back to me."
4. **Stop and wait.** Do not proceed until the user provides the edited file.

The user will edit the markdown file to fill in their answers. This keeps
the Q&A structured and avoids losing context in long chat sessions.

### AMBIGUITY.md Format

Write to `AMBIGUITY.md`:

```markdown
# Ambiguity List — [Project Name]
## Round 1 (Research-Level)

### Method & Scope
- **Q1**: [question]
  - **A1**: <!-- YOUR ANSWER HERE -->
- **Q2**: [question]
  - **A2**: <!-- YOUR ANSWER HERE -->

### Evaluation
- **Q3**: [question]
  - **A3**: <!-- YOUR ANSWER HERE -->

### Core Idea
- **Q4**: [question]
  - **A4**: <!-- YOUR ANSWER HERE -->
```

Output the file, then **stop and wait** for the user to edit and return it.
Do NOT proceed until the edited file is provided. Do NOT ask the questions
again in chat — they belong in the file only.

---

## Phase 2+: Spec Drafting with Implementation-Level Ambiguities

Once the user provides the edited `AMBIGUITY.md` with Round 1 answers,
write the first draft of `SPEC.md` AND append a new round of ambiguities
to `AMBIGUITY.md`. This round adds implementation-level questions:

- **Architecture** — module boundaries, module independence (can each
  module be tested alone?), inter-module communication methods (direct
  call vs callback vs shared artifact), shared types that cross module
  boundaries, input/output tensor shapes
- **Data** — formats, preprocessing, augmentation
- **Infrastructure** — VRAM budget, environment, dependencies
- **Experiment logistics** — ordering, sanity checks vs full runs

Each subsequent round: update `SPEC.md` with answers, add new Qs to
`AMBIGUITY.md` if any, mark resolved Qs as `[resolved]`, mark detail-only
Qs as `[deferred]`, and mark questions the user explicitly cannot answer
yet (e.g., "not sure yet", "TBD", "decide later") as `[pending]`.

Again: write all new questions into `AMBIGUITY.md`, save and present it,
then stop and wait for the user to edit and return it. **Never ask
ambiguity questions in the chat.**

---

## Convergence Check (after each round, max 3 rounds)

After updating the spec, evaluate:

> "Do any remaining unanswered questions affect the **structure** of the
> spec (module boundaries, shared types, inter-module communication
> methods, experiment design, evaluation protocol)?
> Or are they only about **implementation details** (exact hyperparameters,
> specific library versions, visualization style)?"

- If structural → continue to next round
- If detail-only → mark all remaining as `[deferred]`, finalize spec
- If round 3 reached → finalize regardless, marking unresolved as `[deferred]`

Tell the user your assessment explicitly so they can override if needed.

---

## SPEC.md Structure

Every spec must contain exactly these sections in this order. Write each
section with enough detail that a competent ML engineer could implement
the system without further clarification (except `[deferred]` items).

````markdown
# [Project Title] — Specification

## 1. Overview
Brief description of what this research project does and why it matters.

## 2. Problem
The specific problem this project addresses. Include:
- Why existing approaches are insufficient
- What gap this fills

## 3. Modality
Describe the input and output of the system:
- Input format, shape, domain (e.g., "RGB video, T×3×H×W, from dashcam")
- Output format, shape, domain (e.g., "per-frame segmentation mask, T×1×H×W")
- Any intermediate representations

## 4. Core Idea
The central insight or hypothesis, stated concisely. This should be
understandable to someone in the field in under 60 seconds.

## 5. Method
Detailed description of the proposed approach. Include:
- Step-by-step pipeline
- Key design choices and their rationale
- How this differs from prior work

## 6. Method Scope
- Where does this method apply? (data domains, scales, conditions)
- Known failure cases or limitations
- Boundary conditions

## 7. Risk Assessment
For EVERY identifiable risk factor, list:

| Risk Factor | Severity (H/M/L) | Likelihood (H/M/L) | Mitigation |
|---|---|---|---|
| [description] | [H/M/L] | [H/M/L] | [strategy] |

Be exhaustive. Include risks from: method design, data quality,
compute budget, evaluation validity, scope creep.

## 8. Evaluation Protocol
- Primary and secondary metrics
- Baseline methods (with citations/links if possible)
- SOTA comparison targets
- Statistical significance testing plan (if applicable)

## 9. Datasets
### Training Data
- Name, source, size, license
- Data characteristics (resolution, distribution, class balance)
- Preprocessing pipeline

### Evaluation Data
- Name, source, size
- Train/val/test split strategy
- Any domain shift considerations

## 10. Experiment Design
Before listing experiments, state:
- **Hypotheses**: What you expect to observe and why
- **Counter-hypotheses**: What would invalidate the approach
- **Experiment ordering strategy**: State whether to
  (a) build full pipeline first then ablate (lower deadline risk), or
  (b) run sanity checks first to validate assumptions (lower wasted-effort risk),
  and justify the choice.

Then list each experiment:
### Experiment N: [Name]
- **Hypothesis**: ...
- **Expected result**: ...
- **If violated**: What it might mean, and next steps
- **Setup**: Which modules, which data, which config

## 11. System Architecture & Components

**Design principle: maximize module independence.** Each module should
be self-contained with a single, well-defined responsibility. Modules
communicate only through explicitly declared interfaces — never by
reaching into another module's internals. If two modules need to share
data, that data type must be defined in a shared types contract.

- Architecture diagram (describe in text; suggest a mermaid diagram)
- Module list with responsibilities and interfaces
- Inter-module data flow (what tensor/data shapes cross boundaries)

### 11.1 Shared Types

Define ALL data structures that cross module boundaries in a single
`types.py` (or equivalent) file. Every type that appears as an input
or output of more than one module MUST be listed here. This is the
**contract** between modules — changing a shared type requires updating
all modules that use it.

```python
# src/types.py — Shared type definitions
@dataclass
class ExampleBatch:
    """A training batch flowing from DataModule → Model → Loss."""
    images: Tensor      # (B, C, H, W)
    labels: Tensor      # (B,)
    metadata: dict      # per-sample info

@dataclass
class PredictionResult:
    """Model output flowing from Model → Evaluator."""
    logits: Tensor      # (B, num_classes)
    features: Tensor    # (B, D) — intermediate features for visualization
```

List each shared type with:
- Name and fields (with tensor shapes / data types)
- Which modules produce it
- Which modules consume it
- Invariants (e.g., "labels are always 0-indexed integers")

### 11.2 Module Specifications

Design each module to be **independently testable** — it should be
possible to unit-test a module by mocking its dependencies using only
the shared types defined in §11.1.

For each module, specify:
- Class name
- **Responsibility** (single sentence — if you need "and", split it)
- Input types and shapes (reference §11.1 shared types)
- Output types and shapes (reference §11.1 shared types)
- Key methods (public API only)
- Dependencies on other modules (list which modules, and through which
  shared types they communicate)
- **Unit test cases** — list the specific test cases that should be written
  for this module. Each entry should state: what is given (mock inputs),
  what is asserted (expected output or behaviour), and what failure mode
  it guards against. At minimum, cover: (1) the happy path, (2) at least
  one edge case or boundary condition, (3) one invalid/malformed input.
  Example:
  ```
  - test_forward_correct_shape: given a valid ExampleBatch, assert output
    logits have shape (B, num_classes). Guards against shape regression.
  - test_forward_empty_batch: given B=0, assert no crash and empty output.
    Guards against edge case in downstream evaluation loop.
  - test_invalid_input_type: given non-tensor input, assert ValueError raised.
    Guards against silent type coercion bugs.
  ```

### 11.3 Inter-Module Communication

For every pair of modules that interact, explicitly state:

| From → To | Communication Method | Data Exchanged (Shared Type) | Sync/Async | Notes |
|---|---|---|---|---|
| DataModule → Model | function call (forward pass) | `ExampleBatch` | sync | batched |
| Model → Evaluator | function call | `PredictionResult` | sync | |
| Trainer → Logger | callback / event | `MetricsDict` | async | fire-and-forget |

Communication methods include:
- **Direct function call** — simplest; caller invokes callee's public method
- **Callback / event hook** — for decoupled notifications (e.g., logging, checkpointing)
- **Shared file / artifact** — for offline handoff between stages (e.g., saved checkpoint → evaluation script)
- **Config-driven wiring** — modules don't know each other; a top-level script or config connects them

For each pair, justify why this communication method was chosen over
alternatives.

### 11.4 Pseudo Config
```yaml
# List all adjustable hyperparameters, grouped by module
module_a:
  param_1: <default>  # description
  param_2: <default>  # description
```

## 12. Experiment Implementation Architecture
For each experiment in §10:
- Which modules from §11 are used
- Which are excluded or mocked
- How they connect (simplified pipeline diagram)

## 13. Visualization Opportunities
List what can be visualized to aid debugging and insight:
- Per-module intermediate outputs
- Training curves and metrics
- Qualitative result comparisons
- Failure case analysis displays

## 14. VRAM Budget Estimation
For each major operation:

| Operation | Model/Data Size | Est. VRAM | Notes |
|---|---|---|---|
| [operation] | [sizes] | [estimate] | [batch size, precision] |

Include total estimate and whether multi-GPU / gradient checkpointing
is needed.

## 15. Environment & Dependencies
- Python version
- Key packages with version constraints
- Potential conflicts
- Whether multiple conda environments are needed and why

## 16. Directory Structure
```
project-root/
├── configs/
├── data/
├── src/
│   ├── types.py        # shared type definitions (§11.1)
│   ├── models/
│   ├── datasets/
│   ├── losses/
│   ├── ...
├── scripts/
├── experiments/
├── outputs/
├── tests/
├── environment.yml
├── README.md
├── TODO.md
└── SPEC.md
```
Adapt to the specific project. Every directory gets a one-line purpose.
````

---

## TODO.md Structure

After finalizing the spec, generate `TODO.md`. Use a table format,
grouped into these blocks **in this order**:

**Important**: Collect ALL items marked `[deferred]` or `[pending]` in
`AMBIGUITY.md`, as well as any unresolved questions from the final round,
into the **Pending Decisions** section at the end of `TODO.md`. For each
entry, identify which spec section it affects and which tasks it blocks.
This ensures nothing silently falls through the cracks — the human has a
single place to revisit open decisions when they're ready.

```markdown
# TODO — [Project Title]

## Infrastructure
| # | Task | Status | Notes |
|---|------|--------|-------|
| I-1 | Create directory scaffold | ☐ | Per spec §16 |
| I-2 | Write environment.yml / requirements.txt | ☐ | Per spec §15 |
| I-3 | Write base config.yaml with all hyperparams | ☐ | Per spec §11 pseudo config |
| I-4 | Implement visualization scripts | ☐ | Per spec §13 |
| I-5 | Set up test runner (pytest config, fixtures for shared types) | ☐ | Per spec §11.1 |

## Unit Tests
One task per module. Implement ALL test cases listed in that module's §11.2
entry. Tick off only when every listed test case passes.

| # | Task | Status | Notes |
|---|------|--------|-------|
| T-1 | Write unit tests for [ModuleA] | ☐ | Per spec §11.2 test cases |
| T-2 | Write unit tests for [ModuleB] | ☐ | Per spec §11.2 test cases |

## Data Preparation
| # | Task | Status | Notes |
|---|------|--------|-------|
| D-1 | ... | ☐ | |

## Sanity Checks
| # | Task | Status | Notes |
|---|------|--------|-------|
| S-1 | ... | ☐ | |

## Evaluation Protocol
| # | Task | Status | Notes |
|---|------|--------|-------|
| E-1 | ... | ☐ | |

## Baseline Setup
| # | Task | Status | Notes |
|---|------|--------|-------|
| B-1 | ... | ☐ | |

## Main Method
| # | Task | Status | Notes |
|---|------|--------|-------|
| M-1 | ... | ☐ | |

## Ablations
| # | Task | Status | Notes |
|---|------|--------|-------|
| A-1 | ... | ☐ | |

## Pending Decisions
Items marked `[deferred]` (detail-level, deferred by Claude) or
`[pending]` (user could not decide yet) during ambiguity resolution,
plus any open questions that did not get fully resolved within 3 rounds.
These need a human decision before the relevant task can proceed.

| # | Decision Needed | Affects | Blocking Tasks | Resolution |
|---|----------------|---------|----------------|------------|
| P-1 | [describe the open question] | [spec §N] | [task IDs, e.g. M-2, E-1] | <!-- DECISION HERE --> |
| P-2 | ... | | | |

When a decision is made, fill in the **Resolution** column and update
the corresponding spec section and blocked tasks accordingly.
```

Every task must trace back to a specific spec section (note the section
reference). Tasks should be granular enough that each takes roughly
1–4 hours of focused work.

---

## Handoff to Coding

After `SPEC.md` and `TODO.md` are finalized and saved to files:

1. Tell the user: "Spec and TODO are complete."
2. Ask: "Do you have a coding style skill you'd like to use for
   implementation? If so, which one?"
3. Instruct: "To start coding, provide `SPEC.md` and `TODO.md` to your
   coding skill. It will begin from the first uncompleted task in TODO.md."

This skill does NOT write any code. The boundary is clean:
- **sdd-ai-spec** outputs: `SPEC.md`, `TODO.md`, `AMBIGUITY.md`
- **coding skill** inputs: `SPEC.md`, `TODO.md`

---

## Important Reminders

- **Match the user's language.** All chat responses AND file content
  (`AMBIGUITY.md`, `SPEC.md`, `TODO.md`) should be written in the same
  language the user is using. If the user writes in Chinese, write
  everything in Chinese. If in English, write in English. Mirror the
  user's language choice consistently throughout the entire workflow.
- **Always write to files**, not inline. Every ambiguity list and spec
  version should be saved as a `.md` file. This prevents context window
  overflow across iterations.
- **NEVER ask ambiguity questions in the chat session.** All questions
  MUST be written into `AMBIGUITY.md` with answer placeholders. The user
  will edit the file and provide it back. This is the single most important
  rule in this skill. If you catch yourself typing a numbered list of
  questions in the chat, STOP — put them in `AMBIGUITY.md` instead.
- **Stop and wait** after outputting each `AMBIGUITY.md`. Never assume
  answers or proceed without the user providing the edited file.
- **Two-phase ambiguity**: Research-level first (Round 1), then
  implementation-level (Round 2+). Never mix them in Round 1.
- **Be exhaustive in Risk Assessment** (§7). A hand-wavy "it might not
  work" is not acceptable. List every concrete risk factor you can identify.
- **Experiment hypotheses come before experiment design**. Each experiment
  exists to test a specific hypothesis. If you can't state the hypothesis,
  the experiment shouldn't exist yet.
- **Module independence is a first-class concern in §11.** Every module
  should be independently testable. If Module A cannot be tested without
  instantiating Module B, the coupling is too tight — refactor by
  introducing a shared type as the interface. Always define shared types
  BEFORE specifying individual modules.