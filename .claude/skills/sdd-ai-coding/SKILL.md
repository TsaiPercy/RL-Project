---
name: sdd-ai-coding
description: >
  Implements AI/ML research code following a SPEC.md and TODO.md produced
  by the sdd-ai-spec skill. Enforces strict coding standards: Google Style,
  mandatory type hints, visible tensor shapes with einops, no hardcoded
  hyperparameters or paths, device-agnostic code, defensive error handling,
  and single-responsibility modules. Also handles ongoing spec and TODO
  maintenance during implementation — severity-graded spec amendments,
  discovered TODO items, and a structured amendment log to keep SPEC.md
  as the living source of truth. Use this skill whenever the user says
  `/sdd-ai-coding`, "start coding from the spec", "implement the TODO",
  or references a SPEC.md + TODO.md pair and wants to begin writing code.
  Also trigger when the user asks to write ML training/evaluation code in
  a disciplined, specification-driven style, or when they need to update
  spec/TODO during an active implementation session.
---

# SDD AI Coding — Implementation from Spec

This skill takes `SPEC.md` + `TODO.md` (produced by `sdd-ai-spec`) and
implements the codebase task by task. Every line of code follows the
coding standards defined below.

---

## Inputs

Before writing any code, confirm you have:

1. **`SPEC.md`** — The full project specification
2. **`TODO.md`** — The task list with status tracking

Read both files completely before starting. Identify the first task
with status `☐` and begin there. After completing each task, update
its status to `☑` in `TODO.md`.

If the user provides neither, ask for them. If only a spec exists,
generate the TODO first (following sdd-ai-spec's TODO format).

---

## Task Execution Loop

The user drives what gets done. They can issue instructions in two ways:

### 1. Explicit TODO references

The user specifies which tasks to complete by ID or range:

- "Complete I-1 to I-4"
- "Do D-1, D-3, and M-1"

Read the referenced tasks, implement them all, update TODO.md, then
report what was done.

### 2. High-level instructions

The user describes what they want in natural language:

- "Set up the data pipeline"
- "Implement the encoder and decoder modules"
- "Get a basic training loop running"

In this case, read TODO.md and SPEC.md, identify which TODO items
correspond to the user's request, then implement them all. In the
completion report, list which TODO items you mapped the instruction to.

### Execution flow

```
User gives instruction (explicit IDs or natural language)
       │
       ▼
Map to TODO items → read relevant spec sections
       │
       ▼
Implement all mapped tasks (following all coding rules below)
       │
       ▼
Update TODO.md: ☐ → ☑ for each completed task
       │
       ▼
Append decisions to IMPLEMENT_RECORD.md
       │
       ▼
Report: completed tasks, decisions made, any concerns
```

### Dependency awareness

If a requested task depends on an uncompleted prerequisite (e.g., user
asks for M-1 but I-1 directory scaffold doesn't exist yet), complete the
prerequisite first and note it in the report. Infrastructure tasks (I-*)
should always be executed in order.

### IMPLEMENT_RECORD.md

This file is the log of all autonomous decisions and completed work.
Create it on the first coding session if it doesn't exist. Append to it
after each execution.

For the **Tasks Completed** table, copy each task's description verbatim
from `TODO.md` — do not summarize or paraphrase. This ensures the record
is self-contained and readable without cross-referencing TODO.md.

Format:

```markdown
# Implementation Record

## Session: [date or session number]

### Tasks Completed
| ID | Description | Status |
|----|-------------|--------|
| I-1 | Initialize project directory structure | ☑ |
| I-2 | Set up conda environment and dependencies | ☑ |
| I-3 | Create default config.yaml with all hyperparameters | ☑ |
| I-4 | Implement dataset loader with augmentation pipeline | ☑ |

### Decisions Made
| Task | Decision | Rationale |
|------|----------|-----------|
| I-2 | Used `environment.yml` over `requirements.txt` | Project uses conda-managed CUDA deps |
| I-3 | Set default `image_size: [256, 256]` | Spec §9 mentions 256px training data |

### Deferred Items Resolved
| Spec Reference | Original Question | Decision | Rationale |
|----------------|-------------------|----------|-----------|
| §11 [deferred] | "Which attention variant?" | Used flash attention v2 | Best VRAM/speed tradeoff for model size |

### Concerns
- [any issues, risks, or items that need user attention]
```

The user can review this file at any time to see what Claude decided
and override anything they disagree with.

---

## Coding Standards

All code produced by this skill must follow every rule below. These are
not suggestions — they are hard requirements.

### 1. Google Python Style Guide

Follow the Google Python Style Guide for all formatting, naming, and
documentation conventions. Key points:

- Module, function, and class docstrings are mandatory
- Use `snake_case` for functions/variables, `PascalCase` for classes,
  `UPPER_SNAKE_CASE` for module-level constants
- Maximum line length: 88 characters (Black-compatible)
- Imports organized: stdlib → third-party → local, alphabetized within groups

### 2. Type Hints Are Mandatory

Every function signature must include type hints for all parameters and
return values. No exceptions.

```python
# ✗ WRONG
def forward(self, x, mask=None):
    ...

# ✓ CORRECT
def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    ...
```

For complex types, use `TypeAlias` or define custom types:

```python
from typing import TypeAlias

BatchedImages: TypeAlias = torch.Tensor  # (B, C, H, W)
SegmentationMask: TypeAlias = torch.Tensor  # (B, 1, H, W)
```

### 3. Tensor Shapes Must Be Visible

Every tensor variable must have its shape documented in a comment at the
point of creation or transformation. Use the notation `(B, C, H, W)` etc.

```python
# ✓ CORRECT
x = self.encoder(images)  # (B, D, H//16, W//16)
x = rearrange(x, 'b d h w -> b (h w) d')  # (B, N, D) where N = H*W/256
attn_out = self.attention(x)  # (B, N, D)
```

### 4. Einops for All Tensor Manipulation

Use `einops` (`rearrange`, `reduce`, `repeat`) for all reshape, permute,
transpose, squeeze, unsqueeze, and repeat operations. Raw `.view()`,
`.reshape()`, `.permute()`, `.transpose()`, `.squeeze()`, `.unsqueeze()`,
and `.expand()` are **banned**.

```python
# ✗ BANNED
x = x.permute(0, 2, 3, 1).reshape(B, -1, C)

# ✓ REQUIRED
x = rearrange(x, 'b c h w -> b (h w) c')
```

The sole exception: `.flatten()` on a 1D result for a scalar or loss value.

### 5. Zero Hardcoded Values

Nothing that a researcher might want to change should be hardcoded in
`.py` files. This includes:

| Category | Examples | Where it goes |
|---|---|---|
| Hyperparameters | learning rate, weight decay, kernel size, dropout | `config.yaml` |
| Paths | data dirs, checkpoint dirs, output dirs | `config.yaml` or `argparse` |
| Seeds | random seed, torch seed | `config.yaml` or `argparse` |
| Training params | batch size, num epochs, grad accum steps | `config.yaml` |
| Architecture params | hidden dim, num heads, num layers | `config.yaml` |

Relative paths within the project (e.g., `./configs/default.yaml`) are
acceptable. Absolute paths are **banned**.

**Magic number rule**: If a numeric constant appears in the code, it must
either be:
- A mathematically obvious value (0, 1, 2, -1)
- Defined as a module-level `UPPER_SNAKE_CASE` constant with a comment
  explaining its source

```python
# ✓ CORRECT — constant defined at module top with source
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # from torchvision documentation
IMAGENET_STD = [0.229, 0.224, 0.225]   # from torchvision documentation

# ✗ WRONG — magic numbers inline
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
```

### 6. Device Agnostic

Hardcoded device strings are **banned**:

```python
# ✗ BANNED
x = x.cuda()
x = x.to("cuda:0")
model = model.cuda()

# ✓ CORRECT — device from config or argument
x = x.to(device)

# ✓ BEST — use PyTorch Lightning or Accelerate to handle device placement
```

The `device` variable must come from `config.yaml` or `argparse`, never
hardcoded. If using PyTorch Lightning, Accelerate, or similar frameworks,
rely on their built-in device management.

### 7. No Silent Failures

Defensive coding is mandatory. Data corruption, NaN values, shape
mismatches, and missing files must be caught explicitly.

```python
# In DataLoader / Dataset
def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
    try:
        image = load_image(self.paths[idx])
    except (IOError, OSError) as e:
        logger.warning(f"Corrupted image at index {idx}: {e}")
        # Return a fallback or skip — never silently pass corrupted data
        return self.__getitem__((idx + 1) % len(self))

    assert not torch.isnan(image).any(), (
        f"NaN detected in image at index {idx}"
    )
    return {"image": image}
```

```python
# After loss computation
loss = criterion(pred, target)
assert torch.isfinite(loss), f"Non-finite loss: {loss.item()}"
```

Use `assert` for invariants that indicate bugs. Use `try/except` with
logging for expected failure modes (corrupted files, network errors).

### 8. Single Responsibility Principle

Strict module separation:

| Module | Allowed | Forbidden |
|---|---|---|
| `Dataset` | Loading, augmentation, returning tensors | Normalization that depends on model, computing loss |
| `Model` | Forward pass → logits/features | Computing loss, data preprocessing, metrics |
| `Loss` | Computing loss from predictions + targets | Data loading, model forward pass |
| `Trainer` | Orchestrating train loop, logging, checkpointing | Implementing model logic, data transforms |
| `Evaluator` | Running eval loop, computing metrics | Training logic, model modifications |

If you notice a function doing two responsibilities, split it.

---

## Config Structure

Every project must have a base config. Use YAML with clear grouping:

```yaml
# configs/default.yaml

# --- Experiment ---
experiment:
  name: "experiment_name"
  seed: 42
  output_dir: "./outputs"

# --- Data ---
data:
  train_dir: "./data/train"
  val_dir: "./data/val"
  batch_size: 16
  num_workers: 4
  image_size: [224, 224]

# --- Model ---
model:
  name: "model_name"
  hidden_dim: 256
  num_heads: 8
  num_layers: 6
  dropout: 0.1

# --- Training ---
training:
  learning_rate: 1.0e-4
  weight_decay: 1.0e-5
  num_epochs: 100
  grad_accum_steps: 1
  precision: "bf16-mixed"

# --- Device ---
device:
  accelerator: "auto"  # auto / gpu / cpu
  devices: 1

# --- Logging ---
logging:
  wandb_project: "project_name"
  log_every_n_steps: 50
  val_check_interval: 1.0
```

Load configs with a library that supports merging and CLI overrides
(e.g., OmegaConf, Hydra, or simple argparse + yaml).

---

## File Conventions

- One class per file when the class is a major component (model, dataset,
  loss). Small utility classes can share a file.
- File names match the primary class: `class FeatureExtractor` →
  `feature_extractor.py`
- Every `__init__.py` exports the public API of its package.
- Test files mirror source structure: `src/models/vae.py` →
  `tests/models/test_vae.py`

---

## Checklist Before Committing Each Task

Before marking a task as `☑`, verify:

- [ ] All functions have type hints and docstrings
- [ ] All tensor shapes are annotated in comments
- [ ] All reshape/permute ops use einops
- [ ] No hardcoded hyperparameters, paths, seeds, or device strings
- [ ] Defensive checks for data loading and loss computation
- [ ] Each module has a single responsibility
- [ ] Magic numbers are defined as named constants with source comments
- [ ] Code runs with `python -c "import module"` without errors

---

## Handling Deferred Ambiguities

If `SPEC.md` contains `[deferred]` items, do NOT pause to ask the user.
Instead:

1. Make the best decision you can based on spec context, common practice,
   and the project's overall direction
2. Document the decision in a code comment at the relevant location
3. Log it in `IMPLEMENT_RECORD.md` under "Deferred Items Resolved"
   with the original question, your decision, and your rationale

The user reviews `IMPLEMENT_RECORD.md` asynchronously and can override
any decision afterward. This keeps the coding flow uninterrupted.

---

## Spec & TODO Maintenance

`SPEC.md` is a living document. Implementation will reveal gaps,
incorrect assumptions, and new requirements. This section defines how
to handle changes so the spec stays the single source of truth without
blocking coding momentum.

### Severity Levels

Every spec/TODO change falls into one of three levels. Choose the
level based on impact scope, not effort.

#### Level 1 — Minor (fix inline, keep moving)

Small corrections that don't change architecture or interfaces:
parameter renames, typo fixes, minor default-value adjustments,
clarifying wording.

**Action:**
1. Update `SPEC.md` directly. Mark the changed text with
   `[impl-updated]` inline so diffs are easy to spot.
2. Update code to match (or vice versa).
3. Append a one-liner to `SPEC_MODIFICATION.md` (see format below).
4. Log in `IMPLEMENT_RECORD.md` under "Spec Amendments".

No need to pause or ask the user.

#### Level 2 — Medium (fix, then report)

Changes that affect an API surface, add new sub-tasks, or address
edge cases the spec didn't anticipate — but do NOT alter the overall
architecture or expand scope significantly.

**Action:**
1. Update `SPEC.md`: mark modified text with `[impl-updated]` and
   include the **original text** in a `<!-- before: ... -->` HTML
   comment immediately above so the user can see what changed.
2. If new tasks are needed, add them to `TODO.md` using discovered
   IDs derived from the parent task: e.g., `M-3` spawns `M-3.1`,
   `M-3.2`. Mark them `☐`.
3. Append to `SPEC_MODIFICATION.md`.
4. Log in `IMPLEMENT_RECORD.md` under "Spec Amendments" with
   before/after and rationale.
5. In the session completion report, **explicitly list every spec
   change** so the user sees them without digging through files.

#### Level 3 — Major (stop and ask)

Fundamental architecture changes, core assumptions invalidated, or
scope expansion beyond what the spec covers.

**Action:**
1. **Do NOT modify SPEC.md or write code** for the affected area.
2. Write a **Spec Amendment Proposal** in `IMPLEMENT_RECORD.md`:

```markdown
### Spec Amendment Proposal — [short title]
**Affected sections:** §X, §Y
**Problem:** [what broke or what's missing]
**Impact:** [which modules/tasks are affected]
**Proposed change:** [concrete spec text or design sketch]
**Alternatives considered:** [if any]
**Recommendation:** [your preferred option and why]
```

3. Report to the user and wait for confirmation.
4. After approval, update `SPEC.md`, log the amendment in
   `SPEC_MODIFICATION.md`, generate new TODO items, then proceed
   with implementation.

### User-Initiated Requirement Changes

When the user adds new requirements during the coding phase
(e.g., "I also want feature X"), do NOT jump straight to coding.
Follow the upstream-first rule:

```
User requests new feature / change
       │
       ▼
Update SPEC.md (add or modify relevant section)
       │
       ▼
Generate new TODO items (or modify existing ones)
       │
       ▼
Append to SPEC_MODIFICATION.md
       │
       ▼
Implement the new/modified tasks
```

This ensures SPEC.md always remains upstream of code. No feature
should exist in code without a corresponding spec entry.

### Discovered TODO Items

Tasks discovered during implementation that were not in the original
`TODO.md` use a sub-ID scheme derived from the parent task:

```
M-3        ← original task from spec
M-3.1      ← discovered sub-task during M-3 implementation
M-3.2      ← another discovered sub-task
```

If a discovered task is not clearly a child of any existing task,
use the prefix `X-` (for "extra") with a sequential number:

```
X-1        ← standalone discovered task
X-2        ← another standalone discovered task
```

Always log the reason for creating the new task in
`IMPLEMENT_RECORD.md`.

### SPEC_MODIFICATION.md

A standalone changelog file that tracks all modifications to `SPEC.md`.
Create it on the first spec modification if it doesn't exist. Every
spec change at any severity level must be logged here.

```markdown
# Spec Modification Log

| # | Date | Section | Change Summary | Trigger | Level |
|---|------|---------|----------------|---------|-------|
| 1 | 2026-04-30 | §5 Model | Changed attention: vanilla → flash-attn v2 | impl-updated (M-3) | L1 |
| 2 | 2026-04-30 | §7 Eval  | Added FID metric to eval pipeline | user request | L2 |
| 3 | 2026-05-01 | §3 Data  | Redesigned augmentation pipeline for video | amendment proposal #1 | L3 |
```

Columns:
- **#**: Sequential amendment number (never reuse).
- **Date**: Date of the change.
- **Section**: Which `SPEC.md` section(s) were modified.
- **Change Summary**: One-line description of what changed.
- **Trigger**: What caused the change — `impl-updated (task-id)`,
  `user request`, `amendment proposal #N`, or `deferred resolution`.
- **Level**: L1 / L2 / L3.

---

## Important Reminders

- **Read SPEC.md thoroughly** before writing any code. The spec is the
  source of truth for architecture, interfaces, and data flow.
- **Complete everything the user asked for** before reporting back.
  Don't pause mid-execution to ask unless something is genuinely
  blocking (e.g., a missing file that should exist).
- **Record all decisions** in `IMPLEMENT_RECORD.md`. The user may not
  be present during execution — this file is how they review your
  autonomous choices.
- **Infrastructure first**. If a requested task depends on I-* tasks
  that haven't been done, complete them first and note it in the report.
- **Shape comments are not optional**. If a tensor exists, its shape
  must be documented. This is the single most common source of bugs in
  ML code, and visible shapes prevent them.
- **Spec stays upstream of code**. Never let code contain features,
  interfaces, or behaviors that are not reflected in `SPEC.md`. If
  implementation diverges from spec, update the spec first (following
  the severity levels above), then continue coding.
- **For deliberate spec revision outside coding**, direct the user
  to `sdd-ai-spec-update` instead of handling it within this skill.