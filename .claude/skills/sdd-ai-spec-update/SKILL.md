---
name: sdd-ai-spec-update
description: >
  Surgically updates an existing SPEC.md and its corresponding TODO.md
  outside of a coding session. Classifies change requests by operation
  type (modify, add, remove, reframe) and impact scope (localized,
  cascading, fundamental) to choose the right editing strategy. Handles
  TODO delta (new tasks, updated tasks, obsolete tasks, needs-redo),
  impact analysis for cross-section changes, and logs everything to
  SPEC_MODIFICATION.md. Use this skill whenever the user says
  `/sdd-ai-spec-update`, "update the spec", "change the spec",
  "revise the spec", "I want to modify SPEC.md", "add X to the spec",
  "remove X from the spec", or any request to edit an existing SPEC.md
  outside of active coding. Do NOT use for creating a spec from scratch
  (use sdd-ai-spec) or for spec changes discovered during coding
  (use sdd-ai-coding's built-in maintenance).
---

# SDD AI Spec Update — Surgical Spec Revision

This skill takes an existing `SPEC.md` + `TODO.md` and applies
targeted updates based on user requests. It does not rewrite the
spec from scratch — it makes precise changes and tracks their
impact.

---

## Inputs

Before making any changes, confirm you have:

1. **`SPEC.md`** — The current project specification
2. **`TODO.md`** — The current task list with status tracking
3. **`SPEC_MODIFICATION.md`** (if it exists) — Prior change history

Read all three files completely before proceeding.

If the user doesn't have a `SPEC.md`, this is the wrong skill —
direct them to `sdd-ai-spec` to create one first.

---

## Step 1 — Classify the Change

Every update request is classified along two dimensions.

### Operation Type (what kind of change)

| Type | Description | Example |
|------|-------------|---------|
| **Modify** | Change existing content | "Switch backbone from ViT to CNN" |
| **Add** | Introduce new section or component | "Add an evaluation metric" |
| **Remove** | Delete or deprecate existing content | "Drop the ablation study on X" |
| **Reframe** | Change naming, definitions, or framing without altering structure | "Rename 'encoder' to 'feature extractor' throughout" |

### Impact Scope (how far the change reaches)

| Scope | Description | Action |
|-------|-------------|--------|
| **Localized** | Affects 1–2 sections, no downstream ripple | Edit directly, report after |
| **Cascading** | Crosses multiple sections or TODO items | Run impact analysis first, confirm with user, then edit |
| **Fundamental** | Challenges core assumptions or architecture | Write an update proposal, wait for user approval; if scope is too large, recommend re-running `sdd-ai-spec` instead |

Announce the classification to the user before proceeding:

> "This is a **Modify × Cascading** change — it touches §3 Data
> and §5 Model, and affects TODO items D-2, M-1, M-3. Here's the
> full impact. Should I proceed?"

---

## Step 2 — Impact Analysis (Cascading and Fundamental only)

For Localized changes, skip to Step 3.

For Cascading and Fundamental changes, produce an impact report
before editing anything:

```markdown
### Impact Analysis — [short title]

**Requested change:** [user's request in one line]
**Classification:** [Type] × [Scope]

**Affected SPEC.md sections:**
- §X [section name] — [what changes and why]
- §Y [section name] — [what changes and why]

**Affected TODO.md items:**
- [task-id]: [what happens — modified / obsolete / needs-redo]
- [task-id]: [what happens]

**New TODO items needed:**
- [brief description of each new task]

**Risks / things to watch:**
- [anything the user should be aware of]
```

Wait for user confirmation before proceeding. If the user wants
to adjust scope, re-run the analysis with the adjusted parameters.

For Fundamental scope: if the impact analysis reveals that >50%
of spec sections need rewriting, explicitly recommend going back
to `sdd-ai-spec` for a fresh spec. Don't force surgical editing
on a patient that needs reconstruction.

---

## Step 3 — Edit SPEC.md

Apply changes to `SPEC.md` following these rules:

### Editing principles

- **Surgical, not cosmetic.** Only touch what needs to change.
  Don't reformat, reorder, or "improve" unrelated sections.
- **Mark changes visibly.** Add `[updated]` inline next to
  modified text so diffs are easy to spot.
- **Preserve originals for non-trivial changes.** For Modify and
  Remove operations that alter meaning (not just typos), include
  the original text in an HTML comment above:
  `<!-- before: [original text] -->`
- **Maintain section numbering.** Don't renumber existing sections.
  New sections go at the most logical position with a new number
  or sub-number (e.g., §5.1).

### Module spec checklist (§11.2)

If the change involves **adding or modifying a module** in §11.2, verify
that the module entry includes a **Unit test cases** section before saving.
Each test case should state: given (mock inputs) → assert (expected output)
→ failure mode it guards against. Minimum coverage: happy path, one edge
case, one invalid input. If the user's request doesn't include test cases,
flag it explicitly:

> "You're adding/modifying a module spec. The §11.2 format requires a
> **Unit test cases** list. Please provide at least 3 test cases, or I
> can draft placeholders for you to fill in."

Do not silently omit the unit test cases section — an incomplete module
spec is a spec defect.

### Per operation type

**Modify:**
Edit the target text in place. Add `[updated]` marker and
`<!-- before: ... -->` comment.

**Add:**
Insert the new section at the appropriate location. Mark with
`[added]` inline. No need for a `<!-- before -->` comment since
there's no prior version.

**Remove:**
Do NOT delete text from `SPEC.md`. Instead, wrap the removed
content in a clearly marked block:

```markdown
<!-- [removed] — reason: [brief rationale]
[original content stays here for reference]
-->
```

This preserves history. The content is invisible in rendered
markdown but recoverable if the decision is reversed.

**Reframe:**
Apply the rename/redefinition consistently across all occurrences
in `SPEC.md`. List every changed occurrence in the update report
so the user can verify completeness.

---

## Step 4 — Update TODO.md

Apply the minimum necessary changes to `TODO.md`. Never regenerate
the entire file.

### TODO delta rules

| Situation | Action | Marker |
|-----------|--------|--------|
| New task needed | Add with `A-` prefix (update-originated): `A-1`, `A-2`, ... | `☐` |
| Existing task description changed | Edit in place | `[updated]` after task title |
| Existing task no longer needed | Keep in place, don't delete | `[obsolete: reason]`, status stays as-is |
| Completed task must be redone | Change status back | `☐ [needs-redo: reason]` |

### ID scheme for new tasks

- If clearly a child of an existing task: use sub-ID (e.g., `M-3`
  spawns `A-M-3.1`)
- If standalone: use sequential `A-` IDs (`A-1`, `A-2`, ...)

### Ordering

Insert new tasks in the most logical position relative to existing
tasks (e.g., a new data task goes near other D-* tasks), not just
appended at the bottom.

---

## Step 5 — Log to SPEC_MODIFICATION.md

Append an entry for every change. Create the file if it doesn't
exist.

```markdown
# Spec Modification Log

| # | Date | Section | Change Summary | Trigger | Type × Scope |
|---|------|---------|----------------|---------|---------------|
| 1 | 2026-04-30 | §5 Model | Backbone: ViT → CNN | user request | Modify × Cascading |
| 2 | 2026-04-30 | §9 Eval  | Added LPIPS metric | user request | Add × Localized |
| 3 | 2026-05-01 | §6 Ablation | Removed ablation on X | user request | Remove × Localized |
```

Columns:
- **#**: Sequential number (continues from existing entries, never reuse).
- **Date**: Date of the change.
- **Section**: Which `SPEC.md` section(s) were modified.
- **Change Summary**: One-line description.
- **Trigger**: `user request`, `post-experiment revision`, or
  other brief description of why.
- **Type × Scope**: The classification from Step 1.

---

## Step 6 — Report

After all edits are complete, provide a summary:

```
## Update Report

**Classification:** [Type] × [Scope]
**Requested change:** [one-line summary]

### SPEC.md changes
- §X: [what changed]
- §Y: [what changed]

### TODO.md changes
- Added: A-1 [description]
- Updated: M-3 [what changed]
- Obsolete: D-4 [reason]

### SPEC_MODIFICATION.md
- Entry #N logged

### Anything to watch
- [risks, follow-up items, or related sections to review]
```

---

## Scope Boundaries

| Situation | Which skill to use |
|-----------|--------------------|
| No SPEC.md exists yet | `sdd-ai-spec` |
| Coding session, spec issue discovered mid-implementation | `sdd-ai-coding` (built-in maintenance) |
| Outside coding, deliberate spec revision | **this skill** |
| Revision so large that >50% of sections need rewriting | Go back to `sdd-ai-spec` |

If the user's request crosses a boundary (e.g., they ask to update
the spec AND start implementing), handle the spec update first with
this skill, then hand off to `sdd-ai-coding` for implementation.
Spec always stays upstream of code.