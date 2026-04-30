---
name: paper-analyzer
description: "Read and analyze an academic research paper using a structured 14-step protocol that emphasizes active, critical reading — including forming an independent solution before reading the paper's method, then comparing, and ending with a calibrated novelty/effectiveness/generalizability score and conference-acceptance verdict. Use this skill whenever the user asks to read, analyze, summarize, critique, score, or \"help me understand\" an academic paper, whether they provide a PDF, a URL (e.g., arXiv), or pasted text. Trigger even when the user just says things like \"what do you think of this paper\", \"can you go through this paper with me\", or drops a paper into the chat with little other context. Prefer this skill over a generic summary whenever the input is a research paper."
---

# Paper Analyzer

A structured protocol for reading academic papers actively rather than passively. The core insight: summarizing a paper is easy and low-value; the value comes from (a) clearly articulating the *problem* and why it's hard, (b) forming your own solution before peeking at the authors', and (c) honestly comparing. This skill enforces that discipline.

**Notably, don't associate the paper with my professional background. Just focus on the paper itself.**

**Output the analysis report as markdown file. Don't draw a figure.**

**According the user prompt excluding this skill to determine the report language**

## When to use

Use this whenever the user wants help reading, analyzing, or critiquing a research paper. Inputs can be:
- A PDF uploaded to the conversation
- A URL (arXiv abstract page, arXiv PDF, conference page, personal site)
- Pasted text of the paper or its abstract

If the user provides only an abstract or a short excerpt, run the protocol but note explicitly which steps are under-supported by the available text.

## How to get the paper content

- **PDF uploaded**: use the `pdf-reading` skill to extract text. Read the full body, not just the abstract.
- **URL**: use `web_fetch` on the URL. For arXiv, prefer the abstract page (`/abs/`) first to get metadata, then fetch the HTML version (`/html/`) if available, otherwise the PDF.
- **Pasted text**: use it directly.

Do not rely on prior knowledge of the paper even if the title looks familiar — actually read what's in front of you. Training-memory summaries are a common failure mode here.

## The 13-step protocol

Produce the analysis as a Markdown document with the exact headers below. Keep each section tight — substance over length. A typical paper warrants 1000–2000 words total; rich papers (multiple distinct contributions, large experimental sections) can justify up to ~2500. Padding is not. If you're approaching the upper end, ask yourself whether each paragraph is doing real work.

### 1. Research problem
State, in one or two sentences, the concrete problem the paper is trying to solve. Not the broad area ("video generation") but the specific problem ("generating temporally consistent videos longer than 10 seconds without drift"). If the paper is vague about this, say so.

### 2. Why the problem matters
Why should anyone care? Downstream applications, scientific understanding, unlocking other research, economic impact — whatever actually applies. Avoid generic filler ("AI is important"). If you can't find a compelling reason, note that honestly.

### 3. Why the problem is hard
What makes this problem resist easy solutions? Think about the structural reasons — combinatorial explosion, lack of supervision signal, distribution shift, physical/mathematical constraints, evaluation difficulty, etc. This section is where a lot of understanding comes from, so don't shortcut it.

### 4. [Thinking] How would *you* solve it?
**Critical instruction: before reading section 5 of this protocol, do not look at the paper's Related Work, Method, Approach, or Model sections. Read only the Abstract, Introduction and any problem-formulation section.** Then, based only on the problem and its challenges as you understand them from steps 1–3, propose how *you* would attack this problem. Be concrete — name the approach family (e.g., "I'd use a two-stage pipeline where stage one does X and stage two does Y"), sketch the training signal, and note the key design decisions you'd have to make.

This step has real epistemic value only if you commit to a proposal *before* seeing the paper's answer. A vague hedge ("I'd probably try some kind of learning-based approach") defeats the point. Pick something and defend it.

### 5. Prior work
What have other people tried for this problem? Summarize the main lines of attack — not a laundry list of citations, but the 2–4 conceptual families of prior approaches. Draw from the paper's Related Work section, but also use background knowledge if the paper's framing seems one-sided.

### 6. Why prior methods are unsatisfactory
What specific limitations do prior methods have that leave room for this paper? Be concrete: "Method X requires paired supervision which doesn't exist at scale"; "Method Y fails on out-of-distribution inputs because of Z". Avoid generic put-downs.

### 7. The proposed method
Now read the Method section. Describe what the paper actually proposes. Cover the architecture, training procedure, loss/objective, and any inference-time tricks. Be precise but not exhaustive — the goal is to convey the mechanism clearly, not reproduce the paper.

### 8. Key idea
Strip the method down to its core insight in 1–3 sentences. What is the *one thing* that, if you told it to a colleague, would let them reconstruct roughly how the method works? This is usually not the architecture but the conceptual move — "represent X as Y", "use A as a proxy signal for B", "decouple C from D".

### 9. [Thinking] Is the method novel or inspiring?
Honest judgment. Distinguish between:
- **Novel mechanism**: nobody has built this particular thing before.
- **Novel framing**: the mechanism is familiar but the problem lens is new.
- **Incremental**: a sensible combination of known ingredients, executed well.
- **Inspiring**: beyond novelty, does it suggest a new way of thinking that generalizes beyond this paper's setting?

These aren't mutually exclusive. Be willing to say "incremental but well-executed" — that's not an insult.

### 10. [Experiment] Performance and field impact
What does the paper actually demonstrate empirically? Focus on:
- Which benchmarks / datasets / tasks.
- Headline numbers vs. the strongest prior baselines — use quantitative deltas where possible (e.g., "improves FVD from 412 to 287, a ~30% reduction").
- Does it unlock something qualitatively new — a capability that simply didn't exist before — or is it a quantitative improvement on an existing capability?
- Ablations: which components actually matter?
- Honest caveats: cherry-picked settings, small test sets, unfair baselines, missing comparisons.

If the paper makes claims the experiments don't support, flag this clearly.

### 11. [Thinking] Your solution vs. the paper's
Revisit your step 4 proposal. Compare it to the paper's method head-to-head:
- Where did the paper do something you didn't think of, and why is that move good?
- Where did your approach have an advantage the paper's method lacks, if anywhere?
- Are you and the paper fundamentally in the same conceptual family, or in different ones?

This is the payoff step. It's the one that actually trains research taste, so don't skip it or reduce it to "the paper's method is better because it has more experiments."

### 12. Strengths & Weaknesses

List the paper's main strengths and weaknesses as parallel bullet lists. Aim for 2–4 bullets each — not a laundry list, not a single vague judgment. Each bullet should be one concrete, specific point (e.g., "Strength: the proposed loss is plug-and-play and requires no dataset relabeling" or "Weakness: all quantitative results are on a single in-house benchmark with no public access"). Avoid restating the key idea as a strength and avoid generic weaknesses like "limited to a single domain" unless that's actually a meaningful constraint given the paper's claims.

### 13. Conclusion
A tight wrap-up: the one-sentence version of what this paper contributes, what you believe vs. remain skeptical of, and what the most interesting open question is after reading it. Optionally, note which direction a good follow-up paper could take.

### 14. [Verdict] Scoring and conference-worthiness

Score the paper on three dimensions, each 0–100, using the rubrics below. For each score, give the number *and* one or two sentences justifying where on the rubric the paper sits and why — a bare number is not useful. Then answer the acceptance question.

**Novelty (0–100)** — how new is the contribution?
- **1 — Trivial**: Applies conventional methods to a new problem without significant modification.
- **10 — Innovative**: Solves the problem with a sophisticated or "cool" methodology.
- **100 — Groundbreaking**: Opens a new world or shifts the paradigm for the entire research community.

**Effectiveness (0–100)** — how much does the paper actually move the needle?
- **1 — Marginal**: Negligible performance gains; almost no perceptible difference.
- **10 — Incremental**: Improvement equivalent to roughly 1/10th of the field's average annual progress.
- **100 — Significant**: Improvement equivalent to a full year's worth of progress in the field.

**Generalizability & problem size (0–100)** — how broadly does the contribution reach?
- **1 — Specific**: Addresses or fixes a drawback of a single previous work.
- **10 — Sub-domain**: Solves a specific sub-task within a major research field.
- **100 — Universal**: Highly generalizable, applicable across a wide range of domains and scenarios.
- This dimension is about whether the *key idea* (not just the specific method instance) transfers to other problems. A narrow method with a transferable insight can still score high here.

Score each axis independently on its own merits — do not mentally average first. A paper can be highly novel but ineffective, or effective but narrow; let each score reflect that honestly. For each score, give the number *and* one or two sentences justifying where it sits and why.

**Total score = ∛(Novelty × Effectiveness × Generalizability)** (geometric mean).

The geometric mean preserves the "zero on one axis = low total" property while keeping the total on the same 0–100 scale as the individual axes, making it directly comparable to the reference table below. Score each axis honestly, then compute. Do not back-calculate from a target total.

Report the three scores, then the total, then the acceptance argument.

**Acceptance argument.** In 3–6 sentences, argue whether this paper deserves acceptance at a top-tier venue in its field (CVPR / ICCV / ECCV / NeurIPS / ICLR / ICML / SIGGRAPH / etc., whichever is the natural target). The argument should:
- Name the venue you're judging against.
- Identify the strongest case *for* acceptance and the strongest case *against*.
- Take a side. "It depends" is not a verdict. If the paper is borderline, say "weak accept" or "weak reject" and explain which way you lean and why.
- Calibrate against the realistic acceptance bar at that venue, not against an idealized one. Top venues accept lots of incremental-but-clean papers; the question is whether *this* paper clears *that* bar, not whether it's groundbreaking.

**Score reference table** (after scoring, compare your total here — this is a post-hoc sanity check, not a scoring target):

| Total | Benchmark |
|-------|-----------|
| ~10 | Weak accept — barely clears the bar |
| 20–30 | Typical accepted paper at CVPR / ICCV |
| 40–50 | Notable paper / spotlight / oral |
| 60+ | Best paper candidate |
| 100 | Transformer / ResNet tier |

## Style notes

- Prefer precise technical language over hedging. "The method fails on occluded scenes" is better than "the method may have some limitations in certain scenarios."
- It is okay — encouraged, even — to disagree with a paper's framing or claims. Do so specifically, not sweepingly.
- Do not reproduce more than short quoted phrases from the paper; paraphrase.
- If the user asks follow-up questions after the analysis (e.g., "expand on step 10", "what's the connection to [other paper]"), answer them naturally without redoing the whole protocol.

## Anti-patterns to avoid

- **Peeking at the method before step 4.** The single biggest failure mode. If you've already read the whole paper end-to-end before writing step 4, your step 4 is contaminated and you should say so rather than pretend.
- **Generic importance claims** in step 2 that could apply to any paper in the field.
- **Laundry-list related work** in step 5 instead of conceptual families.
- **Summarizing the abstract** instead of doing steps 1–3 yourself from the paper's content.
- **Refusing to judge** in steps 9 and 11 by retreating to "it depends" or "both have merits" without specifics.