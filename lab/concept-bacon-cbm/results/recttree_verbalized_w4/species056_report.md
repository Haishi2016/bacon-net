# Rose-breasted Grosbeak — Disjunctive Graded-Logic Rule

**Model:** recttree OCBM (**W4**, hardness-anneal, seed 0), CUB-200
**Species 56 — Rose-breasted Grosbeak** · test recall 96.7% · faithful rule: 28 literals, 8 ops
**Rule source (W4):** `species056_rule.json` (pruned to top-3 children / node, depth 3)

## Plain-Language Statement
> It calls a bird a **Rose-breasted Grosbeak** when it sees **any** of several
> red-breasted, cone-billed, white-bellied signatures — a rule built on
> alternatives rather than a single checklist.

## Overview
This is a **disjunctive** rule. The root is **LHD (low hyper-disjunction, andness
−0.08 — essentially a pure "OR", "decided by the highest")**, and each of its
three branches is itself a strong disjunction (**HHD, andness −0.80**). So the
score is high if **any** sub-profile matches — the graded-logic equivalent of
"this OR that OR the other."

## Decision-Logic Walkthrough
- **Root — LHD ("any of the following"):** OR over three sub-profiles
  (weights 0.37 / 0.36 / 0.27).
- **Sub-profile 1 — HHD ("enough to have any"):** **red underparts** OR
  **red breast** OR **grey legs** — the rose-red breast patch.
- **Sub-profile 2 — HHD:** **plain head** OR **white underparts** OR **cone bill**
  — the clean white belly and heavy conical bill.
- **Sub-profile 3 — HHD:** **red underparts / breast** OR **grey legs** OR a
  **multi-colored tail / white primary** — the bold black-white-red wing/tail.

## Ornithological Interpretation
- **Why disjunction fits:** the Rose-breasted Grosbeak is **sexually dimorphic** —
  a striking black-and-white male with a red breast triangle vs. a brown-streaked,
  white-browed female — and both sexes share a massive pale **cone bill**. A single
  conjunction would fail one sex; an **OR of profiles** correctly accepts either,
  which is exactly what the model learned (recall 96.7%).
- **Well-matched:** red breast/underparts, white belly, and the diagnostic cone
  bill are all textbook marks.
- **Note:** the OR structure means no single feature is mandatory — an image
  satisfying any one strong sub-profile scores high, giving robustness to pose,
  sex, and lighting.

**Bottom line:** a faithful, high-recall **disjunctive** rule — "any of these
red-breasted, cone-billed signatures" — a natural fit for a dimorphic species.
