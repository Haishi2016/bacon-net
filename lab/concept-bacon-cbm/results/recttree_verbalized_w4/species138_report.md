# Scarlet Tanager — Disjunctive Graded-Logic Rule

**Model:** recttree OCBM (**W4**, hardness-anneal, seed 0), CUB-200
**Species 138 — Scarlet Tanager** · test recall 90.0% · faithful rule: 26 literals
**Rule source (W4):** `species138_rule.json` (pruned to top-3 children / node, depth 3)

## Plain-Language Statement
> It calls a bird a **Scarlet Tanager** when it sees **either** a red-bodied bird
> **or** an olive-bodied bird with a red nape — one branch for the male, one for
> the female.

## Overview
A **disjunctive** rule: the root is **SD+ (high soft disjunction — "nice to have
some")** over branches that are strong disjunctions (**HHD, andness −0.80**). The
two dominant sub-profiles correspond almost exactly to the two plumages of a
**sexually dimorphic** species.

## Decision-Logic Walkthrough
- **Root — SD+ ("any of these is good"):** OR over the sub-profiles below.
- **Male profile (HHC / HHD):** **red upperparts** + **red belly** with a
  **cone bill** — the brilliant scarlet body of the breeding male.
- **Female / non-breeding profile (HHD):** **olive upperparts** OR **red nape**
  OR **plain head** — the yellow-green female and molting male.
- **Support:** **yellow crown**, **grey forehead** (weaker cues shared across
  plumages).

## Ornithological Interpretation
- **Why disjunction fits:** the male Scarlet Tanager is unmistakable scarlet with
  black wings; the female is a plain olive-yellow — visually almost a different
  bird. A conjunction ("red AND olive") is impossible; an **OR of a red profile
  and an olive profile** is exactly right, and the model discovered it (recall
  90%).
- **Well-matched:** red body (male) and olive upperparts (female) are the correct
  diagnostic plumages; the **cone bill** anchors both.
- **Note:** the olive branch overlaps with other yellow-green songbirds, which is
  the main source of residual confusion.

**Bottom line:** a clean **"male-profile OR female-profile"** disjunctive rule —
the graded-logic head capturing sexual dimorphism explicitly.
