# Indigo Bunting — Graded-Logic Rule Verbalization

**Model:** recttree OCBM (W64, hardness-anneal, seed 2), CUB-200
**Species 13 — Indigo Bunting** · test recall 86.7% · faithful rule: 13 literals, 22 ops
**Rule source:** `species013_rule.json` (pruned to top-3 children / node, depth 3)

## Plain-Language Statement
> It calls a bird an **Indigo Bunting** when it sees a **mostly-blue songbird with a dark masked face and a spotted back** — falling back on a brown-crowned, large profile for females and immatures.

## Overview
The classifier scores a bird as an Indigo Bunting with a **high soft conjunction**
(SC+, andness 0.75 — "nice to have most") over three near-equally weighted
evidence bundles (each ≈0.22). Because the root is only a *soft* conjunction, a
bird does not have to satisfy all three bundles perfectly — strong bundles can
compensate for a weaker one — but each individual bundle is a strict
product / hyper-conjunction, so within a bundle the traits must nearly all hold.

## Decision-Logic Walkthrough
- **Root — SC+ (high soft conjunction, a "nice to have most"):** blends the three
  branches below. Soft andness means the overall match degrades gracefully rather
  than collapsing if one branch is imperfect.
- **Branch 1 — CP (product t-norm, "below the lowest"):** a strict product of
  **spotted back** ∧ **masked head** ∧ a blue block
  (**blue upper-tail** 0.52 > **blue underparts** 0.31 > **blue wing** 0.17).
  This is the core "overall-blue songbird with a dark masked face" signature and
  is the single most diagnostic branch.
- **Branch 2 — CP (product t-norm):** re-uses **spotted back** together with a
  **white nape**, a corroborating plumage/structure check. (The *spotted back*
  leaf is the **same shared node** as in Branch 1 — the DAG reuses it, it is not a
  duplicate.)
- **Branch 3 — C (pure conjunction, "decided by the lowest"):** **brown crown** /
  **large size** / **white nape** — a compensatory profile the soft root can fall
  back on.

## Ornithological Interpretation
The dominant field marks the rule keys on — an overall **blue** bird (tail,
underparts, wing) with a **masked** face — are exactly the diagnostic marks of a
breeding male Indigo Bunting, so the rule is well grounded in real avian biology.

- **Well-matched:** the blue-plumage block + masked head is textbook male Indigo
  Bunting.
- **Sensible compensation:** the **brown crown** branch plausibly captures
  females / non-breeding / immature birds (which are brown rather than blue) — a
  reasonable alternative route for a soft conjunction to include.
- **Shared substructure:** **spotted back** appears in multiple branches but is a
  single reused node (fan-out ≤ 2), i.e. the model treats it as a shared
  sub-condition rather than re-learning it.
- **Flag:** **white nape** is comparatively weak and less diagnostic for this
  species; it may be incidental co-occurrence rather than a true field mark, and
  is worth checking if you audit the rule.

**Bottom line:** a faithful, concise (~13-literal) rule whose highest-weight
evidence aligns with the recognized diagnostic features of the Indigo Bunting.
