# Painted Bunting — Graded-Logic Rule Verbalization

**Model:** recttree OCBM (W64, hardness-anneal, seed 2), CUB-200
**Species 15 — Painted Bunting** · test recall 92.9% · faithful rule: 12 literals
**Rule source:** `species015_rule.json` (pruned to top-3 children / node, depth 3)

## Plain-Language Statement
> It calls a bird a **Painted Bunting** when it sees a **multi-colored songbird with a red breast, a green forehead, and a striped belly**.

## Overview
A **high soft conjunction** (SC+, andness 0.68 — "nice to have most") over strict
bundles that combine a **green forehead**, a **red breast**, a **multi-colored
wing**, a **striped belly**, and a **white nape** — the rainbow palette of a male
Painted Bunting.

## Decision-Logic Walkthrough
- **Root — SC+ ("nice to have most"):** blends three color bundles of weight ≈0.22.
- **Head bundle (CP, "below the lowest"):** **blue eye** with a **white nape** —
  the blue head of the male.
- **Body bundle (HC / HC+):** **red breast**, **green forehead**, and a
  **multi-colored wing** together — the red-underparts + green-back signature.
- **Pattern bundle:** **striped belly** with the green forehead — supporting
  plumage structure.

## Ornithological Interpretation
- **Well-matched:** a male Painted Bunting is famously multi-colored — **blue
  head**, **green back/wings**, **red underparts**. The rule's red-breast +
  green-forehead + multi-colored-wing core is squarely on the diagnostic marks.
- **Flags:** the model spreads **eye color** across blue/brown/grey leaves (some
  low-weight and likely incidental), and **striped belly** is a weaker,
  less-diagnostic cue; neither dominates the decision.

**Bottom line:** a faithful, high-recall (92.9%) rule that keys on the correct
rainbow field marks — red breast, green forehead, multi-colored wing.
