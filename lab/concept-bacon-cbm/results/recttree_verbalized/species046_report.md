# American Goldfinch — Graded-Logic Rule Verbalization

**Model:** recttree OCBM (W64, hardness-anneal, seed 2), CUB-200
**Species 46 — American Goldfinch** · test recall 86.7% · faithful rule: 22 literals
**Rule source:** `species046_rule.json` (pruned to top-3 children / node, depth 3)

## Plain-Language Statement
> It calls a bird an **American Goldfinch** when it sees a **bright-yellow finch with a black cap/forehead and black-and-white wings**, topped by an orange bill.

## Overview
A **low hard conjunction** (HC-, "must have most") over several yellow-dominated
bundles. The strongest, most-repeated signals are **yellow body** (underparts,
belly, breast, primary, nape), a **black forehead/crown**, an **orange bill**, and
**white/black wings** — exactly the male American Goldfinch palette.

## Decision-Logic Walkthrough
- **Root — HC- ("must have most"):** requires most of the yellow-plumage bundles.
- **Bill / cap bundle (CP, "below the lowest"):** **orange bill** with a
  **black forehead** — the goldfinch's black cap over a pink-orange bill.
- **Yellow-body bundles (CP):** **yellow nape / underparts / primary / belly /
  breast** appear repeatedly — the canary-yellow signature.
- **Wing bundle (CP):** **white wing** and **multi-colored / black wing** — the
  bold black-and-white wing bars.

## Ornithological Interpretation
- **Well-matched:** a breeding male American Goldfinch is vivid yellow with a
  crisp **black cap**, **black wings with white bars**, and a **pale orange
  conical bill**. The rule's dominant yellow-body + black-cap + black/white-wing
  structure is textbook diagnostic.
- **Flags:** a couple of low-weight leaves (**crown: iridescent**, **wing:
  orange**) are slightly off and likely incidental; they don't drive the score.

**Bottom line:** a faithful, well-grounded rule keyed on the correct field marks —
yellow body, black cap, black-and-white wings.
