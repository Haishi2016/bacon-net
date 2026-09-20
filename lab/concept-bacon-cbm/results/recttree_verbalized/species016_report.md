# Cardinal — Graded-Logic Rule Verbalization

**Model:** recttree OCBM (W64, hardness-anneal, seed 2), CUB-200
**Species 16 — Cardinal** · test recall 96.3% · faithful rule: 17 literals, ~24 ops
**Rule source:** `species016_rule.json` (pruned to top-3 children / node, depth 3)

## Plain-Language Statement
> It calls a bird a **Cardinal** when it sees an **overall bright-red songbird with a black throat/face and a stout red bill**.

## Overview
The classifier uses a **high soft conjunction** (SC+, andness 0.70 — "nice to have
most") over several strict red-dominated bundles. The single most repeated,
highest-weight signal across branches is **red plumage everywhere** — nape,
forehead, underparts, under-tail — reinforced by a **red bill** and a **black
throat**.

## Decision-Logic Walkthrough
- **Root — SC+ ("nice to have most"):** blends ~5 evidence bundles of similar
  weight; a strong red signature in most bundles suffices.
- **Red-color bundles (LHC / CP, "below the lowest"):** repeatedly require
  **red nape + red forehead + red underparts** together, plus **red under-tail**
  and **red bill** — the crimson-all-over male Cardinal signature.
- **Face bundle (C, pure conjunction):** **black throat** with a hooked/short bill
  — the Cardinal's black facial mask around the bill.

## Ornithological Interpretation
- **Well-matched:** a male Northern Cardinal is unmistakable — brilliant red body,
  a black mask/throat around a heavy red-orange conical bill. The rule's dominant
  red-everywhere + black-throat + red-bill signals are textbook diagnostic marks.
- **Flags (spurious / weak):** a few low-weight leaves are off — **bill shape:
  hooked seabird**, **upper-tail: blue**, and **forehead: green** are not Cardinal
  traits and are likely incidental correlations the pruned view surfaces; they
  carry little weight and don't drive the decision.

**Bottom line:** a faithful, high-recall (96.3%) rule anchored on the correct
diagnostic feature — pervasive red plumage with a black face and red bill.
