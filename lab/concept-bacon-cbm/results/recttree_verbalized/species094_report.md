# Baltimore Oriole — Graded-Logic Rule Verbalization

**Model:** recttree OCBM (W64, hardness-anneal, seed 2), CUB-200
**Species 94 — Baltimore Oriole** · test recall 83.3% · faithful rule: 14 literals
**Rule source:** `species094_rule.json` (pruned to top-3 children / node, depth 3)

## Plain-Language Statement
> It calls a bird a **Baltimore Oriole** when it sees an **orange-bellied songbird with a yellow throat and a bold white wing bar**.

## Overview
A soft/hard-conjunction blend over bundles that combine **orange belly**,
**yellow throat and primary**, and a **white wing** (wing bar), with a **white
nape** and **brown eye** as supporting cues — the flame-orange-and-black oriole
pattern.

## Decision-Logic Walkthrough
- **Root ("nice to have most"):** blends the color bundles below.
- **Color bundle (CP, "below the lowest"):** **orange belly** with **yellow
  throat** and **yellow primary** — the bright underparts of an oriole.
- **Wing bundle (CP):** **white wing** — the conspicuous white wing bar on a dark
  wing.
- **Head/support bundle:** **white nape**, **brown eye** (weaker, supporting).

## Ornithological Interpretation
- **Well-matched:** a male Baltimore Oriole shows **brilliant orange underparts**,
  a **black hood**, and a **white wing bar** on black wings. The rule's
  orange-belly + yellow-throat + white-wing structure captures the correct
  diagnostic marks.
- **Flags:** **forehead: green** is spurious (orioles have a black hood, not
  green) and low-weight; the rule under-emphasizes the black hood, which a birder
  would weight more heavily. Recall (83.3%) is the lowest of this set, consistent
  with a slightly less sharply-grounded rule.

**Bottom line:** a faithful rule anchored on the right colors (orange underparts +
white wing bar), with a minor spurious green-forehead leaf and an under-weighted
black hood.
