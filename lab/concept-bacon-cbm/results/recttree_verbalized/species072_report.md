# Blue Jay — Graded-Logic Rule Verbalization

**Model:** recttree OCBM (W64, hardness-anneal, seed 2), CUB-200
**Species 72 — Blue Jay** · test recall 90.0% · faithful rule: 9 literals (very concise)
**Rule source:** `species072_rule.json` (pruned to top-3 children / node, depth 3)

## Plain-Language Statement
> It flags a **large, masked-faced perching bird** as a Blue Jay — leaning on size and a dark facial mask rather than the expected blue plumage.

## Overview
This is one of the most **concise** rules in the model (only 9 literals). A
**low hard conjunction** (HC-, "must have most") combines a few bundles built
around **large size**, a **masked head**, a **blue eye**, and — more weakly — an
**olive primary** and a **hooked bill**.

## Decision-Logic Walkthrough
- **Root — HC- ("must have most"):** requires most of a small set of cues.
- **Size / bill bundle (CP, "below the lowest"):** **large size** with a hooked/
  seabird-like bill shape.
- **Face bundle (HHC, "below the lowest"):** **masked head pattern** with a
  **blue eye** — the black facial framing of a Blue Jay.
- **Primary color:** **olive** (weak, low-weight).

## Ornithological Interpretation
- **Partially matched:** the **masked head** does capture the Blue Jay's black
  facial necklace/eye-line, and **large size** is fair (jays are big for a
  perching bird). Recall is high (90%).
- **Honest flags (poorly grounded):** the rule **misses the diagnostic blue
  plumage and crest** entirely, and leans on **olive primary** and a **hooked
  seabird bill**, which are *not* Blue Jay traits. This is a genuine case where a
  high-recall rule rests on partly **spurious / correlational** features rather
  than the field marks a birder would use — worth showing as an honest limitation,
  not hiding.

**Bottom line:** concise and high-recall, but a good example of an
**imperfectly-grounded** rule: correct on the masked face, wrong to ignore the
blue plumage.
