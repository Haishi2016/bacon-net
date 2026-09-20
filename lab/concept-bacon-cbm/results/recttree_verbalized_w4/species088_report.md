# Hooded Merganser — Disjunctive Graded-Logic Rule

**Model:** recttree OCBM (**W4**, hardness-anneal, seed 0), CUB-200
**Species 88 — Hooded Merganser** · test recall 80.0% · faithful rule: 35 literals
**Rule source (W4):** `species088_rule.json` (pruned to top-3 children / node, depth 3)

## Plain-Language Statement
> It calls a bird a **Hooded Merganser** when it sees **either** a brown-breasted
> diving duck **or** a green-tinged one with a long, thin bill — one branch per
> sex.

## Overview
A **disjunctive** rule whose branches are strong disjunctions (**HHD, andness
−0.80**). The two main sub-profiles line up with the dramatically different male
and female plumages of this **sexually dimorphic** duck, tied together by the
diagnostic **long, thin (longer-than-head) bill**.

## Decision-Logic Walkthrough
- **Root (disjunction — "any of these"):** OR over the sub-profiles below.
- **Female / eclipse profile (HHD):** **grey under-tail** OR **black primary** OR
  **brown breast** — the drab brown-and-grey female.
- **Male profile (HHD):** **blue/iridescent under-tail** OR **bill longer than
  head** OR **green underparts** — the glossy, long-billed male.
- The **long thin bill** recurs across branches — the shared merganser trait that
  separates it from other ducks.

## Ornithological Interpretation
- **Why disjunction fits:** the male Hooded Merganser has a spectacular black-and-
  white fan crest and dark iridescent body; the female is warm brown with a wispy
  cinnamon crest — very different birds. An **OR of a brown profile and an
  iridescent long-billed profile** correctly accepts either sex.
- **Honest flags:** recall is the lowest of this set (80%) — mergansers are
  water-birds photographed at distance/odd angles, and the shared "long bill" cue
  overlaps with other mergansers, so this rule is genuinely harder and less
  crisp. A fair, non-cherry-picked disjunctive example.

**Bottom line:** a **dimorphic "male OR female" disjunctive** rule anchored on the
long thin bill — faithful, but honestly the hardest (80% recall) of the set.
