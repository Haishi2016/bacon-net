# BACON as a concept layer for a Concept Bottleneck Model

Prototype testing the idea: use **fixed, human-authored BACON logic trees** as
the classifier on top of a learned concept bottleneck, so the symbolic structure
*reinforces* human-aligned concept extraction — **without any concept-level
supervision**.

```
image --CNN--> concept logits --sigmoid--> concept probs   (the bottleneck)
      --10 frozen human BACON trees--> per-digit truths --logit--> class logits
```

Only the CNN (plus one global temperature scalar) is trained, and only on the
digit label. The 10 BACON trees are frozen human structure. Because the label
loss is back-propagated through the fixed symbolic logic, the concept layer is
pushed to make the human logic true for the correct digit and false for the
others — i.e. the concepts become the human-defined ones on their own.

## Concepts and rules (configurable)

Human-defined stroke/shape concepts and one BACON formula per digit live in
[`config.py`](config.py). The formulas use a tiny DSL with **AND**, **OR**,
**NOT** and parentheses, e.g.:

```
0: loop_upper AND loop_lower AND NOT horizontal_middle AND NOT vertical_line
7: horizontal_top AND (diagonal OR vertical_line) AND NOT loop_upper AND NOT loop_lower
8: loop_upper AND loop_lower AND horizontal_middle
```

AND/OR are evaluated with BACON's graded-logic power-mean aggregator
(`bacon.aggregators.lsp.full_weight.lsp_power_mean`); NOT is the graded
complement `1 - x`. AND uses andness 1.0, OR uses andness 0.0 (both
configurable). See [`bacon_logic.py`](bacon_logic.py).

Override the defaults with a JSON file:

```json
{ "concepts": ["a", "b"], "rules": { "0": "a AND NOT b", "1": "a AND b" } }
```

```
python train.py --rules my_rules.json
```

## Run

```
python train.py --epochs 6
```

Useful flags: `--bin-weight` (crispness penalty on concepts),
`--and-andness` / `--or-andness` (graded-logic sharpness), `--lr`, `--seed`.

## Result (6 epochs, default rules)

- Test accuracy ≈ **0.994**.
- **Human-alignment ≈ 1.0**: the learned per-digit concept activations match the
  human-defined literals (on the concepts each rule constrains) — e.g. digit 0
  lights up both loops and suppresses the middle bar; digit 8 lights up both
  loops *and* the middle; digit 9 lights the upper loop + vertical stroke and
  suppresses the lower loop — all discovered with **no concept labels**.

The training script prints the per-digit concept activation table and an
alignment score so you can inspect that the emergent concepts are human-aligned.

## Files

- [`config.py`](config.py) — human concept set + per-digit BACON rules.
- [`bacon_logic.py`](bacon_logic.py) — DSL parser + frozen BACON logic bank.
- [`model.py`](model.py) — CNN encoder + concept bottleneck + BACON head.
- [`train.py`](train.py) — MNIST training / evaluation / interpretability report.
- [`shapes.py`](shapes.py) — synthetic geometry-shape generator (MNIST-style).
- [`eval_shapes.py`](eval_shapes.py) — zero-shot circle detection on shapes.
- [`eval_eight.py`](eval_eight.py) — zero-shot "8"-tree probe on multi-circle scenes.
- [`quickdraw.py`](quickdraw.py) — lightweight Quick, Draw! bitmap loader (range download).
- [`eval_quickdraw.py`](eval_quickdraw.py) — "0" detector on everyday-object doodles.
- [`photo_sketch.py`](photo_sketch.py) — photo -> MNIST-style edge-sketch transform.
- [`eval_photos.py`](eval_photos.py) — "0" detector on REAL photos (Caltech-101 / CIFAR-100).

## Zero-shot circle detection (no shape training)

A circle outline is essentially a hand-drawn "0", so the learned concepts + the
frozen "0" tree should detect circles **without any retraining**. There is no
off-the-shelf geometry-shape dataset here, so [`shapes.py`](shapes.py) generates
one procedurally: white outlines on black (circle, ellipse, square, rectangle,
triangle, line, cross), anti-aliased, jittered, 28x28, MNIST-normalized.

```
python train.py --epochs 6        # writes checkpoint.pt (concepts + rules + weights)
python eval_shapes.py --preview   # applies the "0" tree to shapes, saves shapes_preview.png
```

`eval_shapes.py` uses the `"0"` BACON tree's truth value directly as a circle
score. Representative result (zero-shot, nothing trained on shapes):

| shape | "0"-tree score |
|-------|----------------|
| circle | ~0.80 |
| ellipse | ~0.60 |
| square | ~0.67 |
| triangle | ~0.41 |
| line | ~0.16 |
| cross | ~0.10 |

- **circle-only vs polygons/lines ROC-AUC ≈ 0.91**, transferred with zero shape
  training. The concept table shows circles firing both loop concepts and
  suppressing `vertical_line` — exactly what the "0" rule asks for.
- Squares are the honest hard case: a square outline *is* a closed loop, and the
  encoder only learned "loop", not "round vs. cornered", so it scores moderately.
  Lines/crosses (dominated by `vertical_line`, no loop) are rejected cleanly.

## Zero-shot "8"-tree on multi-circle scenes

The `"8"` tree is `loop_upper AND loop_lower AND horizontal_middle` — a real "8"
is two vertically-stacked *touching* circles (two loops + the pinch that forms
the middle junction). [`eval_eight.py`](eval_eight.py) renders circle scenes that
vary along three axes and reads the "8" tree's truth as an "is-this-an-8?" score
(zero-shot, nothing trained on these scenes):

```
python eval_eight.py --preview
```

Representative results (`8`-score, with `0`-score for contrast):

| axis | scene | 8-score | 0-score |
|------|-------|---------|---------|
| count | 1 circle | 0.26 | 0.73 |
| count | 2 stacked touching (an "8") | **0.95** | 0.16 |
| count | 3 stacked touching | 0.92 | 0.16 |
| position | 2 vertical touching | **0.95** | 0.16 |
| position | 2 diagonal touching | 0.82 | 0.16 |
| position | 2 horizontal (side by side) | 0.54 | 0.65 |
| contact | 2 vertical touching | **0.95** | 0.16 |
| contact | 2 vertical small gap | 0.83 | 0.23 |
| contact | 2 vertical far gap | 0.69 | 0.26 |

- **Number**: a single circle reads as a "0" (8-score 0.26 / 0-score 0.73); two or
  three stacked touching circles read as an "8" (~0.95). The middle-junction
  concept jumps from 0.11 (one circle) to 0.97 (two touching).
- **Position**: vertical > diagonal > horizontal. Side-by-side circles lose the
  middle bar (0.36) and revert toward "0".
- **Contact**: the score decreases monotonically as the two circles separate
  (0.95 → 0.83 → 0.69), i.e. the "8" evidence weakens as the pinch disappears.

All of this is symbolic behavior emerging from a fixed human rule over concepts
that were themselves learned with no concept supervision.

## Zero-shot "0" detector on everyday objects (Quick, Draw!)

Google **Quick, Draw!** is a well-known dataset of ~50M doodles of everyday
objects (345 categories). Its `numpy_bitmap` files are 28x28 grayscale, white
strokes on black — the *same format as MNIST* — so the "0" detector applies
directly. [`quickdraw.py`](quickdraw.py) range-downloads just the first few
hundred images per category (a few hundred KB each, cached under
`quickdraw_data/`); [`eval_quickdraw.py`](eval_quickdraw.py) runs the "0" tree.

```
python eval_quickdraw.py --preview
```

Representative results (`0`-score, ranked; everything zero-shot):

| category | round? | 0-score | notes |
|----------|--------|---------|-------|
| circle | ● | 0.81 | clean ring |
| clock | ● | 0.67 | clean ring |
| donut | ● | 0.61 | clean ring |
| cookie | ● | 0.59 | clean ring |
| pants | | 0.51 | two leg-loops (false positive) |
| envelope | | 0.43 | closed rectangle (false positive) |
| basketball | ● | 0.25 | round **but** internal lines |
| pizza | ● | 0.19 | round **but** slice lines |
| wheel | ● | 0.18 | round **but** spokes |
| line / ladder | | 0.13 | no loop |

- The "0" tree is really an **empty circular ring** detector — exactly what the
  digit 0 means — not a generic "round blob" detector. Clean rings (circle,
  clock, donut, cookie) score highest.
- Its errors are faithful to the rule, not random: round objects **with internal
  strokes** (wheel, pizza, basketball) are *correctly rejected* because their
  spokes/slices fire `vertical_line` and `horizontal_middle`, which the rule
  negates (`... AND NOT horizontal_middle AND NOT vertical_line`). Loopy
  non-round objects (pants, envelope) are the main false positives.
- Round-vs-non-round ROC-AUC ≈ 0.67 over this mixed set; restricting "round" to
  clean rings separates near-perfectly. The interpretable concept table
  (`loopU / loopL / vert / midBar` per category) explains every score.

## Pushing to REAL photos (Caltech-101 / CIFAR-100)

Real color photos are far out of distribution for the 28×28 stroke encoder, so
[`photo_sketch.py`](photo_sketch.py) bridges the gap with a Sobel **edge
transform** (blur → gradient magnitude → per-image percentile threshold matched
to MNIST's ~0.12 stroke density → 28×28). A photo of a round object becomes a
white circular outline on black. [`eval_photos.py`](eval_photos.py) then runs the
"0" tree; datasets are fetched from the fast fast.ai S3 mirror.

```
python eval_photos.py --dataset caltech101 --preview   # object-centric photos
python eval_photos.py --dataset cifar100   --preview   # 32x32, harder
```

Caltech-101 result (round = soccer_ball/watch/yin_yang/stop_sign/pizza/sunflower
vs non-round = laptop/scissors/chair/electric_guitar/wrench/airplanes):

| detector | what it asks | ROC-AUC |
|----------|--------------|---------|
| full "0" tree | `loopU AND loopL AND NOT vert AND NOT midBar` (empty ring) | **0.55** |
| loop-only | `loopU AND loopL` (round outline) | **0.67** |

The two numbers tell the real story:

- The learned **loop / roundness concepts genuinely transfer to real photos**
  (loop-only AUC 0.67; ranked top-3 are yin_yang, stop_sign, soccer_ball; bottom
  are airplanes, wrench, guitar) — real zero-shot transfer through the edge bridge.
- The **full "0" tree stays near chance (0.55)** because real round objects are
  *filled and textured*: their interior edges (soccer-ball seams, pizza toppings,
  yin-yang curve, clock hands) fire `vertical_line` / `horizontal_middle`, which
  the "0" rule explicitly negates. This is faithful, not a bug — the digit "0"
  means an **empty** circular loop, so a textured round object is correctly *not*
  a "0". The interpretable rule decomposition shows exactly which clause vetoes it.
- The transfer is modest (0.67, vs 0.91 on clean doodles): the photo→edge domain
  gap and internal texture cost accuracy. CIFAR-100 at 32×32 is harder still
  (near chance) because tiny cluttered images rarely yield a clean object outline.

Takeaway: the concept bottleneck learned reusable, human-aligned "loop" features
that survive an aggressive domain shift, and the symbolic rule remains
transparent about *why* it fires or not on real objects.

