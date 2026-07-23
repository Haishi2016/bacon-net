# Concept-BACON-CBM — experiment summary

Consolidated results across all comparisons in this project. Unless noted,
numbers are single-seed FashionMNIST/MNIST runs (±~1 pt); the cross-model
zero-shot table is averaged over 3 seeds.

---

## 1. CREAM reproduction — in-distribution CBM comparison (FashionMNIST)

Reproduces *Towards Reasonable Concept Bottleneck Models* (arXiv:2506.05014).
Values are **Task / Concept accuracy %**; **Leakage** Λ = task(via concepts) −
concept ceiling (`C_true→Y`). ~40 epochs. See [cream/](cream/README.md).

| model | iFMNIST (incomplete, ceiling **60**) | cFMNIST (complete, ceiling **100**) | sFMNIST *(ours, complete, ceiling **100**)* |
|-------|:---:|:---:|:---:|
| **C_true→Y** (ceiling) | 60.0 | 100.0 | 100.0 |
| **BlackBox** | 92.0 / — | 92.7 / — | 92.6 / — |
| **SoftCBM** | 92.1 / 98.0 · **Λ +32** | 92.5 / 96.6 · Λ0 | 92.6 / 97.7 · Λ0 |
| **CREAM** | 91.3 / 98.0 · Λ0 | 92.5 / 96.5 · Λ0 | 92.4 / 97.6 · Λ0 |
| **BaconCBM (ours, fixed)** | 57.9 / 97.9 · Λ0 | 92.4 / 96.6 · Λ0 | 92.3 / 97.4 · Λ0 |
| **BaconCBM+SC (ours)** | 92.0 / 98.1 · Λ0 | 92.5 / 96.4 · Λ0 | 92.2 / 97.6 · Λ0 |

- SoftCBM **leaks** on incomplete concepts (92 ≫ 60 ceiling); CREAM and our BACON
  are leak-free.
- Our pure BaconCBM sits at the concept ceiling (57.9 incomplete, ~92 complete),
  matching CREAM with fully human-readable logic trees.
- **sFMNIST** = our fix for incompleteness: four meaningful binary attributes
  (`long_sleeve, front_opening, open_toe, ankle_high`) + sophisticated NOT/OR
  trees, instead of cFMNIST's artificial season group.

## 2. Fixed ("critiqued") vs Fine-tuned trees + leakage discriminator

Same tree structure; fine-tuned = trainable andness + input weights
(`bacon.FixedGLTree`), with two aggregator backends: **gl.generic** (anchor
mixture) and **lsp.full_weight** (BACON weighted power-mean; native scalar
andness in [-1,2] + convex weights). **TRUE cpt** = the logic head run on
ground-truth concepts. See [cream/eval_finetune.py](cream/eval_finetune.py).

| setting | model | task (predicted) | task (**TRUE cpt**) | concept | verdict |
|---------|-------|:---:|:---:|:---:|---------|
| iFMNIST | fixed (boolean) | 57.6 | 60.0 | 98.1 | — |
| iFMNIST | ft · gl.generic | 90.9 | **60.0** | 95.5 | **leakage** (+31) |
| iFMNIST | ft · full_weight | 91.5 | **60.0** | 97.8 | **leakage** (+31) |
| sFMNIST | fixed (boolean) | 92.2 | 100.0 | 97.3 | — |
| sFMNIST | ft · gl.generic | 92.3 | **100.0** | 97.4 | genuine **expressiveness** |
| sFMNIST | ft · full_weight | 92.1 | **100.0** | 97.4 | genuine **expressiveness** |

- Andness ≠ leakage. Fine-tuning is real expressiveness (100% on true concepts,
  sFMNIST). It only "leaks" when concepts are incomplete — task on *predicted*
  concepts exceeds the true-concept ceiling.
- Both aggregator backends behave the same on the discriminator; full_weight
  exposes a single interpretable scalar andness per node (range up to 2, i.e. it
  can be *more* conjunctive than pure min) rather than an anchor mixture. On
  incomplete iFMNIST it makes the leakage vivid: concept-identical `T-shirt`
  (a=0.77) and `Pullover` (a=1.28) get very different andness *and* weights.

## 3. Cross-model zero-shot transfer — blob→blob fashion (3-seed average)

Trained on sFMNIST, applied to `clothing-dataset-small` shaded silhouettes.
See [cream/eval_zeroshot_models.py](cream/eval_zeroshot_models.py).

| model | FMNIST test acc | zero-shot sub-cat / 10 |
|-------|:---:|:---:|
| BlackBox | 92.45 ± 0.06 | **5.33 ± 0.47** |
| SoftCBM | 92.67 ± 0.20 | 4.33 ± 0.47 |
| CREAM | 92.36 ± 0.16 | 4.33 ± 0.47 |
| BaconCBM+SC | 92.29 ± 0.10 | 4.00 ± 0.00 |
| BaconCBM | 92.40 ± 0.27 | 3.67 ± 1.25 |

- All ~92% in-distribution but only 3.7–5.3/10 zero-shot (overlapping bands):
  in-distribution accuracy does **not** predict transfer, and task-level
  zero-shot is **not** a leakage detector.
- `t-shirt`→Tops and `longsleeve`→Tops transfer robustly across every model;
  `shorts`, `skirt`, `shoes` never transfer.

## 4. Fashion silhouette preprocessing (single BaconCBM, blob→blob)

See [cream/eval_fashion_photos.py](cream/eval_fashion_photos.py),
[cream/eval_fashion.py](cream/eval_fashion.py).

| source | representation | sub-cat match |
|--------|----------------|:---:|
| clothing-dataset-small | **shaded** grayscale interior | **6/10** |
| clothing-dataset-small | flat binary mask | 3/10 |
| QuickDraw doodles | line→blob (outline / fill) | 2/7 |

- Keeping the garment's internal shading doubled transfer (matches FashionMNIST's
  texture); line→blob barely transfers.

## 5. Zero-shot concept transfer of the DIGIT model (line→line)

MNIST-trained stroke concepts + fixed digit trees, applied unchanged. See the
top-level [README](README.md).

| test | metric | result |
|------|--------|:---:|
| MNIST (in-distribution) | test acc / concept-alignment | 99.5% / 1.00 |
| synthetic shapes | circle vs polygon/line ROC-AUC ("0" tree) | 0.91 |
| "8" tree, 2 stacked touching circles | truth score | 0.95 |
| QuickDraw everyday objects | round vs not ROC-AUC ("0" tree) | 0.67 |
| real photos (Caltech-101, edge sketch) | "0" tree / loop-only ROC-AUC | 0.55 / 0.67 |

### BaconCBM vs CREAM — zero-shot transfer (same concepts, same reasoning, task-only)

Both trained on MNIST digit labels only, same 9 stroke concepts and same per-digit
reasoning graph; BaconCBM uses fixed AND/OR/NOT logic, CREAM uses a trained masked
C→Y linear. See [eval_cream_zeroshot.py](eval_cream_zeroshot.py).

| model | MNIST acc | circle-vs-poly/line AUC | loop concepts on circle vs line |
|---|:---:|:---:|---|
| **BaconCBM** | 99.5 | **0.907** | circle loops 0.88/0.79, line loops 0.15/0.18 ✅ aligned |
| **CREAM** | 90.7 | 0.845 | `loop_lower` fires on **lines** (0.96) not circles (0.07) ❌ scrambled |

- BACON's **rigid logic forces human-aligned concepts** (an AND over `loop_lower`
  requires it to actually mean a lower loop), which transfer zero-shot. CREAM's
  trained masked linear has no such constraint, so its concepts are arbitrary
  (names are meaningless) and transfer worse — even with identical supervision.

**Harder domains — real QuickDraw doodles and Caltech-101 edge sketches.** Same
two models, applied zero-shot; we report circle-detection AUC and a *concept
alignment gap* = mean(loop concepts | round object) − mean(loop | non-round):

| domain | model | round-vs-non AUC | loop(round) | loop(non) | align-gap |
|---|---|:---:|:---:|:---:|:---:|
| QuickDraw | BaconCBM | 0.664 | 0.87 | 0.58 | **+0.30** |
| QuickDraw | CREAM | **0.789** | 0.49 | 0.48 | +0.01 |
| Caltech-101 | BaconCBM | 0.560 | 0.54 | 0.44 | **+0.10** |
| Caltech-101 | CREAM | **0.688** | 0.21 | 0.24 | −0.02 |

- **Honest reversal on task AUC:** on these noisier real-image domains CREAM's
  flexible head actually *transfers the digit-0 task better* than BaconCBM — the
  synthetic-shapes AUC advantage does **not** generalize.
- **But the concept story holds firmly:** only BaconCBM keeps a positive
  alignment gap (loop concepts genuinely fire more on round objects, +0.30 /
  +0.10); CREAM's per-concept gap collapses to ~0 (+0.01 / −0.02), i.e. its named
  concepts remain meaningless everywhere. So rigid logic buys **interpretable,
  auditable concepts** (which is the point of a CBM), not necessarily the best raw
  task transfer. CREAM wins task accuracy by exploiting whatever pixels correlate,
  at the cost of concept semantics.

## 6. CUB-200-2011 — 200 manually-generated BACON trees

200 fixed per-class trees auto-generated from concept **prototypes** (majority
vote of the 112 attributes per species); only the concept extractor trains. See
[cub/cub_trees.py](cub/cub_trees.py) (tree generation + ceiling) and
[cub/cub_compare.py](cub/cub_compare.py) (frozen ResNet-18 features).

**Discriminability ceiling (true concepts → argmax tree, no training):**

| tree design | train | test |
|---|:---:|:---:|
| signed-AND (all 112 signed literals) | **100.0** | **100.0** |
| positive-AND (only ~23 ON concepts) | 92.3 | 91.9 |
| weighted (freq-confidence, graded AND) | 98.1 | 97.9 |

**Trained comparison (frozen ResNet-18 features; ceiling = 100%, so Λ=0 for all):**

| model | task | concept |
|---|:---:|:---:|
| BlackBox | 61.3 | — |
| SoftCBM (trained linear head) | 52.6 | 86.8 |
| BaconCBM (signed-AND, fixed) | 42.0 | 88.3 |
| BaconCBM (positive-AND, fixed) | 44.6 | 87.2 |
| BaconCBM (weighted / freq, fixed) | 41.1 | 88.5 |
| BaconCBM (soft weighted-mean, fixed) | 42.3 | 88.3 |
| BaconCBM (distinct / TF-IDF, fixed) | 42.5 | 86.6 |

- The 200 generated trees are **complete** (100% ceiling with perfect concepts);
  statistical weighting (frequency / confidence / distinctiveness) keeps the
  ceiling ~98% while making trees softer.
- But with *predicted* concepts (~88%), **every fixed design lands at ~42–45%** —
  tree design (strict vs weighted vs soft vs TF-IDF) moves it only ~3 pt. The
  dominant gap is **fixed vs trained**: SoftCBM's trainable linear head (52.6)
  co-adapts to the concept encoder's specific error pattern, which no fixed
  prototype head can. `positive-AND` (ignores the many error-prone absent-feature
  literals) is the best fixed design; softening andness (weighted-mean) helps over
  the strict geometric-mean AND.
- Frozen features cap everything (BlackBox 61 vs ~75 fine-tuned); the *relative*
  ordering is the point. All models are leak-free (ceiling is 100%).
- Implication: statistics are a good *initialisation* for the trees, but closing
  the gap on noisy concepts needs the trees to **fine-tune** (`bacon.FixedGLTree`),
  co-adapting andness/weights while staying interpretable.

---

## Cross-cutting conclusions

- **Domain law:** concept transfer works **line→line** and **blob→blob**, never
  line↔blob — it's governed by input representation, not subject matter.
- **Leakage** is a concept-completeness property, detected by comparing task
  accuracy on *predicted* vs *true* concepts — not by learned andness, and not by
  zero-shot accuracy.
- Our **fixed BACON trees** match CREAM's accuracy on both complete settings while
  staying leak-free and fully human-readable; **`bacon.FixedGLTree`** adds optional
  graded-logic fine-tuning (trainable andness + weights on a frozen structure).

*Caveat:* sections 1, 2, 4, 5 are single-seed (±~1 pt); section 3 is 3-seed averaged.
