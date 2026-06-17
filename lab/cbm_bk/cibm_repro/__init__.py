"""Faithful reproduction of CIBM (Galliamov et al., ICLR 2026).

This package is a near-verbatim port of the official implementation
(github.com/dsb-ifi/cibm, ``src/``) so that we recreate the paper's *exact*
training pipeline rather than an adaptation of it:

* ``models.py``  -- ``StochasticMLP`` / ``BasicMLP`` ported from their ``models.py``.
* ``losses.py``  -- ``est_MI`` / ``est_HC`` ported from their ``losses.py``.
* ``data.py``    -- cached-embedding dataset matching their ``CUBDataset`` (the
                    ``embed_image=True`` / ``train_backbone=False`` default).
* ``precompute_embeddings.py`` -- builds the InceptionV3 (fc=Identity, 299px,
                    ImageNet-norm) 2048-d features their pipeline trains on.
* ``train.py``   -- their ``train_model`` / ``run_experiment`` recipe verbatim
                    (Adam lr 1e-3, wd 0, 20 epochs, cosine, batch 128,
                    beta 0.5, beta_lr -1e-2, MI_const 1, samples_mi 200,
                    Lagrangian dual-beta update).

Deliberately self-contained and SEPARATE from the BACON comparison harness in
``lab/cbm`` so no paper-faithful knob leaks into the apples-to-apples harness.
"""
