"""Train an emergent K CUB OCBM with the permutation-free FULL TREE head
(egress-hardened branching funnel, static continuous andness).

Isolates the full-tree head vs the left-tree (VectorTreeLogicHead). Hardened ->
faithful frozen tree. Optional decorrelation (--decorr-lam).

    py -3 -u run_cub_fulltree.py --epochs 400 --K 112 --branching 4
    -> saved/cub_ocbm_k{K}_fulltree_b{b}[_decorr{lam}]_{epochs}ep.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "table"))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

import _cub                                                     # noqa: E402
from cub_emergent import train_one                             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K", type=int, default=112)
    ap.add_argument("--branching", type=int, default=4)
    ap.add_argument("--head", choices=["auto", "fulltree", "hybrid", "recttree"],
                    default="auto",
                    help="'auto'=hybrid if --bin-frac>0 else fulltree; "
                         "'recttree'=rectangular learned-convergence DAG (constant width, "
                         "single-root/fan-out/non-compactness penalties)")
    ap.add_argument("--rect-width", type=int, default=None,
                    help="recttree hidden width W (default = K)")
    ap.add_argument("--rect-depth", type=int, default=4,
                    help="recttree number of width-W layers")
    ap.add_argument("--max-parents", type=int, default=1,
                    help="recttree fan-out budget N per node (1 = tree)")
    ap.add_argument("--root-lam", type=float, default=1.0,
                    help="recttree single-active-root (pointer entropy) penalty weight")
    ap.add_argument("--parent-lam", type=float, default=0.1,
                    help="recttree <=N-parent fan-out penalty weight")
    ap.add_argument("--compact-lam", type=float, default=0.05,
                    help="recttree non-compactness (nodes-per-layer) penalty weight")
    ap.add_argument("--binarize-lam", type=float, default=0.0,
                    help="recttree edge-entropy penalty (polarizes gates for cleaner hardening)")
    ap.add_argument("--leaf-shortcut", action="store_true",
                    help="recttree: let high-impact concepts (leaves) feed any upper layer "
                         "directly (skip/dense edges), impact-driven via the edge gates")
    ap.add_argument("--early-stop-patience", type=int, default=0,
                    help="if >0: after this many epochs without a SOFT test-acc improvement "
                         "(and past --min-freeze-frac), restore the soft-peak checkpoint and "
                         "harden FROM IT (avoids freezing an overfit late epoch)")
    ap.add_argument("--min-freeze-frac", type=float, default=0.3,
                    help="earliest fraction of epochs at which patience early-stop may fire")
    ap.add_argument("--branching-schedule", type=str, default=None,
                    help="per-layer branching factors, comma list e.g. '2,2,4,8' "
                         "(deep-then-wide); overrides --branching")
    ap.add_argument("--max-egress", type=int, default=1,
                    help="parents each node may feed after hardening (1=tree; >1=relaxed DAG)")
    ap.add_argument("--bin-frac", type=float, default=0.0,
                    help="hybrid head: fraction of inputs on the binary spine (0 = pure full tree)")
    ap.add_argument("--perm-free-spine", action="store_true",
                    help="hybrid head: full sub-tree over ALL K + egress-style spine "
                         "SELECTION (no Sinkhorn permutation -> clean freeze)")
    ap.add_argument("--negation", action="store_true",
                    help="per-concept identity/negation gate (leaves may read NOT c)")
    ap.add_argument("--sigmoid-scale", type=float, default=1.0,
                    help="steepen the concept sigmoid: sigmoid(scale * logits) (1.0 = normal)")
    ap.add_argument("--concept-lam", type=float, default=0.0,
                    help="concept-BCE supervision weight; >0 pins the K emergent concepts "
                         "to the K CUB attributes (supervised-concept OCBM)")
    ap.add_argument("--final-temp", type=float, default=0.05,
                    help="routing final annealing temperature (lower=sharper=near-one-hot)")
    ap.add_argument("--straight-through", action="store_true",
                    help="hard-argmax egress in forward (trains AS the discrete tree)")
    ap.add_argument("--coefficients", action="store_true",
                    help="add smooth per-source coefficient (relevance) layers")
    ap.add_argument("--scan", type=int, default=0,
                    help="candidate-scan freeze: sample N hard trees from the soft "
                         "routing, keep per head the most faithful (0 = greedy argmax)")
    ap.add_argument("--perm-sparsity", type=float, default=None,
                    help="egress peaking weight (default 5.0; 1.0 under --scan to keep routing soft)")
    ap.add_argument("--anneal-cap", type=float, default=None,
                    help="routing sharpening cap (default 1.0; 0.5 under --scan)")
    ap.add_argument("--freeze-frac", type=float, default=None,
                    help="fraction of epochs before freeze (default 0.85; 0.6 under --scan for more finetune)")
    ap.add_argument("--decorr-lam", type=float, default=0.0)
    ap.add_argument("--aggregator", choices=["full_weight", "aigcd"],
                    default="full_weight",
                    help="node aggregator: full_weight (single power mean) or aigcd "
                         "(array of full_weight experts + value-routing)")
    ap.add_argument("--experts", type=int, default=5,
                    help="number of full_weight experts per node when --aggregator aigcd")
    ap.add_argument("--save-dir", default=os.path.join(_HERE, "saved"))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Scan freeze wants the soft routing to stay EXPLORABLE (a real distribution
    # over trees), so under --scan we sharpen less and freeze earlier for more
    # frozen finetuning. Explicit flags still override these.
    scan = args.scan > 0
    perm_sparsity = args.perm_sparsity if args.perm_sparsity is not None else (1.0 if scan else 5.0)
    anneal_cap = args.anneal_cap if args.anneal_cap is not None else (0.5 if scan else 1.0)
    freeze_frac = args.freeze_frac if args.freeze_frac is not None else (0.6 if scan else 0.85)
    final_temp = args.final_temp if not scan or args.final_temp != 0.05 else 0.5

    # branching: per-layer schedule (list) overrides the scalar funnel.
    if args.branching_schedule:
        branching = [int(b) for b in args.branching_schedule.split(",")]
        btag = "b" + "-".join(str(b) for b in branching)
    else:
        branching = args.branching
        btag = f"b{args.branching}"
    etag = "" if args.max_egress <= 1 else f"_eg{args.max_egress}"
    ntag = "_neg" if args.negation else ""
    sigtag = "" if args.sigmoid_scale == 1.0 else f"_sig{args.sigmoid_scale:g}".replace(".", "p")
    # concept supervision: pin the first K emergent concepts to the K CUB attributes.
    concept_targets = list(range(args.K)) if args.concept_lam > 0 else None
    suptag = "" if args.concept_lam <= 0 else f"_sup{args.concept_lam:g}".replace(".", "p")
    # hybrid head when a binary-spine fraction is requested.
    head_kind = "hybrid" if args.bin_frac > 0 else "fulltree"
    if args.head != "auto":
        head_kind = args.head
    htag = f"_hyb{args.bin_frac:g}".replace(".", "p") if args.bin_frac > 0 else ""
    if args.bin_frac > 0 and args.perm_free_spine:
        htag += "pf"

    tl = DataLoader(_cub._CUBImages("train", True), batch_size=args.batch_size,
                    shuffle=True, num_workers=args.workers, pin_memory=True)
    vl = DataLoader(_cub._CUBImages("test", False), batch_size=128,
                    shuffle=False, num_workers=args.workers, pin_memory=True)
    os.makedirs(args.save_dir, exist_ok=True)

    dtag = "" if args.decorr_lam == 0 else f"_decorr{args.decorr_lam:g}".replace(".", "p")
    stag = "_st" if args.straight_through else ""
    ctag = "_coef" if args.coefficients else ""
    scantag = f"_scan{args.scan}" if scan else ""
    aggtag = "" if args.aggregator == "full_weight" else f"_aigcd{args.experts}"
    if head_kind == "recttree":
        rw = args.rect_width if args.rect_width else args.K
        rtag = f"w{rw}_d{args.rect_depth}_p{args.max_parents}"
        sctag = "_sc" if args.leaf_shortcut else ""
        path = os.path.join(args.save_dir,
                            f"cub_ocbm_k{args.K}_recttree_{rtag}{sctag}{ntag}{sigtag}{suptag}{ctag}{stag}_{args.epochs}ep.pt")
    else:
        path = os.path.join(args.save_dir,
                            f"cub_ocbm_k{args.K}_fulltree_{btag}{etag}{ntag}{sigtag}{suptag}{htag}{stag}{ctag}{aggtag}{scantag}{dtag}_{args.epochs}ep.pt")
    if os.path.exists(path):
        print(f"checkpoint exists, SKIPPING ({path})", flush=True)
        return

    print(f"\n===== k{args.K}_{head_kind} {btag} egress={args.max_egress} bin_frac={args.bin_frac} decorr={args.decorr_lam} "
          f"scan={args.scan} sparsity={perm_sparsity} anneal_cap={anneal_cap} "
          f"freeze_frac={freeze_frac} final_temp={final_temp} "
          f"({args.epochs}ep, hardened) =====", flush=True)
    acc, model = train_one(
        args.K, tl, vl, device, epochs=args.epochs, seed=args.seed,
        head=head_kind, branching=branching,
        loss_mode="hybrid", ovr_weight=0.3,
        harden=True, perm_sparsity=perm_sparsity, freeze_frac=freeze_frac,
        anneal_cap=anneal_cap,
        concept_lam=args.concept_lam, concept_targets=concept_targets, decorr_lam=args.decorr_lam,
        fulltree_final_temp=final_temp,
        fulltree_straight_through=args.straight_through,
        fulltree_coefficients=args.coefficients,
        fulltree_scan=args.scan, fulltree_max_egress=args.max_egress,
        fulltree_bin_frac=args.bin_frac, fulltree_negation=args.negation,
        concept_scale=args.sigmoid_scale,
        fulltree_aggregator=args.aggregator, fulltree_experts=args.experts,
        hybrid_permutation=not args.perm_free_spine,
        rect_width=args.rect_width, rect_depth=args.rect_depth,
        rect_max_parents=args.max_parents, rect_root_lam=args.root_lam,
        rect_parent_lam=args.parent_lam, rect_compact_lam=args.compact_lam,
        rect_binarize_lam=args.binarize_lam,
        rect_straight_through=args.straight_through,
        rect_leaf_shortcut=args.leaf_shortcut,
        early_stop_patience=args.early_stop_patience,
        min_freeze_frac=args.min_freeze_frac,
        ckpt_path=path + ".partial", ckpt_every=50)
    torch.save({"state_dict": model.state_dict(), "K": args.K,
                "head": head_kind, "branching": args.branching,
                "bin_frac": args.bin_frac, "negation": args.negation,
                "coefficients": args.coefficients, "max_egress": args.max_egress,
                "rect_width": args.rect_width, "rect_depth": args.rect_depth,
                "rect_max_parents": args.max_parents,
                "rect_leaf_shortcut": args.leaf_shortcut,
                "decorr_lam": args.decorr_lam}, path)
    print(f"  DONE k{args.K}_{head_kind}: acc {acc * 100:.2f}%  saved {path}", flush=True)


if __name__ == "__main__":
    main()
