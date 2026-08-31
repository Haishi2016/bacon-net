r"""AIGCD (over full_weight experts) on MNIST -- task accuracy + which non-plain
behaviours (value-based routing / partial absorption) actually get activated.

A small OCBM: ConceptCNN -> K sigmoid concepts -> ONE AIGCDFullWeightAggregator
per digit (each aggregates the K concepts into a class truth) -> logits. Trains
on MNIST labels only (no concept supervision).

Two variants for comparison:
  * plain  -- aggregators are plain full_weight (routing to experts, but value
              gate OFF and partial-absorption transform OFF).
  * aigcd  -- full capability: value-based routing gate + partial-absorption R
              (both start inert; the aggregator LEARNS whether to use them).

Reports per-epoch test accuracy (to observe the training curve) and, for the
aigcd model, per-class activation diagnostics: how input-dependent the routing
became (value-based signal) and whether R engaged (partial absorption).

    py -3 -u mnist_aigcd.py --epochs 15 --K 16 --variant both
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_CBM = os.path.join(_REPO_ROOT, "lab", "concept-bacon-cbm")
sys.path.insert(0, _CBM)
sys.path.insert(0, _REPO_ROOT)

from model import ConceptCNN                                     # noqa: E402
from train import make_loaders                                   # noqa: E402
from bacon.aggregators.lsp.aigcd_full_weight import AIGCDFullWeightAggregator  # noqa: E402

_DATA = os.path.join(_REPO_ROOT, "benchmarks", "mnist-addition", "data")


def _make_agg(num_inputs, variant, num_experts, tau=1.0):
    gate_use_values = variant == "aigcd"
    use_transform = variant == "aigcd"
    return AIGCDFullWeightAggregator(
        num_inputs=num_inputs, num_experts=num_experts,
        gate_use_values=gate_use_values, use_transform=use_transform,
        tau=tau, identity_reg=1e-3)


class AIGCDHead(nn.Module):
    """FLAT head: one AIGCD aggregator per class over ALL K concepts."""

    def __init__(self, K, n_classes=10, variant="aigcd", num_experts=5, tau=1.0):
        super().__init__()
        self.aggs = nn.ModuleList([
            _make_agg(K, variant, num_experts, tau) for _ in range(n_classes)
        ])

    def forward(self, concepts):                 # concepts [B, K]
        x = concepts.transpose(0, 1)             # [K, B]
        outs = [agg(x) for agg in self.aggs]     # each [B]
        return torch.stack(outs, dim=1)          # [B, n_classes]

    def transform_reg(self):
        return sum(a.transform_regularization() for a in self.aggs)

    def all_nodes(self):
        return list(self.aggs)


def _binary_plan(K):
    """Balanced binary reduction plan over K leaves.

    Returns (layers, n_nodes). Each layer is a list of (a, b) referencing the
    current working list: b is None -> carry the single node up unchanged.
    """
    layers, idxs = [], list(range(K))
    while len(idxs) > 1:
        layer, i = [], 0
        while i < len(idxs):
            if i + 1 < len(idxs):
                layer.append((idxs[i], idxs[i + 1]))   # 2-input node
                i += 2
            else:
                layer.append((idxs[i], None))          # carry
                i += 1
        layers.append(layer)
        idxs = list(range(len(layer)))
    n_nodes = sum(1 for layer in layers for (a, b) in layer if b is not None)
    return layers, n_nodes


class BinaryTreeAIGCDHead(nn.Module):
    """TREE head: a balanced binary tree of 2-input AIGCD nodes per class.

    Every internal node aggregates exactly two children, so nodes sit in the
    ``n_eff ~ 2`` regime where the partial-absorption transform R can engage.
    """

    def __init__(self, K, n_classes=10, variant="aigcd", num_experts=5, tau=1.0):
        super().__init__()
        self.K = K
        self.layers, self.n_nodes = _binary_plan(K)
        self.trees = nn.ModuleList([
            nn.ModuleList([_make_agg(2, variant, num_experts, tau)
                           for _ in range(self.n_nodes)])
            for _ in range(n_classes)
        ])

    def _eval_tree(self, nodes, leaves):          # leaves [K, B]
        cur = [leaves[i] for i in range(self.K)]  # list of [B]
        ptr = 0
        for layer in self.layers:
            nxt = []
            for (a, b) in layer:
                if b is None:
                    nxt.append(cur[a])            # carry
                else:
                    x2 = torch.stack([cur[a], cur[b]], dim=0)   # [2, B]
                    nxt.append(nodes[ptr](x2))
                    ptr += 1
            cur = nxt
        return cur[0]                             # [B]

    def forward(self, concepts):                 # concepts [B, K]
        x = concepts.transpose(0, 1)             # [K, B]
        outs = [self._eval_tree(nodes, x) for nodes in self.trees]
        return torch.stack(outs, dim=1)          # [B, n_classes]

    def transform_reg(self):
        return sum(n.transform_regularization()
                   for nodes in self.trees for n in nodes)

    def all_nodes(self):
        return [n for nodes in self.trees for n in nodes]



class OCBM(nn.Module):
    def __init__(self, K, n_classes=10, variant="aigcd", num_experts=5,
                 head="flat", logit_temperature=4.0):
        super().__init__()
        self.encoder = ConceptCNN(K)
        HeadCls = BinaryTreeAIGCDHead if head == "tree" else AIGCDHead
        self.head = HeadCls(K, n_classes, variant=variant, num_experts=num_experts)
        self.log_temp = nn.Parameter(
            torch.tensor(float(torch.log(torch.tensor(logit_temperature)))))

    def concept_probs(self, x):
        return torch.sigmoid(self.encoder(x))

    def forward(self, x):
        probs = self.concept_probs(x)
        truths = self.head(probs).clamp(1e-6, 1 - 1e-6)          # [B, C]
        logits = self.log_temp.exp() * (torch.log(truths) - torch.log1p(-truths))
        return logits, probs, truths


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x)[0].argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def train(model, tr, te, device, epochs, lr, tag):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    print(f"\n===== training [{tag}] {epochs} epochs =====", flush=True)
    best = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            logits, _, _ = model(x)
            loss = F.cross_entropy(logits, y) + model.head.transform_reg()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * y.size(0)
        sched.step()
        acc = evaluate(model, te, device)
        best = max(best, acc)
        print(f"  [{tag}] epoch {ep:2d}/{epochs} | loss {tot/len(tr.dataset):.4f} "
              f"| test {acc*100:.2f}%", flush=True)
    print(f"  [{tag}] BEST {best*100:.2f}%", flush=True)
    return best


@torch.no_grad()
def activation_report(model, te, device, n_classes=10):
    """Per-class: value-based routing signal + partial-absorption engagement."""
    model.eval()
    # one test batch of concept activations
    x, _ = next(iter(te))
    concepts = model.concept_probs(x.to(device))        # [B, K]
    xb = concepts.transpose(0, 1)                        # [K, B]
    print("\n===== AIGCD activation (per class) =====", flush=True)
    print(f"{'cls':>3} {'eff_and':>8} {'route_std':>9} {'route_H':>8} "
          f"{'pa_gate':>8} {'|R-I|':>7} {'n_eff':>6}  behaviour", flush=True)
    vd_active = pa_active = 0
    for c in range(n_classes):
        agg = model.head.aggs[c]
        u = agg.transform_matrix() @ xb if agg.use_transform else xb
        alpha = agg.routing_weights(u)                  # [k, B]
        a = agg.expert_andness()                        # [k]
        eff = (alpha * a.unsqueeze(1)).sum(0)           # [B] per-sample andness
        # value-based signal: how much routing varies across inputs
        route_std = float(alpha.std(dim=1).mean())
        route_H = float(-(alpha.mean(1) * (alpha.mean(1) + 1e-9).log()).sum())
        # partial absorption engagement
        if agg.use_transform:
            R = F.softmax(agg.r_logits, dim=1)
            RmI = float((R - torch.eye(R.size(0), device=R.device)).abs().mean())
            pa = float(agg.partial_absorption_gate())
        else:
            RmI, pa = 0.0, 0.0
        ne = float(agg.n_eff())
        vd = route_std > 0.02
        pae = agg.use_transform and pa > 0.3 and RmI > 0.02
        vd_active += int(vd)
        pa_active += int(pae)
        beh = []
        if vd:
            beh.append("VALUE-BASED")
        if pae:
            beh.append("PARTIAL-ABSORP")
        if not beh:
            beh.append("plain")
        print(f"{c:>3} {eff.mean():>8.2f} {route_std:>9.3f} {route_H:>8.3f} "
              f"{pa:>8.2f} {RmI:>7.3f} {ne:>6.2f}  {'+'.join(beh)}", flush=True)
    print(f"\n  classes with value-based routing active: {vd_active}/{n_classes}", flush=True)
    print(f"  classes with partial absorption active : {pa_active}/{n_classes}", flush=True)


@torch.no_grad()
def activation_report_tree(model, te, device):
    """Aggregate over ALL 2-input tree nodes: how many engage value-based routing
    and partial absorption, and the distribution of node effective-active-inputs."""
    model.eval()
    nodes = model.head.all_nodes()
    # feed a batch through the leaves so each node sees realistic inputs is hard
    # (node inputs are intermediate); instead evaluate structural activation +
    # a routing-variability probe on random 2-input batches drawn from [0,1].
    probe = torch.rand(2, 512, device=device)
    vd = pa = 0
    n_eff_two = 0
    pa_gates, rmis, route_stds = [], [], []
    for agg in nodes:
        u = agg.transform_matrix() @ probe if agg.use_transform else probe
        alpha = agg.routing_weights(u)                  # [k, B]
        route_std = float(alpha.std(dim=1).mean())
        route_stds.append(route_std)
        if agg.use_transform:
            R = F.softmax(agg.r_logits, dim=1)
            rmi = float((R - torch.eye(2, device=R.device)).abs().mean())
            g = float(agg.partial_absorption_gate())
        else:
            rmi, g = 0.0, 0.0
        rmis.append(rmi)
        pa_gates.append(g)
        if float(agg.n_eff()) > 1.6:                    # both children active
            n_eff_two += 1
        if route_std > 0.02:
            vd += 1
        if agg.use_transform and g > 0.3 and rmi > 0.02:
            pa += 1
    n = len(nodes)
    import statistics as st
    print("\n===== AIGCD tree-node activation (all 2-input nodes) =====", flush=True)
    print(f"  total 2-input nodes            : {n}", flush=True)
    print(f"  nodes with both children active: {n_eff_two}/{n}  (n_eff > 1.6)", flush=True)
    print(f"  value-based routing active     : {vd}/{n}  (route_std > 0.02)", flush=True)
    print(f"  PARTIAL ABSORPTION active      : {pa}/{n}  (pa_gate>0.3 & |R-I|>0.02)", flush=True)
    print(f"  mean pa_gate {st.mean(pa_gates):.3f} | mean |R-I| {st.mean(rmis):.3f} "
          f"| mean route_std {st.mean(route_stds):.3f}", flush=True)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--num-experts", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--variant", choices=["plain", "aigcd", "both"], default="both")
    ap.add_argument("--head", choices=["flat", "tree"], default="flat",
                    help="flat = one N-ary aggregator per class; tree = balanced "
                         "binary tree of 2-input nodes (partial absorption regime)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, te = make_loaders(_DATA, args.batch_size)
    print(f"device={device}  K={args.K}  experts={args.num_experts}  "
          f"head={args.head}  epochs={args.epochs}", flush=True)

    results = {}
    if args.variant in ("plain", "both"):
        torch.manual_seed(args.seed)
        m = OCBM(args.K, variant="plain", num_experts=args.num_experts, head=args.head)
        results["plain"] = train(m, tr, te, device, args.epochs, args.lr, "plain")
    if args.variant in ("aigcd", "both"):
        torch.manual_seed(args.seed)
        m = OCBM(args.K, variant="aigcd", num_experts=args.num_experts, head=args.head)
        results["aigcd"] = train(m, tr, te, device, args.epochs, args.lr, "aigcd")
        if args.head == "tree":
            activation_report_tree(m, te, device)
        else:
            activation_report(m, te, device)

    if len(results) == 2:
        print(f"\n===== SUMMARY =====", flush=True)
        print(f"  plain full_weight : {results['plain']*100:.2f}%", flush=True)
        print(f"  aigcd (full)      : {results['aigcd']*100:.2f}%", flush=True)
        print(f"  delta             : {(results['aigcd']-results['plain'])*100:+.2f} pt",
              flush=True)


if __name__ == "__main__":
    main()
