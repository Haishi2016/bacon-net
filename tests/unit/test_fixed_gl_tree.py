"""
Unit tests for FixedGLTree / FixedGLTreeBank (trainable fixed-structure trees).

Covers:
  1. Fixed structure evaluates the intended logic at init.
  2. Training adjusts the ANDNESS of a node (AND-init tree fits an OR target).
  3. Training adjusts INPUT WEIGHTS (down-weights an irrelevant input).
  4. Structure is frozen: no routing/topology parameters, only andness + weights.
  5. FixedGLTreeBank evaluates one tree per class.
"""

import torch

from bacon.fixedGLTree import FixedGLTree, FixedGLTreeBank


NAMES = ["a", "b", "c"]


def _fit(tree, X, y, steps=400, lr=0.1):
    opt = torch.optim.Adam(tree.parameters(), lr=lr)
    lossf = torch.nn.BCELoss()
    for _ in range(steps):
        opt.zero_grad()
        out = tree(X).clamp(1e-6, 1 - 1e-6)
        loss = lossf(out, y)
        loss.backward()
        opt.step()
    return loss.item()


def test_evaluates_logic_at_init():
    tree = FixedGLTree("a AND b", NAMES)
    X = torch.tensor([[1., 1., 0.], [1., 0., 0.], [0., 0., 0.]])
    out = tree(X)
    # a AND b: high only when both a and b are 1
    assert out[0] > 0.7
    assert out[1] < 0.5
    assert out[2] < 0.5


def test_not_is_fixed_complement():
    tree = FixedGLTree("NOT a", NAMES)
    X = torch.tensor([[1., 0., 0.], [0., 0., 0.]])
    out = tree(X)
    assert torch.allclose(out, torch.tensor([0.0, 1.0]), atol=1e-5)


def test_training_adjusts_andness():
    """AND-initialised node should lower its andness to fit an OR target."""
    torch.manual_seed(0)
    tree = FixedGLTree("a AND b", NAMES, and_init=0.85)
    before = tree.describe()[0]["andness"]

    # target = OR(a, b)
    X = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])
    y = torch.tensor([0., 1., 1., 1.])
    _fit(tree, X, y)

    after = tree.describe()[0]["andness"]
    assert after < before - 0.1, f"andness did not drop: {before:.3f} -> {after:.3f}"
    # and the tree now behaves like OR
    out = tree(X)
    assert out[0] < 0.5 and out[1] > 0.5 and out[3] > 0.5


def test_training_adjusts_input_weights():
    """With target = a, the weight on the irrelevant input b should shrink."""
    torch.manual_seed(0)
    tree = FixedGLTree("a AND b", NAMES, and_init=0.75, init_weight=0.9)
    w_b_before = tree.describe()[0]["weights"][1]

    # target depends only on a; b is random noise
    torch.manual_seed(1)
    X = torch.rand(256, 3)
    X[:, 0] = (X[:, 0] > 0.5).float()          # a is the informative bit
    y = X[:, 0].clone()
    _fit(tree, X, y, steps=500)

    w = tree.describe()[0]["weights"]
    w_a, w_b_after = w[0], w[1]
    assert w_b_after < w_b_before - 0.1, f"weight on b did not shrink: {w_b_before}->{w_b_after}"
    assert w_a > w_b_after, "informative input a should keep a higher weight than b"


def test_structure_is_frozen():
    """Only andness (anchor logits) + input weights are parameters -- no routing."""
    tree = FixedGLTree("a AND (b OR c)", NAMES)
    pnames = [n for n, _ in tree.named_parameters()]
    # every parameter belongs to an aggregator (alpha_logits) or weight_logits
    assert all(("alpha_logits" in n) or ("weight_logits" in n) for n in pnames), pnames
    # two internal nodes (AND, OR) -> 2 aggregators + 2 weight vectors
    assert len(tree.aggregators) == 2
    assert len(tree.weight_logits) == 2


def test_tree_bank():
    bank = FixedGLTreeBank(NAMES, {0: "a AND b", 1: "a OR c", 2: "NOT b"})
    X = torch.tensor([[1., 1., 0.], [1., 0., 1.]])
    out = bank(X)
    assert out.shape == (2, 3)
    assert out[0, 0] > 0.7        # a AND b true for row 0
    assert out[1, 1] > 0.7        # a OR c true for row 1


# ---- lsp.full_weight aggregator backend ----

def test_full_weight_evaluates_and_init():
    tree = FixedGLTree("a AND b", NAMES, aggregator="lsp.full_weight", and_init=1.0)
    X = torch.tensor([[1., 1., 0.], [1., 0., 0.], [0., 0., 0.]])
    out = tree(X)
    assert out[0] > 0.7 and out[1] < 0.5 and out[2] < 0.5
    # scalar andness near AND at init
    assert tree.describe()[0]["andness"] > 0.8


def test_full_weight_training_adjusts_andness():
    torch.manual_seed(0)
    tree = FixedGLTree("a AND b", NAMES, aggregator="lsp.full_weight", and_init=1.0)
    before = tree.describe()[0]["andness"]
    X = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])
    y = torch.tensor([0., 1., 1., 1.])           # OR target
    _fit(tree, X, y)
    after = tree.describe()[0]["andness"]
    assert after < before - 0.3, f"andness did not drop: {before:.3f}->{after:.3f}"
    out = tree(X)
    assert out[0] < 0.5 and out[1] > 0.5 and out[3] > 0.5


def test_full_weight_training_adjusts_weights():
    torch.manual_seed(0)
    tree = FixedGLTree("a AND b", NAMES, aggregator="lsp.full_weight", and_init=0.9)
    w_b_before = tree.describe()[0]["weights"][1]
    torch.manual_seed(1)
    X = torch.rand(256, 3)
    X[:, 0] = (X[:, 0] > 0.5).float()
    y = X[:, 0].clone()
    _fit(tree, X, y, steps=500)
    w = tree.describe()[0]["weights"]           # convex (sum to 1)
    assert w[0] > w[1], f"informative input a should outweigh b: {w}"
    assert w[1] < w_b_before, "weight on noise input b should shrink"

