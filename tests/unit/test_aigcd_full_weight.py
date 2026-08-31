"""
Unit tests for AIGCDFullWeightAggregator.

The aggregator is AIGCD (routing + value/context gating + partial-absorption
transform R) over an *array of full_weight power-mean experts*. It should:

  1. produce valid [0, 1] outputs for N-ary batched inputs;
  2. reduce to a plain single full_weight power mean (k=1, or routing one-hot);
  3. let routing select an expert's andness (higher andness -> more conjunctive);
  4. engage partial absorption R only near TWO ACTIVE inputs (n_eff ~ 2),
     tolerating extra inactive (~0-weight) inputs;
  5. learn value-dependent routing and recover composite (partial-absorption)
     structure from data;
  6. expose the AggregatorBase tensor/float interface and gradients.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from bacon.aggregators.lsp.aigcd_full_weight import AIGCDFullWeightAggregator
from bacon.aggregators.lsp.full_weight import lsp_power_mean


def _grid(n=21):
    xs = torch.linspace(0.0, 1.0, n)
    gx, gy = torch.meshgrid(xs, xs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=0)  # [2, n*n]


# --------------------------------------------------------------- basic contract
class TestBasics:
    def test_output_range_and_shape_batched(self):
        agg = AIGCDFullWeightAggregator(num_inputs=3, num_experts=5)
        torch.manual_seed(0)
        x = torch.rand(3, 32)
        out = agg(x)
        assert out.shape == (32,)
        assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_unbatched_and_list_inputs(self):
        agg = AIGCDFullWeightAggregator(num_inputs=2, num_experts=4, use_transform=False)
        out_vec = agg(torch.tensor([0.3, 0.7]))
        assert out_vec.dim() == 0
        out_list = agg([torch.tensor([0.3]), torch.tensor([0.7])])
        assert out_list.shape == (1,)
        assert torch.allclose(out_vec, out_list.squeeze(0), atol=1e-6)

    def test_wrong_arity_raises(self):
        agg = AIGCDFullWeightAggregator(num_inputs=2)
        with pytest.raises(ValueError):
            agg(torch.rand(3, 5))

    def test_repr_and_describe(self):
        agg = AIGCDFullWeightAggregator(num_inputs=2, num_experts=3)
        assert "AIGCDFullWeightAggregator" in repr(agg)
        info = agg.describe()
        assert info["num_experts"] == 3
        assert len(info["expert_andness"]) == 3


# ------------------------------------------------ reduction to plain full_weight
class TestPlainReduction:
    def test_k1_equals_power_mean(self):
        agg = AIGCDFullWeightAggregator(num_inputs=2, num_experts=1,
                                        use_transform=False, gate_use_values=False)
        x = torch.rand(2, 50)
        a0 = agg.expert_andness()[0]                  # scalar
        w = agg.input_weights().view(2, 1)
        expected = lsp_power_mean(x, a0, w)           # [50]
        # k=1 -> routing weight is 1
        assert torch.allclose(agg(x), expected, atol=1e-5)

    def test_routing_one_hot_selects_expert(self):
        agg = AIGCDFullWeightAggregator(num_inputs=2, num_experts=5,
                                        use_transform=False, gate_use_values=False,
                                        tau=0.02)
        j = 3
        with torch.no_grad():
            agg.routing_logits.zero_()
            agg.routing_logits[j] = 20.0
        x = torch.rand(2, 40)
        a_j = agg.expert_andness()[j]
        w = agg.input_weights().view(2, 1)
        expected = lsp_power_mean(x, a_j, w)
        assert torch.allclose(agg(x), expected, atol=2e-3)

    def test_init_routing_uniform(self):
        agg = AIGCDFullWeightAggregator(num_inputs=2, num_experts=4)
        u = torch.full((2, 1), 0.5)
        alpha = agg.routing_weights(u).squeeze(1)
        assert torch.allclose(alpha, torch.full((4,), 0.25), atol=1e-5)


# ----------------------------------------------------------- andness behaviour
class TestAndness:
    def test_high_andness_more_conjunctive(self):
        # Two experts: near hard-OR and near hard-AND. Route to each; AND <= OR.
        agg = AIGCDFullWeightAggregator(num_inputs=2, num_experts=2,
                                        andness_range=(-1.0, 2.0),
                                        use_transform=False, gate_use_values=False,
                                        tau=0.02)
        x = torch.tensor([[0.3], [0.9]])              # [2, 1]
        with torch.no_grad():
            agg.routing_logits.data = torch.tensor([20.0, -20.0])   # -> OR expert
        or_out = agg(x)
        with torch.no_grad():
            agg.routing_logits.data = torch.tensor([-20.0, 20.0])   # -> AND expert
        and_out = agg(x)
        assert and_out.item() < or_out.item()
        # conjunction pulls toward the low input, disjunction toward the high
        assert and_out.item() < 0.6 and or_out.item() > 0.6

    def test_effective_andness_tracks_routing(self):
        agg = AIGCDFullWeightAggregator(num_inputs=2, num_experts=2,
                                        andness_range=(0.0, 1.0),
                                        use_transform=False, gate_use_values=False,
                                        tau=0.02)
        with torch.no_grad():
            agg.routing_logits.data = torch.tensor([-20.0, 20.0])
        # dominant expert is the high-andness one
        assert agg.effective_andness() > 0.9


# ------------------------------------------------ partial absorption gating (2-active)
class TestPartialAbsorptionGate:
    def test_two_inputs_uniform_gate_on(self):
        agg = AIGCDFullWeightAggregator(num_inputs=2, num_experts=3, pa_sigma=0.5)
        assert float(agg.n_eff()) == pytest.approx(2.0, abs=1e-4)
        assert float(agg.partial_absorption_gate()) > 0.99

    def test_one_dominant_input_gate_off(self):
        agg = AIGCDFullWeightAggregator(num_inputs=2, num_experts=3, pa_sigma=0.5)
        with torch.no_grad():
            agg.input_weight_logits.data = torch.tensor([10.0, 0.0])  # one active
        assert float(agg.n_eff()) < 1.2
        assert float(agg.partial_absorption_gate()) < 0.2

    def test_two_active_among_inactive_gate_on(self):
        # 4 inputs, only 2 carry weight -> still "two active" -> absorption on.
        agg = AIGCDFullWeightAggregator(num_inputs=4, num_experts=3, pa_sigma=0.5)
        with torch.no_grad():
            agg.input_weight_logits.data = torch.tensor([5.0, 5.0, 0.0, 0.0])
        assert float(agg.n_eff()) == pytest.approx(2.0, abs=0.1)
        assert float(agg.partial_absorption_gate()) > 0.9

    def test_three_active_gate_off(self):
        agg = AIGCDFullWeightAggregator(num_inputs=3, num_experts=3, pa_sigma=0.5)
        # uniform over 3 -> n_eff = 3 -> not a pair
        assert float(agg.n_eff()) == pytest.approx(3.0, abs=1e-4)
        assert float(agg.partial_absorption_gate()) < 0.2

    def test_disabled_for_single_input(self):
        agg = AIGCDFullWeightAggregator(num_inputs=1, num_experts=3)
        assert agg.use_transform is False
        assert float(agg.partial_absorption_gate()) == 0.0

    def test_transform_identity_at_init(self):
        agg = AIGCDFullWeightAggregator(num_inputs=2, num_experts=3)
        R = agg.transform_matrix()
        assert torch.allclose(R, torch.eye(2), atol=1e-2)
        assert float(agg.transform_regularization()) < 1e-2


# ------------------------------------------------ partial absorption mechanism
class TestPartialAbsorptionMechanism:
    def test_transform_realizes_absorption_matrix(self):
        # R row0 -> [1, 0] (keep x), row1 -> [lam, 1-lam] (A_lambda(x, y)).
        lam = 0.3
        agg = AIGCDFullWeightAggregator(num_inputs=2, num_experts=1, pa_sigma=0.5)
        with torch.no_grad():
            agg.input_weight_logits.zero_()          # uniform -> gate ~ 1
            l1 = math.log((1 - lam) / lam)           # softmax([0, l1]) = [lam, 1-lam]
            agg.r_logits.data = torch.tensor([[20.0, 0.0], [0.0, l1]])
        R = agg.transform_matrix()
        assert torch.allclose(R, torch.tensor([[1.0, 0.0], [lam, 1 - lam]]), atol=2e-2)

    def test_absorption_off_when_not_two_active(self):
        # Same R, but a dominant input -> gate ~0 -> R_eff ~ identity.
        lam = 0.3
        agg = AIGCDFullWeightAggregator(num_inputs=2, num_experts=1, pa_sigma=0.5)
        with torch.no_grad():
            agg.input_weight_logits.data = torch.tensor([10.0, 0.0])
            l1 = math.log((1 - lam) / lam)
            agg.r_logits.data = torch.tensor([[20.0, 0.0], [0.0, l1]])
        R = agg.transform_matrix()
        assert torch.allclose(R, torch.eye(2), atol=0.15)


# ---------------------------------------------------------- value dependence
class TestValueDependence:
    def test_gate_makes_routing_input_dependent(self):
        agg = AIGCDFullWeightAggregator(num_inputs=2, num_experts=4,
                                        use_transform=False, gate_use_values=True)
        torch.manual_seed(0)
        with torch.no_grad():                        # wake the (zero-init) gate
            agg.gate_net[-1].weight.data = torch.randn_like(agg.gate_net[-1].weight)
            agg.gate_net[-1].bias.data = torch.randn_like(agg.gate_net[-1].bias)
        u_lo = torch.full((2, 1), 0.2)
        u_hi = torch.full((2, 1), 0.8)
        a_lo = agg.routing_weights(u_lo).squeeze(1)
        a_hi = agg.routing_weights(u_hi).squeeze(1)
        assert not torch.allclose(a_lo, a_hi, atol=1e-3)

    def test_context_gate_requires_context_dim(self):
        with pytest.raises(ValueError):
            AIGCDFullWeightAggregator(num_inputs=2, gate_use_context=True, context_dim=0)


# ---------------------------------------------------------- AggregatorBase API
class TestAggregatorInterface:
    def test_aggregate_tensor_matches_forward(self):
        agg = AIGCDFullWeightAggregator(num_inputs=2, num_experts=3, use_transform=False)
        vals = [torch.rand(8), torch.rand(8)]
        out_api = agg.aggregate_tensor(vals)
        out_fwd = agg(torch.stack(vals, dim=0))
        assert torch.allclose(out_api, out_fwd, atol=1e-6)

    def test_aggregate_float(self):
        agg = AIGCDFullWeightAggregator(num_inputs=3, num_experts=3)
        out = agg.aggregate_float([0.2, 0.5, 0.9])
        assert isinstance(out, float) and 0.0 <= out <= 1.0

    def test_gradients_flow(self):
        agg = AIGCDFullWeightAggregator(num_inputs=2, num_experts=4,
                                        gate_use_values=True)
        x = torch.rand(2, 16)
        loss = agg(x).mean() + agg.transform_regularization()
        loss.backward()
        assert agg.expert_andness_logits.grad is not None
        assert agg.routing_logits.grad is not None
        assert agg.r_logits.grad is not None
        assert agg.gate_net[0].weight.grad is not None


# ------------------------------------------------------------- learning tests
class TestLearning:
    def test_recover_single_power_mean(self):
        torch.manual_seed(0)
        agg = AIGCDFullWeightAggregator(num_inputs=2, num_experts=5,
                                        use_transform=False, gate_use_values=False)
        X = _grid(21)
        w = torch.tensor([0.5, 0.5]).view(2, 1)
        a_target = torch.tensor(0.9)
        target = lsp_power_mean(X, a_target, w)
        opt = torch.optim.Adam(agg.parameters(), lr=0.05)
        loss = None
        for _ in range(400):
            opt.zero_grad()
            loss = F.mse_loss(agg(X), target)
            loss.backward()
            opt.step()
        assert math.sqrt(float(loss)) < 0.02

    def test_recover_partial_absorption_structure(self):
        # Target is itself a power mean over absorbed coords -> representable.
        torch.manual_seed(0)
        lam = 0.3
        X = _grid(21)
        Rt = torch.tensor([[1.0, 0.0], [lam, 1.0 - lam]])
        u = Rt @ X
        w = torch.tensor([0.5, 0.5]).view(2, 1)
        target = lsp_power_mean(u, torch.tensor(1.2), w)

        agg = AIGCDFullWeightAggregator(num_inputs=2, num_experts=6,
                                        gate_use_values=True, identity_reg=0.0)
        opt = torch.optim.Adam(agg.parameters(), lr=0.05)
        loss = None
        for _ in range(800):
            opt.zero_grad()
            loss = F.mse_loss(agg(X), target)
            loss.backward()
            opt.step()
        assert math.sqrt(float(loss)) < 0.03
