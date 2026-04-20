# BACON

**Making Agents And Machines Reason Like Humans**

In the 1970s, Dr. Jozo Dujmovic envisioned a real-valued logic system designed to reflect how people actually make decisions. He called it Graded Logic. Rather than relying on rigid true-or-false rules, it captures nuance, trade-offs, and degrees of preference during human decision processes. Building on this idea, he developed the Logic Scoring of Preference (LSP), a structured approach that enables professionals to reason through complex decisions with clarity and consistency. Over the past fifty years, these ideas have been used to support high-stakes decision making in areas such as vendor selection, medical diagnosis, and the evaluation of complex systems.

Fast forwarding to 2020s, autonomous agents and intelligent systems are increasingly making decisions on behalf of humans, often in mission-critical and safety-sensitive scenarios. Yet many of these systems operate as opaque black boxes, making it difficult to understand how a decision was made, to diagnose failures, or to ensure alignment with human values, policies, and regulatory requirements. As these systems become more capable and more widely deployed, the need for transparency, control, and trust becomes essential.

BACON brings the principles of Graded Logic and LSP into modern AI systems. It enables agents to reason in ways that reflect human thinking, to produce decisions that can be understood and explained, and to operate under meaningful human guidance. By making decision processes transparent, interpretable, and controllable, BACON helps ensure that intelligent systems are not only powerful, but also trustworthy and aligned with the people who depend on them.


## Getting Started

Install BACON as a Python package:

```bash
pip install bacon-net
```

Then use it in your code:

```python
import torch
from bacon.baconNet import baconNet
from bacon.visualization import print_tree_structure
from bacon.utils import generate_classic_boolean_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x, y, expr_info = generate_classic_boolean_data(3, device=device)

model = baconNet(3, aggregator='bool.min_max', tree_layout='left')
model.find_best_model(x, y, x, y, max_epochs=900, save_model=False)
print_tree_structure(model.assembler, expr_info['var_names'], classic_boolean=True, layout='left')
```

For full documentation, visit the [BACON docs](https://haishi2016.github.io/bacon-net/) or clone this repository and open `docs/index.html`.

## Samples

Explore the [samples/](./samples) directory for end-to-end examples on real datasets:

* [**Hello, World!**](./samples/hello-world/README.md) — Discover a Boolean expression from synthetic data
* [**Breast Cancer**](./samples/breast-cancer/README.md) — Diagnosis prediction on the UCI Breast Cancer dataset
* [**Heart Disease**](./samples/heart-disease/README.md) — Prediction using the UCI Heart Disease dataset
