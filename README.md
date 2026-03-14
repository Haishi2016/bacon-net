# BACON

⚠️ Visit HTML-based documentation [here](https://haishi2016.github.io/bacon-net/), or clone this repository and open `docs/index.html` ⚠️

BACON is a structural reinforcement mechanism that constrains autonomous agents to reason through coherent, human-aligned logical aggregation patterns. By regulating how information can be combined, BACON ensures that learned decision processes remain structurally transparent, semantically stable, and diagnostically interpretable.

This structural discipline is applicable to high-stakes decision domains where interpretability and logical coherence are essential. For instance, in medical diagnosis, BACON can discover cost-efficient and transparent diagnostic pathways by explicitly modeling mandatory and compensatory clinical features. In AI code generation systems, BACON can regulate architectural impact patterns, ensuring that generated changes respect established structural constraints. In human–robot interaction scenarios, BACON can constrain how social cues, contextual factors, and normative rules are integrated, promoting humanoid decisions that follow coherent commonsense reasoning patterns rather than arbitrary latent correlations.

By enforcing a shared structural reasoning framework, BACON establishes common ground between humans and autonomous agents. Such structural transparency forms a foundation for trust, reliability, and effective human–AI collaboration.

## Demonstrated Results

- Identified a minimal sufficient subset of 8 out of 30 clinical features while preserving diagnostic accuracy in breast cancer prediction (see Paper I).
- Reduced projected gallstone screening costs by over 62% through structurally optimized diagnostic pathways (see Paper II).
- Discovered stable and interpretable aggregation patterns across 10 heterogeneous disease datasets (see Paper III).

## How BACON Works


-------------------------
TTT
-------------------------

## Getting Started
To begin, try the [Hello, World sample](./samples/hello-world/README.md), which uses BACON to approximate a randomly generated classic Boolean expression (e.g., A and B or C) from synthetic data.

To use BACON in your own program, note that it is implemented as a Python module built on top of PyTorch. Before the package is officially published, you can clone this repository and reference the module via a local path:

```python
import sys
sys.path.append('<path to the bacon folder under this repo>')

from bacon.baconNet import baconNet
from bacon.visualization import print_tree_structure
from bacon.utils import generate_classic_boolean_data
```

## Samples

* [**Hello, World!**](./samples/hello-world/README.md)

  Discover classic Boolean expression from synthetic data.

## Testing

This repository now uses a structured `pytest` layout for maintainable growth:

- `tests/unit/` for fast, isolated tests
- `tests/integration/` for multi-component or end-to-end tests

Install test dependencies:

```bash
pip install -e .[test]
```

Run unit tests only:

```bash
pytest -m "unit"
```

Run integration tests only:

```bash
pytest -m "integration"
```

Run all structured tests:

```bash
pytest
```

GitHub Actions runs both unit and integration suites on pull requests to `main`.