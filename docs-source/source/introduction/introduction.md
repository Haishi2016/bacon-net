# Introduction

BACON is a structural reinforcement mechanism that constrains autonomous agents to reason through coherent, human-aligned logical aggregation patterns. By regulating how information can be combined, BACON ensures that learned decision processes remain structurally transparent, semantically stable, and diagnostically interpretable.

This structural discipline is applicable to high-stakes decision domains where interpretability and logical coherence are essential. For instance, in medical diagnosis, BACON can discover cost-efficient and transparent diagnostic pathways by explicitly modeling mandatory and compensatory clinical features. In AI code generation systems, BACON can regulate architectural impact patterns, ensuring that generated changes respect established structural constraints. In human–robot interaction scenarios, BACON can constrain how social cues, contextual factors, and normative rules are integrated, promoting humanoid decisions that follow coherent commonsense reasoning patterns rather than arbitrary latent correlations.

By enforcing a shared structural reasoning framework, BACON establishes common ground between humans and autonomous agents. Such structural transparency forms a foundation for trust, reliability, and effective human–AI collaboration.

## Overview

BACON is a neural-symbolic decision network designed to transform raw inputs into transparent, interpretable decisions. Each input represents a graded truth value—for example, “the cell boundary is irregular” may carry a 0.85 truth score on a scale from 0 to 1. BACON incrementally aggregates these truth values through a structured decision tree, where each node applies a symbolic graded logic operator such as the [graded conjunction/disjunction (GCD) aggregator](https://ieeexplore.ieee.org/document/8471076). This produces a final decision such as “there is a 0.62 likelihood of malignancy.”

Unlike conventional deep neural networks, BACON is fully interpretable by design. Every aggregation step is symbolic, mathematically defined, and aligned with human reasoning primitives (e.g., AND-like, OR-like, and compensatory logic). This allows BACON to generate step-by-step explanations that attribute why a decision was made. A sample explanation of a BACON network diagnosing breast cancer is available [here](./sample_report.md) (excerpt from [https://arxiv.org/pdf/2505.14510](https://arxiv.org/pdf/2505.14510)).

BACON is also highly versatile and can function in multiple roles:

* As a standalone interpretable model trained directly from labeled data
* As a surrogate explainer distilled from a black-box network to expose latent decision logic
* As a decision layer stacked on top of deep feature extractors (e.g. CNNs, ViTs, large encoders)

In addition, a trained BACON network can be distilled into a compact set of symbolic functions that execute using only standard arithmetic operations. This removes any dependency on AI runtimes such as PyTorch or TensorFlow and allows BACON to run ultra-fast inference on edge devices and safety-critical systems with minimal computational cost. This makes BACON suitable for deployment in real-world environments where latency, energy efficiency, and verifiability are essential.

## Getting Started

To install the published [BACON package](https://pypi.org/project/bacon-net/):
```bash
python -m pip install bacon-net==0.3.0
```

Then, check out our sample apps:

* [Hello, World!](https://github.com/Haishi2016/bacon-net/blob/main/samples/hello-world/README.md)

    Discover classic Boolean expression from synthetic data.

* [Breast Cancer Diagnosis](https://github.com/Haishi2016/bacon-net/blob/main/samples/breast-cancer/README.md)

    Breast cancer diagnosis.