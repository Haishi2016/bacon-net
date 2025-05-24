# BACON

⚠️ To browse HTML-based documentation, clone this repository and open `docs/build/html/index.html` ⚠️

BACON is an end-to-end explainable AI framework designed to produce transparent, logically grounded decision models. Unlike black-box systems, BACON explicitly constructs interpretable aggregation trees using [Graded Logic]((https://link.springer.com/book/9783031885570)), a formalism that captures nuanced reasoning with degrees of truth. These trees reveal how individual inputs contribute to the final output, offering step-by-step insight into the decision-making process. BACON has been applied to diverse, high-stakes domains such as housing decisions, vendor evaluation, clinical diagnosis, and robotic control, where clarity, trust, and human-alignment are critical.

The goal of BACON is to uncover how AI models and autonomous agents—such as humanoid robots—make decisions, particularly in mission-critical, life-threatening contexts. Gaining this understanding is essential for the safe and responsible deployment of AI, and forms the foundation for building trust in AI systems as they become integrated into everyday life.

BACON takes as input the degrees of truth associated with various feature-based statements, and uses graded logic to systematically aggregate them into a single global truth value, which then guides the final decision. The BACON architecture is built from the ground up to support explainability. It includes a permutation layer that explores possible input orderings (since commutativity is not assumed), and an aggregation layer that merges inputs based on a logical model such as the [Logic Scoring of Preference (LSP) method](https://books.google.com/books/about/Soft_Computing_Evaluation_Logic.html?id=PgtuDwAAQBAJ). The resulting model is fully explainable, highly precise, and efficient for inference.

![bacon](./docs/images/bacon.png)

## Benefits

BACON is designed for decision-making problems where multiple factors must be logically integrated to reach a final outcome. In modern AI-integrated environments, such decisions are often the result of human-AI collaboration. BACON emphasizes alignment with human reasoning to ensure that experts can examine, review, and refine the model in detail, allowing them to infuse their expertise into the decision process and provide essential guidance and governance. Its key benefits include:

* **End-to-End Explainability**

  BACON delivers full transparency across the decision-making pipeline. It not only identifies which features influenced the outcome (feature attribution) but also how they were logically combined using graded logic. With tunable parameters, practitioners can adapt the model’s behavior to align with human judgment, ethical constraints, or policy requirements.

* **Human-AI Collaboration**

  Designed for interpretability at every stage, BACON allows human experts to inspect internal logic, validate reasoning, and inject domain expertise during training. This fosters trust, enables precise refinements, and is especially valuable in domains requiring human oversight, such as healthcare, law, and mission-critical operations.

* **Lightweight and Deployable** 
  
  BACON can be trimmed and distilled into compact, interpretable functions that require no deep learning frameworks. This makes it ideal for resource-constrained applications, including edge devices, real-time systems, drones, and robotics, where speed, cost, and reliability are paramount.

## Getting Started
To begin, try the [Hello, World sample](./samples/hello-world/README.md), which uses BACON to discover a randomly generated classic Boolean expression (e.g., A and B or C) from synthetic data.

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

* [**Iris flower classification**](./samples/iris-flowers/README.md)

  Classify Iris flower types by training one-vs-rest models to distinguish each species—Setosa, Versicolor, and Virginica—from the others based on measured petal and sepal features.

* [**House purchasing decisions**](./samples/house-purchase/README.md)

  Making house purchasing decisions based on different conditions. 

## Resources

* [BACON Documentation](./docs/build/html/index.html)