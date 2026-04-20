LSP Aggregators
===============

In Graded Logic, an aggregator combines a set of weighted truth values—each ranging between 0 and 1—into a single, aggregated degree of truth. The behavior of the aggregator is governed by an "andness" parameter, which controls the degree of conjunction (i.e., simultaneity) required for the aggregated result to be considered true. A higher andness enforces stronger simultaneity, meaning more inputs must be simultaneously true to yield a high output. Conversely, a lower andness produces a more disjunctive behavior, where even a single high input can significantly influence the output.

BACON uses a Generic Conjunctive/Disjunctive (GCD) aggregator, whose andness parameter is learned during training. This enables the model to adaptively determine the optimal degree of conjunction or disjunction required for the task, allowing it to better align with the underlying logic of the data. BACON offers several variations of the Generic Conjunctive/Disjunctive (GCD) aggregator, each potentially better suited to different types of decision-making scenarios. When combined with optional weight configurations—such as trainable or fixed weights—and linear transformations, BACON can be tailored to align more closely with the structure and requirements of specific tasks.

.. toctree::
   :maxdepth: 1
   
   full_weight
   half_weight
   softmax_lsp
   generic_gl
