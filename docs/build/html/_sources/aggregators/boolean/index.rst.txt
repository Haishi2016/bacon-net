Boolean Aggregators
===================

To emulate Boolean operators such as AND and OR, BACON uses a bool.min_max aggregator defined as:

.. math::

   \alpha \cdot \min(x_1, x_2) + (1 - \alpha) \cdot \max(x_1, x_2)

Here, the andness parameter :math:`\alpha` is normalized to the range :math:`[0,1]` using a sharpened sigmoid function. This encourages :math:`\alpha` to converge toward binary extremes (0 or 1), more closely mimicking the behavior of true Boolean operations.
Note that when the bool.min_max aggregator is in use, the model’s normalize_andness option should be disabled, since :math:`\alpha` is already normalized by the aggregator itself.

.. toctree::
   :maxdepth: 1
   
   min_max
