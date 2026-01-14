Sum Modules
===========

Sum
---

Weighted sum over input modules with learnable log-space weights.

.. autoclass:: spflow.modules.sums.sum.Sum
   :exclude-members: weights

ElementwiseSum
--------------

Element-wise summation over multiple inputs with the same scope.

.. autoclass:: spflow.modules.sums.elementwise_sum.ElementwiseSum

RepetitionMixingLayer
---------------------

A specialized sum layer used to sum over repetitions.

.. autoclass:: spflow.modules.sums.repetition_mixing_layer.RepetitionMixingLayer

SignedSum
---------

Linear combination node that allows negative, non-normalized weights.

.. autoclass:: spflow.modules.sums.signed_sum.SignedSum
   :members:
   :show-inheritance:
