RAT-SPN Architecture
====================

Random and Tensorized Sum-Product Networks (RAT-SPN) - hyperparameter-defined probabilistic circuit architecture.

RatSPN
------

The RAT-SPN class provides a fully automated architecture for building probabilistic circuits with depth, factorization, and mixing layers.

.. autoclass:: spflow.modules.rat.RatSPN

RepetitionMixingLayer
---------------------

A specialized sum layer used as the first layer in RAT-SPN to sum over repetitions.

.. autoclass:: spflow.modules.sums.RepetitionMixingLayer
