Autoencoding Probabilistic Circuits (APC)
=========================================

APC combines a tractable probabilistic-circuit encoder over joint variables :math:`(X, Z)` with a pluggable decoder and trains with a hybrid objective:

.. math::

    \mathcal{L} = \lambda_{rec}\mathcal{L}_{rec} + \lambda_{kld}\mathcal{L}_{kld} + \lambda_{nll}\mathcal{L}_{nll}

Reference
---------

APC is described in:

- `Autoencoding Probabilistic Circuits (ICLR 2025) <https://arxiv.org/abs/2502.05554>`_

Status Note
-----------

APC inference APIs remain available (encode/decode/sampling/likelihood). Latent-stat extraction and KL-style
training helpers are currently unsupported.

Main Components
---------------

Configuration
~~~~~~~~~~~~~

.. autoclass:: spflow.zoo.apc.config.ApcConfig
   :members:

.. autoclass:: spflow.zoo.apc.config.ApcLossWeights
   :members:

.. autoclass:: spflow.zoo.apc.config.ApcTrainConfig
   :members:

Model
~~~~~

.. autoclass:: spflow.zoo.apc.model.AutoencodingPC
   :members:

Decoders
~~~~~~~~

.. autoclass:: spflow.zoo.apc.decoders.MLPDecoder1D
   :members:

.. autoclass:: spflow.zoo.apc.decoders.ConvDecoder2D
   :members:

Trainer Helpers
~~~~~~~~~~~~~~~

.. autofunction:: spflow.zoo.apc.train.train_apc_step

.. autofunction:: spflow.zoo.apc.train.evaluate_apc

.. autofunction:: spflow.zoo.apc.train.fit_apc

Minimal Example (Einet APC)
---------------------------

.. code-block:: python

    import torch
    from spflow.zoo.apc.config import ApcConfig, ApcLossWeights, ApcTrainConfig
    from spflow.zoo.apc.decoders import MLPDecoder1D
    from spflow.zoo.apc.encoders.einet_joint_encoder import EinetJointEncoder
    from spflow.zoo.apc.model import AutoencodingPC
    from spflow.zoo.apc.train import fit_apc

    encoder = EinetJointEncoder(
        num_x_features=32,
        latent_dim=8,
        num_sums=8,
        num_leaves=8,
        depth=3,
        num_repetitions=1,
        layer_type="linsum",
        structure="top-down",
    )
    decoder = MLPDecoder1D(latent_dim=8, output_dim=32, hidden_dims=(128, 128))

    cfg = ApcConfig(
        latent_dim=8,
        rec_loss="mse",
        sample_tau=1.0,
        loss_weights=ApcLossWeights(rec=1.0, kld=0.1, nll=1.0),
    )
    model = AutoencodingPC(encoder=encoder, decoder=decoder, config=cfg)

    data = torch.randn(512, 32)
    history = fit_apc(model, data, config=ApcTrainConfig(epochs=5, batch_size=64))

Conv-PC APC Note
----------------

``ConvPcJointEncoder`` supports image-shaped inputs and latent fusion at a configurable hierarchy depth. In the current implementation, ``latent_dim`` must match the feature count at the selected latent fusion depth.
