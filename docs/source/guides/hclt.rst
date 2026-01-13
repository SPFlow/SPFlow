====
HCLT
====

Hidden Chow–Liu Trees (HCLTs) are a simple latent-variable model built from a Chow–Liu tree
over observed variables. SPFlow provides helpers to learn an HCLT structure (optionally top-k)
and return it as a standard :class:`spflow.modules.module.Module`.

Binary HCLT
===========

For binary observed variables (values in ``{0, 1}``), use :func:`spflow.learn.hclt.learn_hclt_binary`.

.. code-block:: python

    import torch
    from spflow.learn import learn_hclt_binary

    # (N, D) with values in {0, 1}
    data = torch.randint(0, 2, (1024, 50), dtype=torch.float32)

    # Build an HCLT with H latent states per variable; optionally mix over top-k CLTs
    model = learn_hclt_binary(
        data,
        num_hidden_cats=8,
        num_trees=5,
        init="uniform",
    )

    ll = model.log_likelihood(data)  # (N, 1, 1, 1)


Categorical HCLT
================

For categorical observed variables (values in ``{0, ..., K-1}``), use
:func:`spflow.learn.hclt.learn_hclt_categorical`.

.. code-block:: python

    import torch
    from spflow.learn import learn_hclt_categorical

    # (N, D) with values in {0, ..., K-1}
    data = torch.randint(0, 4, (1024, 50), dtype=torch.float32)

    model = learn_hclt_categorical(
        data,
        num_hidden_cats=8,
        num_cats=4,
        num_trees=3,
        init="uniform",
    )

    ll = model.log_likelihood(data)  # (N, 1, 1, 1)

