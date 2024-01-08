from typing import List, Union

import tensorly as tl
from spflow.tensor.ops import Tensor

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class BernoulliLayer:
    def __new__(
        cls,
        scope: Union[Scope, list[Scope]],
        p: Union[int, float, list[float], Tensor] = 0.5,
        n_nodes: int = 1,
        **kwargs,
    ):
        # from spflow.base.structure.general.layer.leaf.bernoulli import BernoulliLayer as TensorlyBernoulli
        from spflow.base.structure.general.layer.leaf.bernoulli import BernoulliLayer as TensorlyBernoulli
        from spflow.torch.structure.general.layer.leaf.bernoulli import BernoulliLayer as TorchBernoulli

        """TODO"""
        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyBernoulli(scope=scope, p=p, n_nodes=n_nodes, **kwargs)
        elif backend == "pytorch":
            return TorchBernoulli(scope=scope, p=p, n_nodes=n_nodes, **kwargs)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        from spflow.base.structure.general.layer.leaf.bernoulli import BernoulliLayer as TensorlyBernoulli
        from spflow.torch.structure.general.layer.leaf.bernoulli import BernoulliLayer as TorchBernoulli

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyBernoulli.accepts(signatures)
        elif backend == "pytorch":
            return TorchBernoulli.accepts(signatures)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]):
        from spflow.base.structure.general.layer.leaf.bernoulli import BernoulliLayer as TensorlyBernoulli
        from spflow.torch.structure.general.layer.leaf.bernoulli import BernoulliLayer as TorchBernoulli

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyBernoulli.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchBernoulli.from_signatures(signatures)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")
