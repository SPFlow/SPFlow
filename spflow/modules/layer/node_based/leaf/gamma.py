from typing import List, Union

import tensorly as tl
from spflow.tensor.ops import Tensor
from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class GammaLayer:
    def __new__(
        cls,
        scope: Union[Scope, list[Scope]],
        alpha: Union[int, float, list[float], Tensor] = 1.0,
        beta: Union[int, float, list[float], Tensor] = 1.0,
        n_nodes: int = 1,
        **kwargs,
    ):
        from spflow.base.structure.general.layer.leaf.gamma import GammaLayer as TensorlyGamma
        from spflow.torch.structure.general.layer.leaf.gamma import GammaLayer as TorchGamma

        """TODO"""
        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyGamma(scope=scope, alpha=alpha, beta=beta, n_nodes=n_nodes, **kwargs)
        elif backend == "pytorch":
            return TorchGamma(scope=scope, alpha=alpha, beta=beta, n_nodes=n_nodes, **kwargs)
        else:
            raise NotImplementedError("Gamma is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        from spflow.base.structure.general.layer.leaf.gamma import GammaLayer as TensorlyGamma
        from spflow.torch.structure.general.layer.leaf.gamma import GammaLayer as TorchGamma

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyGamma.accepts(signatures)
        elif backend == "pytorch":
            return TorchGamma.accepts(signatures)
        else:
            raise NotImplementedError("Gamma is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]):
        from spflow.base.structure.general.layer.leaf.gamma import GammaLayer as TensorlyGamma
        from spflow.torch.structure.general.layer.leaf.gamma import GammaLayer as TorchGamma

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyGamma.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchGamma.from_signatures(signatures)
        else:
            raise NotImplementedError("Gamma is not implemented for this backend")
