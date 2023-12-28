from typing import List

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class Gamma:
    def __new__(cls, scope: Scope, alpha: float = 1.0, beta: float = 1.0):
        from spflow.base.structure.general.node.leaf.gamma import Gamma as TensorlyGamma
        from spflow.torch.structure.general.node.leaf.gamma import Gamma as TorchGamma

        """TODO"""
        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyGamma(scope=scope, alpha=alpha, beta=beta)
        elif backend == "pytorch":
            return TorchGamma(scope=scope, alpha=alpha, beta=beta)
        else:
            raise NotImplementedError("Gamma is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        from spflow.base.structure.general.node.leaf.gamma import Gamma as TensorlyGamma
        from spflow.torch.structure.general.node.leaf.gamma import Gamma as TorchGamma

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyGamma.accepts(signatures)
        elif backend == "pytorch":
            return TorchGamma.accepts(signatures)
        else:
            raise NotImplementedError("Gamma is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]):
        from spflow.base.structure.general.node.leaf.gamma import Gamma as TensorlyGamma
        from spflow.torch.structure.general.node.leaf.gamma import Gamma as TorchGamma

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyGamma.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchGamma.from_signatures(signatures)
        else:
            raise NotImplementedError("Gamma is not implemented for this backend")
