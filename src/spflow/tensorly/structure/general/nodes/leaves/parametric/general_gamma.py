from typing import List

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class Gamma:
    def __new__(cls, scope: Scope, alpha: float = 1.0, beta: float = 1.0):
        from spflow.tensorly.structure.general.nodes.leaves import Gamma as TensorlyGamma
        from spflow.torch.structure.general.nodes.leaves import Gamma as TorchGamma
        """TODO"""
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyGamma(scope=scope, alpha=alpha, beta=beta)
        elif backend == "pytorch":
            return TorchGamma(scope=scope, alpha=alpha, beta=beta)
        else:
            raise NotImplementedError("Gamma is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        from spflow.tensorly.structure.general.nodes.leaves import Gamma as TensorlyGamma
        from spflow.torch.structure.general.nodes.leaves import Gamma as TorchGamma
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyGamma.accepts(signatures)
        elif backend == "pytorch":
            return TorchGamma.accepts(signatures)
        else:
            raise NotImplementedError("Gamma is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]):
        from spflow.tensorly.structure.general.nodes.leaves import Gamma as TensorlyGamma
        from spflow.torch.structure.general.nodes.leaves import Gamma as TorchGamma
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyGamma.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchGamma.from_signatures(signatures)
        else:
            raise NotImplementedError("Gamma is not implemented for this backend")
