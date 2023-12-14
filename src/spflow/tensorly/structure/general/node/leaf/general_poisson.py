from typing import List

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class Poisson:  # ToDo: backend über tl.getBackend() abfragen
    def __new__(cls, scope: Scope, l: float = 1.0):
        from spflow.base.structure.general.node.leaf.poisson import Poisson as TensorlyPoisson
        from spflow.torch.structure.general.node.leaf.poisson import Poisson as TorchPoisson
        """TODO"""
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyPoisson(scope=scope, l=l)
        elif backend == "pytorch":
            return TorchPoisson(scope=scope, l=l)
        else:
            raise NotImplementedError("Poisson is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        from spflow.base.structure.general.node.leaf.poisson import Poisson as TensorlyPoisson
        from spflow.torch.structure.general.node.leaf.poisson import Poisson as TorchPoisson
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyPoisson.accepts(signatures)
        elif backend == "pytorch":
            return TorchPoisson.accepts(signatures)
        else:
            raise NotImplementedError("Poisson is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]):
        from spflow.base.structure.general.node.leaf.poisson import Poisson as TensorlyPoisson
        from spflow.torch.structure.general.node.leaf.poisson import Poisson as TorchPoisson
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyPoisson.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchPoisson.from_signatures(signatures)
        else:
            raise NotImplementedError("Poisson is not implemented for this backend")
