from typing import List

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class Geometric:  # ToDo: backend über tl.getBackend() abfragen
    def __new__(cls, scope: Scope, p: float = 0.5):
        from spflow.base.structure.general.nodes.leaves.parametric.geometric import Geometric as TensorlyGeometric
        from spflow.torch.structure.general.nodes.leaves.parametric.geometric import Geometric as TorchGeometric
        """TODO"""
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyGeometric(scope=scope, p=p)
        elif backend == "pytorch":
            return TorchGeometric(scope=scope, p=p)
        else:
            raise NotImplementedError("Geometric is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        from spflow.base.structure.general.nodes.leaves.parametric.geometric import Geometric as TensorlyGeometric
        from spflow.torch.structure.general.nodes.leaves.parametric.geometric import Geometric as TorchGeometric
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyGeometric.accepts(signatures)
        elif backend == "pytorch":
            return TorchGeometric.accepts(signatures)
        else:
            raise NotImplementedError("Geometric is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]):
        from spflow.base.structure.general.nodes.leaves.parametric.geometric import Geometric as TensorlyGeometric
        from spflow.torch.structure.general.nodes.leaves.parametric.geometric import Geometric as TorchGeometric
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyGeometric.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchGeometric.from_signatures(signatures)
        else:
            raise NotImplementedError("Geometric is not implemented for this backend")
