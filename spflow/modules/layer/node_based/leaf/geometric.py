from typing import List, Union

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope
from spflow.tensor.ops import Tensor


class GeometricLayer:  # ToDo: backend über T.getBackend() abfragen
    def __new__(
        cls,
        scope: Union[Scope, list[Scope]],
        p: Union[int, float, list[float], Tensor] = 0.5,
        n_nodes: int = 1,
        **kwargs,
    ):
        from spflow.base.structure.general.layer.leaf.geometric import GeometricLayer as TensorlyGeometric
        from spflow.torch.structure.general.layer.leaf.geometric import GeometricLayer as TorchGeometric

        """TODO"""
        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyGeometric(scope=scope, p=p, n_nodes=n_nodes, **kwargs)
        elif backend == "pytorch":
            return TorchGeometric(scope=scope, p=p, n_nodes=n_nodes, **kwargs)
        else:
            raise NotImplementedError("Geometric is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        from spflow.base.structure.general.layer.leaf.geometric import GeometricLayer as TensorlyGeometric
        from spflow.torch.structure.general.layer.leaf.geometric import GeometricLayer as TorchGeometric

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyGeometric.accepts(signatures)
        elif backend == "pytorch":
            return TorchGeometric.accepts(signatures)
        else:
            raise NotImplementedError("Geometric is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]):
        from spflow.base.structure.general.layer.leaf.geometric import GeometricLayer as TensorlyGeometric
        from spflow.torch.structure.general.layer.leaf.geometric import GeometricLayer as TorchGeometric

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyGeometric.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchGeometric.from_signatures(signatures)
        else:
            raise NotImplementedError("Geometric is not implemented for this backend")
