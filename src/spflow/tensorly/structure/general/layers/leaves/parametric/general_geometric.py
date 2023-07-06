from typing import List, Union

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope
from spflow.tensorly.utils.helper_functions import T


class GeometricLayer:  # ToDo: backend über tl.getBackend() abfragen
    def __new__(cls, scope: Union[Scope, List[Scope]],
        p: Union[int, float, List[float], T] = 0.5,
        n_nodes: int = 1,
        **kwargs):
        from spflow.tensorly.structure.general.layers.leaves import GeometricLayer as TensorlyGeometric
        from spflow.torch.structure.general.layers.leaves import GeometricLayer as TorchGeometric
        """TODO"""
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyGeometric(scope=scope, p=p, n_nodes=n_nodes, **kwargs)
        elif backend == "pytorch":
            return TorchGeometric(scope=scope, p=p, n_nodes=n_nodes, **kwargs)
        else:
            raise NotImplementedError("Geometric is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        from spflow.tensorly.structure.general.layers.leaves import GeometricLayer as TensorlyGeometric
        from spflow.torch.structure.general.layers.leaves import GeometricLayer as TorchGeometric
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyGeometric.accepts(signatures)
        elif backend == "pytorch":
            return TorchGeometric.accepts(signatures)
        else:
            raise NotImplementedError("Geometric is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]):
        from spflow.tensorly.structure.general.layers.leaves import GeometricLayer as TensorlyGeometric
        from spflow.torch.structure.general.layers.leaves import GeometricLayer as TorchGeometric
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyGeometric.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchGeometric.from_signatures(signatures)
        else:
            raise NotImplementedError("Geometric is not implemented for this backend")
