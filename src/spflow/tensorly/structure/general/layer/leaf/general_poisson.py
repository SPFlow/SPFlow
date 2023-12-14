from typing import List, Union

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope
from spflow.tensorly.utils.helper_functions import T

class PoissonLayer:  # ToDo: backend über tl.getBackend() abfragen
    def __new__(cls, scope: Union[Scope, List[Scope]],
        l: Union[int, float, List[float], T] = 1.0,
        n_nodes: int = 1,
        **kwargs):
        from spflow.base.structure.general.layer.leaf.poisson import PoissonLayer as TensorlyPoisson
        from spflow.torch.structure.general.layer.leaf.poisson import PoissonLayer as TorchPoisson
        """TODO"""
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyPoisson(scope=scope, l=l, n_nodes=n_nodes, **kwargs)
        elif backend == "pytorch":
            return TorchPoisson(scope=scope, l=l, n_nodes=n_nodes, **kwargs)
        else:
            raise NotImplementedError("Poisson is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        from spflow.base.structure.general.layer.leaf.poisson import PoissonLayer as TensorlyPoisson
        from spflow.torch.structure.general.layer.leaf.poisson import PoissonLayer as TorchPoisson
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyPoisson.accepts(signatures)
        elif backend == "pytorch":
            return TorchPoisson.accepts(signatures)
        else:
            raise NotImplementedError("Poisson is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]):
        from spflow.base.structure.general.layer.leaf.poisson import PoissonLayer as TensorlyPoisson
        from spflow.torch.structure.general.layer.leaf.poisson import PoissonLayer as TorchPoisson
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyPoisson.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchPoisson.from_signatures(signatures)
        else:
            raise NotImplementedError("Poisson is not implemented for this backend")
