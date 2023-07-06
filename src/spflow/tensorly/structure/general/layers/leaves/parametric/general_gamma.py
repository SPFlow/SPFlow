from typing import List, Union

import tensorly as tl
from spflow.tensorly.utils.helper_functions import T
from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class GammaLayer:
    def __new__(cls, scope: Union[Scope, List[Scope]],
        alpha: Union[int, float, List[float], T] = 1.0,
        beta: Union[int, float, List[float], T] = 1.0,
        n_nodes: int = 1,
        **kwargs):
        from spflow.tensorly.structure.general.layers.leaves import GammaLayer as TensorlyGamma
        from spflow.torch.structure.general.layers.leaves import GammaLayer as TorchGamma
        """TODO"""
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyGamma(scope=scope, alpha=alpha, beta=beta, n_nodes=n_nodes, **kwargs)
        elif backend == "pytorch":
            return TorchGamma(scope=scope, alpha=alpha, beta=beta, n_nodes=n_nodes, **kwargs)
        else:
            raise NotImplementedError("Gamma is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        from spflow.tensorly.structure.general.layers.leaves import GammaLayer as TensorlyGamma
        from spflow.torch.structure.general.layers.leaves import GammaLayer as TorchGamma
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyGamma.accepts(signatures)
        elif backend == "pytorch":
            return TorchGamma.accepts(signatures)
        else:
            raise NotImplementedError("Gamma is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]):
        from spflow.tensorly.structure.general.layers.leaves import GammaLayer as TensorlyGamma
        from spflow.torch.structure.general.layers.leaves import GammaLayer as TorchGamma
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyGamma.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchGamma.from_signatures(signatures)
        else:
            raise NotImplementedError("Gamma is not implemented for this backend")
