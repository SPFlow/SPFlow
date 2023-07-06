from typing import List, Union

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope
from spflow.tensorly.utils.helper_functions import T

class LogNormalLayer:  # ToDo: backend über tl.getBackend() abfragen
    def __new__(cls, scope: Union[Scope, List[Scope]],
        mean: Union[int, float, List[float], T] = 0.0,
        std: Union[int, float, List[float], T] = 1.0,
        n_nodes: int = 1,
        **kwargs):
        from spflow.tensorly.structure.general.layers.leaves import LogNormalLayer as TensorlyLogNormal
        from spflow.torch.structure.general.layers.leaves import LogNormalLayer as TorchLogNormal
        """TODO"""
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyLogNormal(scope=scope, mean=mean, std=std, n_nodes=n_nodes, **kwargs)
        elif backend == "pytorch":
            return TorchLogNormal(scope=scope, mean=mean, std=std, n_nodes=n_nodes, **kwargs)
        else:
            raise NotImplementedError("LogNormal is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        from spflow.tensorly.structure.general.layers.leaves import LogNormalLayer as TensorlyLogNormal
        from spflow.torch.structure.general.layers.leaves import LogNormalLayer as TorchLogNormal
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyLogNormal.accepts(signatures)
        elif backend == "pytorch":
            return TorchLogNormal.accepts(signatures)
        else:
            raise NotImplementedError("LogNormal is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]):
        from spflow.tensorly.structure.general.layers.leaves import LogNormalLayer as TensorlyLogNormal
        from spflow.torch.structure.general.layers.leaves import LogNormalLayer as TorchLogNormal
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyLogNormal.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchLogNormal.from_signatures(signatures)
        else:
            raise NotImplementedError("LogNormal is not implemented for this backend")
