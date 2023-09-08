from typing import List

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class LogNormal:  # ToDo: backend über tl.getBackend() abfragen
    def __new__(cls, scope: Scope, mean: float = 0.0, std: float = 1.0):
        from spflow.base.structure.general.nodes.leaves.parametric.log_normal import LogNormal as TensorlyLogNormal
        from spflow.torch.structure.general.nodes.leaves.parametric.log_normal import LogNormal as TorchLogNormal
        """TODO"""
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyLogNormal(scope=scope, mean=mean, std=std)
        elif backend == "pytorch":
            return TorchLogNormal(scope=scope, mean=mean, std=std)
        else:
            raise NotImplementedError("LogNormal is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        from spflow.base.structure.general.nodes.leaves.parametric.log_normal import LogNormal as TensorlyLogNormal
        from spflow.torch.structure.general.nodes.leaves.parametric.log_normal import LogNormal as TorchLogNormal
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyLogNormal.accepts(signatures)
        elif backend == "pytorch":
            return TorchLogNormal.accepts(signatures)
        else:
            raise NotImplementedError("LogNormal is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]):
        from spflow.base.structure.general.nodes.leaves.parametric.log_normal import LogNormal as TensorlyLogNormal
        from spflow.torch.structure.general.nodes.leaves.parametric.log_normal import LogNormal as TorchLogNormal
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyLogNormal.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchLogNormal.from_signatures(signatures)
        else:
            raise NotImplementedError("LogNormal is not implemented for this backend")
