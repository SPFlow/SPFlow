from typing import List, Optional, Callable

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class CondLogNormal:  # ToDo: backend über tl.getBackend() abfragen
    def __new__(cls, scope: Scope, cond_f: Optional[Callable] = None):
        from spflow.base.structure.general.node.leaf.cond_log_normal import CondLogNormal as TensorlyCondLogNormal
        from spflow.torch.structure.general.node.leaf.cond_log_normal import CondLogNormal as TorchCondLogNormal
        """TODO"""
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondLogNormal(scope=scope, cond_f=cond_f)
        elif backend == "pytorch":
            return TorchCondLogNormal(scope=scope, cond_f=cond_f)
        else:
            raise NotImplementedError("CondLogNormal is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        from spflow.base.structure.general.node.leaf.cond_log_normal import CondLogNormal as TensorlyCondLogNormal
        from spflow.torch.structure.general.node.leaf.cond_log_normal import CondLogNormal as TorchCondLogNormal
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondLogNormal.accepts(signatures)
        elif backend == "pytorch":
            return TorchCondLogNormal.accepts(signatures)
        else:
            raise NotImplementedError("CondLogNormal is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]):
        from spflow.base.structure.general.node.leaf.cond_log_normal import CondLogNormal as TensorlyCondLogNormal
        from spflow.torch.structure.general.node.leaf.cond_log_normal import CondLogNormal as TorchCondLogNormal
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondLogNormal.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchCondLogNormal.from_signatures(signatures)
        else:
            raise NotImplementedError("CondLogNormal is not implemented for this backend")
