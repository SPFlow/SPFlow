from typing import List, Optional, Callable

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class CondGaussian:  # ToDo: backend über tl.getBackend() abfragen
    def __new__(cls, scope: Scope, cond_f: Optional[Callable] = None):
        from spflow.tensorly.structure.general.nodes.leaves import CondGaussian as TensorlyCondGaussian
        from spflow.torch.structure.general.nodes.leaves import CondGaussian as TorchCondGaussian
        """TODO"""
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondGaussian(scope=scope, cond_f=cond_f)
        elif backend == "pytorch":
            return TorchCondGaussian(scope=scope, cond_f=cond_f)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        from spflow.tensorly.structure.general.nodes.leaves import CondGaussian as TensorlyCondGaussian
        from spflow.torch.structure.general.nodes.leaves import CondGaussian as TorchCondGaussian
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondGaussian.accepts(signatures)
        elif backend == "pytorch":
            return TorchCondGaussian.accepts(signatures)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]):
        from spflow.tensorly.structure.general.nodes.leaves import CondGaussian as TensorlyCondGaussian
        from spflow.torch.structure.general.nodes.leaves import CondGaussian as TorchCondGaussian
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondGaussian.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchCondGaussian.from_signatures(signatures)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")
