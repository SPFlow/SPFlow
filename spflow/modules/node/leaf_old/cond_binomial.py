from typing import List, Optional, Callable

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class CondBinomial:  # ToDo: backend über T.getBackend() abfragen
    def __new__(cls, scope: Scope, n: int, cond_f: Optional[Callable] = None):
        from spflow.base.structure.general.node.leaf.cond_binomial import CondBinomial as TensorlyCondBinomial
        from spflow.torch.structure.general.node.leaf.cond_binomial import CondBinomial as TorchCondBinomial

        """TODO"""
        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyCondBinomial(scope=scope, n=n, cond_f=cond_f)
        elif backend == "pytorch":
            return TorchCondBinomial(scope=scope, n=n, cond_f=cond_f)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        from spflow.base.structure.general.node.leaf.cond_binomial import CondBinomial as TensorlyCondBinomial
        from spflow.torch.structure.general.node.leaf.cond_binomial import CondBinomial as TorchCondBinomial

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyCondBinomial.accepts(signatures)
        elif backend == "pytorch":
            return TorchCondBinomial.accepts(signatures)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]):
        from spflow.base.structure.general.node.leaf.cond_binomial import CondBinomial as TensorlyCondBinomial
        from spflow.torch.structure.general.node.leaf.cond_binomial import CondBinomial as TorchCondBinomial

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyCondBinomial.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchCondBinomial.from_signatures(signatures)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")
