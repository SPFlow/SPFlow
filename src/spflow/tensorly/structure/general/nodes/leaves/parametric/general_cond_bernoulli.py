from typing import List, Optional, Callable

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class CondBernoulli:
    def __new__(cls, scope: Scope, cond_f: Optional[Callable] = None):
        from spflow.base.structure.general.nodes.leaves.parametric.cond_bernoulli import CondBernoulli as TensorlyCondBernoulli
        from spflow.torch.structure.general.nodes.leaves.parametric.cond_bernoulli import CondBernoulli as TorchCondBernoulli
        """TODO"""
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondBernoulli(scope=scope, cond_f=cond_f)
        elif backend == "pytorch":
            return TorchCondBernoulli(scope=scope, cond_f=cond_f)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        from spflow.base.structure.general.nodes.leaves.parametric.cond_bernoulli import CondBernoulli as TensorlyCondBernoulli
        from spflow.torch.structure.general.nodes.leaves.parametric.cond_bernoulli import CondBernoulli as TorchCondBernoulli
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondBernoulli.accepts(signatures)
        elif backend == "pytorch":
            return TorchCondBernoulli.accepts(signatures)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]):
        from spflow.base.structure.general.nodes.leaves.parametric.cond_bernoulli import CondBernoulli as TensorlyCondBernoulli
        from spflow.torch.structure.general.nodes.leaves.parametric.cond_bernoulli import CondBernoulli as TorchCondBernoulli
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondBernoulli.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchCondBernoulli.from_signatures(signatures)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")
