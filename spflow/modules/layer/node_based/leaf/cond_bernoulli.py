from typing import List, Optional, Callable, Union

import tensorly as tl
from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class CondBernoulliLayer:
    def __new__(
        cls,
        scope: Union[Scope, list[Scope]],
        cond_f: Optional[Union[Callable, list[Callable]]] = None,
        n_nodes: int = 1,
        **kwargs,
    ):
        from spflow.base.structure.general.layer.leaf.cond_bernoulli import (
            CondBernoulliLayer as TensorlyCondBernoulli,
        )
        from spflow.torch.structure.general.layer.leaf.cond_bernoulli import (
            CondBernoulliLayer as TorchCondBernoulli,
        )

        """TODO"""
        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyCondBernoulli(scope=scope, cond_f=cond_f, n_nodes=n_nodes, **kwargs)
        elif backend == "pytorch":
            return TorchCondBernoulli(scope=scope, cond_f=cond_f, n_nodes=n_nodes, **kwargs)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        from spflow.base.structure.general.layer.leaf.cond_bernoulli import (
            CondBernoulliLayer as TensorlyCondBernoulli,
        )
        from spflow.torch.structure.general.layer.leaf.cond_bernoulli import (
            CondBernoulliLayer as TorchCondBernoulli,
        )

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyCondBernoulli.accepts(signatures)
        elif backend == "pytorch":
            return TorchCondBernoulli.accepts(signatures)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]):
        from spflow.base.structure.general.layer.leaf.cond_bernoulli import (
            CondBernoulliLayer as TensorlyCondBernoulli,
        )
        from spflow.torch.structure.general.layer.leaf.cond_bernoulli import (
            CondBernoulliLayer as TorchCondBernoulli,
        )

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyCondBernoulli.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchCondBernoulli.from_signatures(signatures)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")
