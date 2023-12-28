from typing import List, Optional, Callable, Union

import tensorly as tl
from spflow.tensor.ops import Tensor
from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class CondBinomialLayer:  # ToDo: backend über T.getBackend() abfragen
    def __new__(
        cls,
        scope: Union[Scope, list[Scope]],
        n: Union[int, list[int], Tensor],
        cond_f: Optional[Union[Callable, list[Callable]]] = None,
        n_nodes: int = 1,
        **kwargs,
    ):
        from spflow.base.structure.general.layer.leaf.cond_binomial import (
            CondBinomialLayer as TensorlyCondBinomial,
        )
        from spflow.torch.structure.general.layer.leaf.cond_binomial import (
            CondBinomialLayer as TorchCondBinomial,
        )

        """TODO"""
        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyCondBinomial(scope=scope, n=n, cond_f=cond_f, n_nodes=n_nodes, **kwargs)
        elif backend == "pytorch":
            return TorchCondBinomial(scope=scope, n=n, cond_f=cond_f, n_nodes=n_nodes, **kwargs)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        from spflow.base.structure.general.layer.leaf.cond_binomial import (
            CondBinomialLayer as TensorlyCondBinomial,
        )
        from spflow.torch.structure.general.layer.leaf.cond_binomial import (
            CondBinomialLayer as TorchCondBinomial,
        )

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyCondBinomial.accepts(signatures)
        elif backend == "pytorch":
            return TorchCondBinomial.accepts(signatures)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]):
        from spflow.base.structure.general.layer.leaf.cond_binomial import (
            CondBinomialLayer as TensorlyCondBinomial,
        )
        from spflow.torch.structure.general.layer.leaf.cond_binomial import (
            CondBinomialLayer as TorchCondBinomial,
        )

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyCondBinomial.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchCondBinomial.from_signatures(signatures)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")
