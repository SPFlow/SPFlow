from typing import List, Optional, Callable, Union

import tensorly as tl
from spflow.tensor.ops import Tensor

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class CondNegativeBinomialLayer:  # ToDo: backend über T.getBackend() abfragen
    def __new__(
        cls,
        scope: Union[Scope, list[Scope]],
        n: Union[int, list[int], Tensor],
        cond_f: Optional[Union[Callable, list[Callable]]] = None,
        n_nodes: int = 1,
        **kwargs,
    ):
        from spflow.base.structure.general.layer.leaf.cond_negative_binomial import (
            CondNegativeBinomialLayer as TensorlyCondNegativeBinomial,
        )
        from spflow.torch.structure.general.layer.leaf.cond_negative_binomial import (
            CondNegativeBinomialLayer as TorchCondNegativeBinomial,
        )

        """TODO"""
        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyCondNegativeBinomial(scope=scope, n=n, cond_f=cond_f, n_nodes=n_nodes, **kwargs)
        elif backend == "pytorch":
            return TorchCondNegativeBinomial(scope=scope, n=n, cond_f=cond_f, n_nodes=n_nodes, **kwargs)
        else:
            raise NotImplementedError("CondNegativeBinomial is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        from spflow.base.structure.general.layer.leaf.cond_negative_binomial import (
            CondNegativeBinomialLayer as TensorlyCondNegativeBinomial,
        )
        from spflow.torch.structure.general.layer.leaf.cond_negative_binomial import (
            CondNegativeBinomialLayer as TorchCondNegativeBinomial,
        )

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyCondNegativeBinomial.accepts(signatures)
        elif backend == "pytorch":
            return TorchCondNegativeBinomial.accepts(signatures)
        else:
            raise NotImplementedError("CondNegativeBinomial is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]):
        from spflow.base.structure.general.layer.leaf.cond_negative_binomial import (
            CondNegativeBinomialLayer as TensorlyCondNegativeBinomial,
        )
        from spflow.torch.structure.general.layer.leaf.cond_negative_binomial import (
            CondNegativeBinomialLayer as TorchCondNegativeBinomial,
        )

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyCondNegativeBinomial.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchCondNegativeBinomial.from_signatures(signatures)
        else:
            raise NotImplementedError("CondNegativeBinomial is not implemented for this backend")
