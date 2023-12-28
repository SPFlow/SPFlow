from typing import List, Union

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope
from spflow.tensor.ops import Tensor


class NegativeBinomialLayer:  # ToDo: backend über T.getBackend() abfragen
    def __new__(
        cls,
        scope: Union[Scope, list[Scope]],
        n: Union[int, list[int], Tensor],
        p: Union[int, float, list[float], Tensor] = 0.5,
        n_nodes: int = 1,
        **kwargs,
    ):
        from spflow.base.structure.general.layer.leaf.negative_binomial import (
            NegativeBinomialLayer as TensorlyNegativeBinomial,
        )
        from spflow.torch.structure.general.layer.leaf.negative_binomial import (
            NegativeBinomialLayer as TorchNegativeBinomial,
        )

        """TODO"""
        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyNegativeBinomial(scope=scope, n=n, p=p, n_nodes=n_nodes, **kwargs)
        elif backend == "pytorch":
            return TorchNegativeBinomial(scope=scope, n=n, p=p, n_nodes=n_nodes, **kwargs)
        else:
            raise NotImplementedError("NegativeBinomial is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        from spflow.base.structure.general.layer.leaf.negative_binomial import (
            NegativeBinomialLayer as TensorlyNegativeBinomial,
        )
        from spflow.torch.structure.general.layer.leaf.negative_binomial import (
            NegativeBinomialLayer as TorchNegativeBinomial,
        )

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyNegativeBinomial.accepts(signatures)
        elif backend == "pytorch":
            return TorchNegativeBinomial.accepts(signatures)
        else:
            raise NotImplementedError("NegativeBinomial is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]):
        from spflow.base.structure.general.layer.leaf.negative_binomial import (
            NegativeBinomialLayer as TensorlyNegativeBinomial,
        )
        from spflow.torch.structure.general.layer.leaf.negative_binomial import (
            NegativeBinomialLayer as TorchNegativeBinomial,
        )

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyNegativeBinomial.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchNegativeBinomial.from_signatures(signatures)
        else:
            raise NotImplementedError("NegativeBinomial is not implemented for this backend")
