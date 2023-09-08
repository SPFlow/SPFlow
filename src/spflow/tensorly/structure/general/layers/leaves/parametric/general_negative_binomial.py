from typing import List, Union

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope
from spflow.tensorly.utils.helper_functions import T


class NegativeBinomialLayer:  # ToDo: backend über tl.getBackend() abfragen
    def __new__(cls, scope: Union[Scope, List[Scope]],
        n: Union[int, List[int], T],
        p: Union[int, float, List[float], T] = 0.5,
        n_nodes: int = 1,
        **kwargs):
        from spflow.base.structure.general.layers.leaves.parametric.negative_binomial import NegativeBinomialLayer as TensorlyNegativeBinomial
        from spflow.torch.structure.general.layers.leaves.parametric.negative_binomial import NegativeBinomialLayer as TorchNegativeBinomial
        """TODO"""
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyNegativeBinomial(scope=scope, n=n, p=p, n_nodes=n_nodes, **kwargs)
        elif backend == "pytorch":
            return TorchNegativeBinomial(scope=scope, n=n, p=p, n_nodes=n_nodes, **kwargs)
        else:
            raise NotImplementedError("NegativeBinomial is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        from spflow.base.structure.general.layers.leaves.parametric.negative_binomial import NegativeBinomialLayer as TensorlyNegativeBinomial
        from spflow.torch.structure.general.layers.leaves.parametric.negative_binomial import NegativeBinomialLayer as TorchNegativeBinomial
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyNegativeBinomial.accepts(signatures)
        elif backend == "pytorch":
            return TorchNegativeBinomial.accepts(signatures)
        else:
            raise NotImplementedError("NegativeBinomial is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]):
        from spflow.base.structure.general.layers.leaves.parametric.negative_binomial import NegativeBinomialLayer as TensorlyNegativeBinomial
        from spflow.torch.structure.general.layers.leaves.parametric.negative_binomial import NegativeBinomialLayer as TorchNegativeBinomial
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyNegativeBinomial.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchNegativeBinomial.from_signatures(signatures)
        else:
            raise NotImplementedError("NegativeBinomial is not implemented for this backend")
