from typing import List

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class NegativeBinomial:  # ToDo: backend über tl.getBackend() abfragen
    def __new__(cls, scope: Scope, n: int, p: float = 0.5):
        from spflow.base.structure.general.nodes.leaves.parametric.negative_binomial import NegativeBinomial as TensorlyNegativeBinomial
        from spflow.torch.structure.general.nodes.leaves.parametric.negative_binomial import NegativeBinomial as TorchNegativeBinomial
        """TODO"""
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyNegativeBinomial(scope=scope, n=n, p=p)
        elif backend == "pytorch":
            return TorchNegativeBinomial(scope=scope, n=n, p=p)
        else:
            raise NotImplementedError("NegativeBinomial is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        from spflow.base.structure.general.nodes.leaves.parametric.negative_binomial import NegativeBinomial as TensorlyNegativeBinomial
        from spflow.torch.structure.general.nodes.leaves.parametric.negative_binomial import NegativeBinomial as TorchNegativeBinomial
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyNegativeBinomial.accepts(signatures)
        elif backend == "pytorch":
            return TorchNegativeBinomial.accepts(signatures)
        else:
            raise NotImplementedError("NegativeBinomial is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]):
        from spflow.base.structure.general.nodes.leaves.parametric.negative_binomial import NegativeBinomial as TensorlyNegativeBinomial
        from spflow.torch.structure.general.nodes.leaves.parametric.negative_binomial import NegativeBinomial as TorchNegativeBinomial
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyNegativeBinomial.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchNegativeBinomial.from_signatures(signatures)
        else:
            raise NotImplementedError("NegativeBinomial is not implemented for this backend")
