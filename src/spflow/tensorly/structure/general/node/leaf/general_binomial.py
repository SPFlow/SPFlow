from typing import List

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class Binomial:
    def __new__(cls, scope: Scope, n: int, p: float = 0.5):
        from spflow.base.structure.general.node.leaf.binomial import Binomial as TensorlyBinomial
        from spflow.torch.structure.general.node.leaf.binomial import Binomial as TorchBinomial
        """TODO"""
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyBinomial(scope=scope, n=n, p=p)
        elif backend == "pytorch":
            return TorchBinomial(scope=scope, n=n, p=p)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        from spflow.base.structure.general.node.leaf.binomial import Binomial as TensorlyBinomial
        from spflow.torch.structure.general.node.leaf.binomial import Binomial as TorchBinomial
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyBinomial.accepts(signatures)
        elif backend == "pytorch":
            return TorchBinomial.accepts(signatures)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]):
        from spflow.base.structure.general.node.leaf.binomial import Binomial as TensorlyBinomial
        from spflow.torch.structure.general.node.leaf.binomial import Binomial as TorchBinomial
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyBinomial.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchBinomial.from_signatures(signatures)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")

