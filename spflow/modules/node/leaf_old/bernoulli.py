from typing import List

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class Bernoulli:
    def __new__(cls, scope: Scope, p: float = 0.5):
        from spflow.base.structure.general.node.leaf.bernoulli import Bernoulli as TensorlyBernoulli
        from spflow.torch.structure.general.node.leaf.bernoulli import Bernoulli as TorchBernoulli

        """TODO"""
        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyBernoulli(scope=scope, p=p)
        elif backend == "pytorch":
            return TorchBernoulli(scope=scope, p=p)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        from spflow.base.structure.general.node.leaf.bernoulli import Bernoulli as TensorlyBernoulli
        from spflow.torch.structure.general.node.leaf.bernoulli import Bernoulli as TorchBernoulli

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyBernoulli.accepts(signatures)
        elif backend == "pytorch":
            return TorchBernoulli.accepts(signatures)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]):
        from spflow.base.structure.general.node.leaf.bernoulli import Bernoulli as TensorlyBernoulli
        from spflow.torch.structure.general.node.leaf.bernoulli import Bernoulli as TorchBernoulli

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyBernoulli.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchBernoulli.from_signatures(signatures)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")
