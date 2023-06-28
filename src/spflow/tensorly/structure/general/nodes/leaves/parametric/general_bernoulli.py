from typing import List

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class Bernoulli:
    def __new__(cls, scope: Scope, p: float = 0.5):
        from spflow.tensorly.structure.general.nodes.leaves.parametric.bernoulli import Bernoulli as TensorlyBernoulli
        from spflow.torch.structure.general.nodes.leaves.parametric.bernoulli import Bernoulli as TorchBernoulli
        """TODO"""
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyBernoulli(scope=scope, p=p)
        elif backend == "pytorch":
            return TorchBernoulli(scope=scope, p=p)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        from spflow.tensorly.structure.general.nodes.leaves.parametric.bernoulli import Bernoulli as TensorlyBernoulli
        from spflow.torch.structure.general.nodes.leaves.parametric.bernoulli import Bernoulli as TorchBernoulli
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyBernoulli.accepts(signatures)
        elif backend == "pytorch":
            return TorchBernoulli.accepts(signatures)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]):
        from spflow.tensorly.structure.general.nodes.leaves.parametric.bernoulli import Bernoulli as TensorlyBernoulli
        from spflow.torch.structure.general.nodes.leaves.parametric.bernoulli import Bernoulli as TorchBernoulli
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyBernoulli.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchBernoulli.from_signatures(signatures)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")

