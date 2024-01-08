from typing import List, Optional, Callable

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class CondExponential:  # ToDo: backend über T.getBackend() abfragen
    def __new__(cls, scope: Scope, cond_f: Optional[Callable] = None):
        from spflow.base.structure.general.node.leaf.cond_exponential import (
            CondExponential as TensorlyCondExponential,
        )
        from spflow.torch.structure.general.node.leaf.cond_exponential import (
            CondExponential as TorchCondExponential,
        )

        """TODO"""
        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyCondExponential(scope=scope, cond_f=cond_f)
        elif backend == "pytorch":
            return TorchCondExponential(scope=scope, cond_f=cond_f)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        from spflow.base.structure.general.node.leaf.cond_exponential import (
            CondExponential as TensorlyCondExponential,
        )
        from spflow.torch.structure.general.node.leaf.cond_exponential import (
            CondExponential as TorchCondExponential,
        )

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyCondExponential.accepts(signatures)
        elif backend == "pytorch":
            return TorchCondExponential.accepts(signatures)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]):
        from spflow.base.structure.general.node.leaf.cond_exponential import (
            CondExponential as TensorlyCondExponential,
        )
        from spflow.torch.structure.general.node.leaf.cond_exponential import (
            CondExponential as TorchCondExponential,
        )

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyCondExponential.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchCondExponential.from_signatures(signatures)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")
