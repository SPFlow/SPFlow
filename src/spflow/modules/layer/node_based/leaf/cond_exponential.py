from typing import List, Optional, Callable, Union

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class CondExponentialLayer:  # ToDo: backend über T.getBackend() abfragen
    def __new__(
        cls,
        scope: Union[Scope, list[Scope]],
        cond_f: Optional[Union[Callable, list[Callable]]] = None,
        n_nodes: int = 1,
        **kwargs,
    ):
        from spflow.base.structure.general.layer.leaf.cond_exponential import (
            CondExponentialLayer as TensorlyCondExponential,
        )
        from spflow.torch.structure.general.layer.leaf.cond_exponential import (
            CondExponentialLayer as TorchCondExponential,
        )

        """TODO"""
        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyCondExponential(scope=scope, cond_f=cond_f, n_nodes=n_nodes, **kwargs)
        elif backend == "pytorch":
            return TorchCondExponential(scope=scope, cond_f=cond_f, n_nodes=n_nodes, **kwargs)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        from spflow.base.structure.general.layer.leaf.cond_exponential import (
            CondExponentialLayer as TensorlyCondExponential,
        )
        from spflow.torch.structure.general.layer.leaf.cond_exponential import (
            CondExponentialLayer as TorchCondExponential,
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
        from spflow.base.structure.general.layer.leaf.cond_exponential import (
            CondExponentialLayer as TensorlyCondExponential,
        )
        from spflow.torch.structure.general.layer.leaf.cond_exponential import (
            CondExponentialLayer as TorchCondExponential,
        )

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyCondExponential.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchCondExponential.from_signatures(signatures)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")
