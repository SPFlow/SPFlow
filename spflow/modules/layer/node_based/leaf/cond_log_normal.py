from typing import List, Optional, Callable, Union

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class CondLogNormalLayer:  # ToDo: backend über T.getBackend() abfragen
    def __new__(
        cls,
        scope: Union[Scope, list[Scope]],
        cond_f: Optional[Union[Callable, list[Callable]]] = None,
        n_nodes: int = 1,
        **kwargs,
    ):
        from spflow.base.structure.general.layer.leaf.cond_log_normal import (
            CondLogNormalLayer as TensorlyCondLogNormal,
        )
        from spflow.torch.structure.general.layer.leaf.cond_log_normal import (
            CondLogNormalLayer as TorchCondLogNormal,
        )

        """TODO"""
        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyCondLogNormal(scope=scope, cond_f=cond_f, n_nodes=n_nodes, **kwargs)
        elif backend == "pytorch":
            return TorchCondLogNormal(scope=scope, cond_f=cond_f, n_nodes=n_nodes, **kwargs)
        else:
            raise NotImplementedError("CondLogNormal is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        from spflow.base.structure.general.layer.leaf.cond_log_normal import (
            CondLogNormalLayer as TensorlyCondLogNormal,
        )
        from spflow.torch.structure.general.layer.leaf.cond_log_normal import (
            CondLogNormalLayer as TorchCondLogNormal,
        )

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyCondLogNormal.accepts(signatures)
        elif backend == "pytorch":
            return TorchCondLogNormal.accepts(signatures)
        else:
            raise NotImplementedError("CondLogNormal is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]):
        from spflow.base.structure.general.layer.leaf.cond_log_normal import (
            CondLogNormalLayer as TensorlyCondLogNormal,
        )
        from spflow.torch.structure.general.layer.leaf.cond_log_normal import (
            CondLogNormalLayer as TorchCondLogNormal,
        )

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyCondLogNormal.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchCondLogNormal.from_signatures(signatures)
        else:
            raise NotImplementedError("CondLogNormal is not implemented for this backend")
