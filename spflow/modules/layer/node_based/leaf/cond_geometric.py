from typing import List, Optional, Callable, Union

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class CondGeometricLayer:  # ToDo: backend über T.getBackend() abfragen
    def __new__(
        cls,
        scope: Union[Scope, list[Scope]],
        cond_f: Optional[Union[Callable, list[Callable]]] = None,
        n_nodes: int = 1,
        **kwargs,
    ):
        from spflow.base.structure.general.layer.leaf.cond_geometric import (
            CondGeometricLayer as TensorlyCondGeometric,
        )
        from spflow.torch.structure.general.layer.leaf.cond_geometric import (
            CondGeometricLayer as TorchCondGeometric,
        )

        """TODO"""
        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyCondGeometric(scope=scope, cond_f=cond_f, n_nodes=n_nodes, **kwargs)
        elif backend == "pytorch":
            return TorchCondGeometric(scope=scope, cond_f=cond_f, n_nodes=n_nodes, **kwargs)
        else:
            raise NotImplementedError("CondGeometric is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        from spflow.base.structure.general.layer.leaf.cond_geometric import (
            CondGeometricLayer as TensorlyCondGeometric,
        )
        from spflow.torch.structure.general.layer.leaf.cond_geometric import (
            CondGeometricLayer as TorchCondGeometric,
        )

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyCondGeometric.accepts(signatures)
        elif backend == "pytorch":
            return TorchCondGeometric.accepts(signatures)
        else:
            raise NotImplementedError("CondGeometric is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]):
        from spflow.base.structure.general.layer.leaf.cond_geometric import (
            CondGeometricLayer as TensorlyCondGeometric,
        )
        from spflow.torch.structure.general.layer.leaf.cond_geometric import (
            CondGeometricLayer as TorchCondGeometric,
        )

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyCondGeometric.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchCondGeometric.from_signatures(signatures)
        else:
            raise NotImplementedError("CondGeometric is not implemented for this backend")
