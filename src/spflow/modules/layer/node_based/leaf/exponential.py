from typing import List, Union

import tensorly as tl
from spflow.tensor.ops import Tensor

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class ExponentialLayer:  # ToDo: backend über T.getBackend() abfragen
    def __new__(
        cls,
        scope: Union[Scope, list[Scope]],
        l: Union[int, float, list[float], Tensor] = 1.0,
        n_nodes: int = 1,
        **kwargs,
    ):
        from spflow.base.structure.general.layer.leaf.exponential import (
            ExponentialLayer as TensorlyExponential,
        )
        from spflow.torch.structure.general.layer.leaf.exponential import ExponentialLayer as TorchExponential

        """TODO"""
        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyExponential(scope=scope, l=l, n_nodes=n_nodes, **kwargs)
        elif backend == "pytorch":
            return TorchExponential(scope=scope, l=l, n_nodes=n_nodes, **kwargs)
        else:
            raise NotImplementedError("Exponential is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        from spflow.base.structure.general.layer.leaf.exponential import (
            ExponentialLayer as TensorlyExponential,
        )
        from spflow.torch.structure.general.layer.leaf.exponential import ExponentialLayer as TorchExponential

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyExponential.accepts(signatures)
        elif backend == "pytorch":
            return TorchExponential.accepts(signatures)
        else:
            raise NotImplementedError("Exponential is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]):
        from spflow.base.structure.general.layer.leaf.exponential import (
            ExponentialLayer as TensorlyExponential,
        )
        from spflow.torch.structure.general.layer.leaf.exponential import ExponentialLayer as TorchExponential

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyExponential.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchExponential.from_signatures(signatures)
        else:
            raise NotImplementedError("Exponential is not implemented for this backend")
