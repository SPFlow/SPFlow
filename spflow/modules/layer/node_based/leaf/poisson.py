from typing import List, Union

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope
from spflow.tensor.ops import Tensor


class PoissonLayer:  # ToDo: backend über T.getBackend() abfragen
    def __new__(
        cls,
        scope: Union[Scope, list[Scope]],
        l: Union[int, float, list[float], Tensor] = 1.0,
        n_nodes: int = 1,
        **kwargs,
    ):
        from spflow.base.structure.general.layer.leaf.poisson import PoissonLayer as TensorlyPoisson
        from spflow.torch.structure.general.layer.leaf.poisson import PoissonLayer as TorchPoisson

        """TODO"""
        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyPoisson(scope=scope, l=l, n_nodes=n_nodes, **kwargs)
        elif backend == "pytorch":
            return TorchPoisson(scope=scope, l=l, n_nodes=n_nodes, **kwargs)
        else:
            raise NotImplementedError("Poisson is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        from spflow.base.structure.general.layer.leaf.poisson import PoissonLayer as TensorlyPoisson
        from spflow.torch.structure.general.layer.leaf.poisson import PoissonLayer as TorchPoisson

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyPoisson.accepts(signatures)
        elif backend == "pytorch":
            return TorchPoisson.accepts(signatures)
        else:
            raise NotImplementedError("Poisson is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]):
        from spflow.base.structure.general.layer.leaf.poisson import PoissonLayer as TensorlyPoisson
        from spflow.torch.structure.general.layer.leaf.poisson import PoissonLayer as TorchPoisson

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyPoisson.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchPoisson.from_signatures(signatures)
        else:
            raise NotImplementedError("Poisson is not implemented for this backend")
