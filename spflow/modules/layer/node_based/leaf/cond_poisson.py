from typing import List, Optional, Callable, Union

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class CondPoissonLayer:  # ToDo: backend über T.getBackend() abfragen
    def __new__(
        cls,
        scope: Union[Scope, list[Scope]],
        cond_f: Optional[Union[Callable, list[Callable]]] = None,
        n_nodes: int = 1,
        **kwargs,
    ):
        from spflow.base.structure.general.layer.leaf.cond_poisson import (
            CondPoissonLayer as TensorlyCondPoisson,
        )
        from spflow.torch.structure.general.layer.leaf.cond_poisson import (
            CondPoissonLayer as TorchCondPoisson,
        )

        """TODO"""
        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyCondPoisson(scope=scope, cond_f=cond_f, n_nodes=n_nodes, **kwargs)
        elif backend == "pytorch":
            return TorchCondPoisson(scope=scope, cond_f=cond_f, n_nodes=n_nodes, **kwargs)
        else:
            raise NotImplementedError("CondPoisson is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        from spflow.base.structure.general.layer.leaf.cond_poisson import (
            CondPoissonLayer as TensorlyCondPoisson,
        )
        from spflow.torch.structure.general.layer.leaf.cond_poisson import (
            CondPoissonLayer as TorchCondPoisson,
        )

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyCondPoisson.accepts(signatures)
        elif backend == "pytorch":
            return TorchCondPoisson.accepts(signatures)
        else:
            raise NotImplementedError("CondPoisson is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]):
        from spflow.base.structure.general.layer.leaf.cond_poisson import (
            CondPoissonLayer as TensorlyCondPoisson,
        )
        from spflow.torch.structure.general.layer.leaf.cond_poisson import (
            CondPoissonLayer as TorchCondPoisson,
        )

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyCondPoisson.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchCondPoisson.from_signatures(signatures)
        else:
            raise NotImplementedError("CondPoisson is not implemented for this backend")
