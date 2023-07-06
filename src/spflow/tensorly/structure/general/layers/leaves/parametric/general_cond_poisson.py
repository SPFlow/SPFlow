from typing import List, Optional, Callable, Union

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class CondPoissonLayer:  # ToDo: backend über tl.getBackend() abfragen
    def __new__(cls, scope: Union[Scope, List[Scope]],
        cond_f: Optional[Union[Callable, List[Callable]]] = None,
        n_nodes: int = 1,
        **kwargs):
        from spflow.tensorly.structure.general.layers.leaves import CondPoissonLayer as TensorlyCondPoisson
        from spflow.torch.structure.general.layers.leaves import CondPoissonLayer as TorchCondPoisson
        """TODO"""
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondPoisson(scope=scope, cond_f=cond_f, n_nodes=n_nodes, **kwargs)
        elif backend == "pytorch":
            return TorchCondPoisson(scope=scope, cond_f=cond_f, n_nodes=n_nodes, **kwargs)
        else:
            raise NotImplementedError("CondPoisson is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        from spflow.tensorly.structure.general.layers.leaves import CondPoissonLayer as TensorlyCondPoisson
        from spflow.torch.structure.general.layers.leaves import CondPoissonLayer as TorchCondPoisson
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondPoisson.accepts(signatures)
        elif backend == "pytorch":
            return TorchCondPoisson.accepts(signatures)
        else:
            raise NotImplementedError("CondPoisson is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]):
        from spflow.tensorly.structure.general.layers.leaves import CondPoissonLayer as TensorlyCondPoisson
        from spflow.torch.structure.general.layers.leaves import CondPoissonLayer as TorchCondPoisson
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondPoisson.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchCondPoisson.from_signatures(signatures)
        else:
            raise NotImplementedError("CondPoisson is not implemented for this backend")
