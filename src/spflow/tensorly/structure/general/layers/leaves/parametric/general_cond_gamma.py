from typing import List, Optional, Callable, Union

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class CondGammaLayer:  # ToDo: backend über tl.getBackend() abfragen
    def __new__(cls, scope: Union[Scope, List[Scope]],
        cond_f: Optional[Union[Callable, List[Callable]]] = None,
        n_nodes: int = 1,
        **kwargs):
        from spflow.tensorly.structure.general.layers.leaves import CondGammaLayer as TensorlyCondGamma
        from spflow.torch.structure.general.layers.leaves import CondGammaLayer as TorchCondGamma
        """TODO"""
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondGamma(scope=scope, cond_f=cond_f, n_nodes=n_nodes, **kwargs)
        elif backend == "pytorch":
            return TorchCondGamma(scope=scope, cond_f=cond_f, n_nodes=n_nodes, **kwargs)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        from spflow.tensorly.structure.general.layers.leaves import CondGammaLayer as TensorlyCondGamma
        from spflow.torch.structure.general.layers.leaves import CondGammaLayer as TorchCondGamma
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondGamma.accepts(signatures)
        elif backend == "pytorch":
            return TorchCondGamma.accepts(signatures)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]):
        from spflow.tensorly.structure.general.layers.leaves import CondGammaLayer as TensorlyCondGamma
        from spflow.torch.structure.general.layers.leaves import CondGammaLayer as TorchCondGamma
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondGamma.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchCondGamma.from_signatures(signatures)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")
