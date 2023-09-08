from typing import List, Optional, Callable, Union

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class CondLogNormalLayer:  # ToDo: backend über tl.getBackend() abfragen
    def __new__(cls, scope: Union[Scope, List[Scope]],
        cond_f: Optional[Union[Callable, List[Callable]]] = None,
        n_nodes: int = 1,
        **kwargs):
        from spflow.base.structure.general.layers.leaves.parametric.cond_log_normal import CondLogNormalLayer as TensorlyCondLogNormal
        from spflow.torch.structure.general.layers.leaves.parametric.cond_log_normal import CondLogNormalLayer as TorchCondLogNormal
        """TODO"""
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondLogNormal(scope=scope, cond_f=cond_f, n_nodes=n_nodes, **kwargs)
        elif backend == "pytorch":
            return TorchCondLogNormal(scope=scope, cond_f=cond_f, n_nodes=n_nodes, **kwargs)
        else:
            raise NotImplementedError("CondLogNormal is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        from spflow.base.structure.general.layers.leaves.parametric.cond_log_normal import CondLogNormalLayer as TensorlyCondLogNormal
        from spflow.torch.structure.general.layers.leaves.parametric.cond_log_normal import CondLogNormalLayer as TorchCondLogNormal
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondLogNormal.accepts(signatures)
        elif backend == "pytorch":
            return TorchCondLogNormal.accepts(signatures)
        else:
            raise NotImplementedError("CondLogNormal is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]):
        from spflow.base.structure.general.layers.leaves.parametric.cond_log_normal import CondLogNormalLayer as TensorlyCondLogNormal
        from spflow.torch.structure.general.layers.leaves.parametric.cond_log_normal import CondLogNormalLayer as TorchCondLogNormal
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondLogNormal.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchCondLogNormal.from_signatures(signatures)
        else:
            raise NotImplementedError("CondLogNormal is not implemented for this backend")
