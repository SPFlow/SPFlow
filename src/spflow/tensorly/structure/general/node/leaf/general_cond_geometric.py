from typing import List, Optional, Callable

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class CondGeometric:  # ToDo: backend über tl.getBackend() abfragen
    def __new__(cls, scope: Scope, cond_f: Optional[Callable] = None):
        from spflow.base.structure.general.node.leaf.cond_geometric import CondGeometric as TensorlyCondGeometric
        from spflow.torch.structure.general.node.leaf.cond_geometric import CondGeometric as TorchCondGeometric
        """TODO"""
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondGeometric(scope=scope, cond_f=cond_f)
        elif backend == "pytorch":
            return TorchCondGeometric(scope=scope, cond_f=cond_f)
        else:
            raise NotImplementedError("CondGeometric is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        from spflow.base.structure.general.node.leaf.cond_geometric import CondGeometric as TensorlyCondGeometric
        from spflow.torch.structure.general.node.leaf.cond_geometric import CondGeometric as TorchCondGeometric
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondGeometric.accepts(signatures)
        elif backend == "pytorch":
            return TorchCondGeometric.accepts(signatures)
        else:
            raise NotImplementedError("CondGeometric is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]):
        from spflow.base.structure.general.node.leaf.cond_geometric import CondGeometric as TensorlyCondGeometric
        from spflow.torch.structure.general.node.leaf.cond_geometric import CondGeometric as TorchCondGeometric
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondGeometric.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchCondGeometric.from_signatures(signatures)
        else:
            raise NotImplementedError("CondGeometric is not implemented for this backend")
