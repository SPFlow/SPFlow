from typing import List, Optional, Callable

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class CondMultivariateGaussian:  # ToDo: backend über tl.getBackend() abfragen
    def __new__(cls, scope: Scope, cond_f: Optional[Callable] = None):
        from spflow.tensorly.structure.general.nodes.leaves import \
            CondMultivariateGaussian as TensorlyCondMultivariateGaussian
        from spflow.torch.structure.general.nodes.leaves import \
            CondMultivariateGaussian as TorchCondMultivariateGaussian
        """TODO"""
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondMultivariateGaussian(scope=scope, cond_f=cond_f)
        elif backend == "pytorch":
            return TorchCondMultivariateGaussian(scope=scope, cond_f=cond_f)
        else:
            raise NotImplementedError("CondMultivariateGaussian is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        from spflow.tensorly.structure.general.nodes.leaves import \
            CondMultivariateGaussian as TensorlyCondMultivariateGaussian
        from spflow.torch.structure.general.nodes.leaves import \
            CondMultivariateGaussian as TorchCondMultivariateGaussian
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondMultivariateGaussian.accepts(signatures)
        elif backend == "pytorch":
            return TorchCondMultivariateGaussian.accepts(signatures)
        else:
            raise NotImplementedError("CondMultivariateGaussian is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]):
        from spflow.tensorly.structure.general.nodes.leaves import \
            CondMultivariateGaussian as TensorlyCondMultivariateGaussian
        from spflow.torch.structure.general.nodes.leaves import \
            CondMultivariateGaussian as TorchCondMultivariateGaussian
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondMultivariateGaussian.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchCondMultivariateGaussian.from_signatures(signatures)
        else:
            raise NotImplementedError("CondMultivariateGaussian is not implemented for this backend")
