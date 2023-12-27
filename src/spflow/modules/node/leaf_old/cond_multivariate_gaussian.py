from typing import List, Optional, Callable

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class CondMultivariateGaussian:  # ToDo: backend über T.getBackend() abfragen
    def __new__(cls, scope: Scope, cond_f: Optional[Callable] = None):
        from spflow.base.structure.general.node.leaf.cond_multivariate_gaussian import (
            CondMultivariateGaussian as TensorlyCondMultivariateGaussian,
        )
        from spflow.torch.structure.general.node.leaf.cond_multivariate_gaussian import (
            CondMultivariateGaussian as TorchCondMultivariateGaussian,
        )

        """TODO"""
        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyCondMultivariateGaussian(scope=scope, cond_f=cond_f)
        elif backend == "pytorch":
            return TorchCondMultivariateGaussian(scope=scope, cond_f=cond_f)
        else:
            raise NotImplementedError("CondMultivariateGaussian is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        from spflow.base.structure.general.node.leaf.cond_multivariate_gaussian import (
            CondMultivariateGaussian as TensorlyCondMultivariateGaussian,
        )
        from spflow.torch.structure.general.node.leaf.cond_multivariate_gaussian import (
            CondMultivariateGaussian as TorchCondMultivariateGaussian,
        )

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyCondMultivariateGaussian.accepts(signatures)
        elif backend == "pytorch":
            return TorchCondMultivariateGaussian.accepts(signatures)
        else:
            raise NotImplementedError("CondMultivariateGaussian is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]):
        from spflow.base.structure.general.node.leaf.cond_multivariate_gaussian import (
            CondMultivariateGaussian as TensorlyCondMultivariateGaussian,
        )
        from spflow.torch.structure.general.node.leaf.cond_multivariate_gaussian import (
            CondMultivariateGaussian as TorchCondMultivariateGaussian,
        )

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyCondMultivariateGaussian.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchCondMultivariateGaussian.from_signatures(signatures)
        else:
            raise NotImplementedError("CondMultivariateGaussian is not implemented for this backend")
