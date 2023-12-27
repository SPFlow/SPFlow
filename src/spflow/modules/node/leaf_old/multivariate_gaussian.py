from typing import List, Optional, Union

import tensorly as tl
from spflow.tensor.ops import Tensor
from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class MultivariateGaussian:
    def __new__(
        cls,
        scope: Scope,
        mean: Optional[Union[list[float], Tensor]] = None,
        cov: Optional[Union[list[list[float]], Tensor]] = None,
    ):
        from spflow.base.structure.general.node.leaf.multivariate_gaussian import (
            MultivariateGaussian as TensorlyMultivariateGaussian,
        )
        from spflow.torch.structure.general.node.leaf.multivariate_gaussian import (
            MultivariateGaussian as TorchMultivariateGaussian,
        )

        """TODO"""
        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyMultivariateGaussian(scope=scope, mean=mean, cov=cov)
        elif backend == "pytorch":
            return TorchMultivariateGaussian(scope=scope, mean=mean, cov=cov)
        else:
            raise NotImplementedError("MultivariateGaussian is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        from spflow.base.structure.general.node.leaf.multivariate_gaussian import (
            MultivariateGaussian as TensorlyMultivariateGaussian,
        )
        from spflow.torch.structure.general.node.leaf.multivariate_gaussian import (
            MultivariateGaussian as TorchMultivariateGaussian,
        )

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyMultivariateGaussian.accepts(signatures)
        elif backend == "pytorch":
            return TorchMultivariateGaussian.accepts(signatures)
        else:
            raise NotImplementedError("MultivariateGaussian is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]):
        from spflow.base.structure.general.node.leaf.multivariate_gaussian import (
            MultivariateGaussian as TensorlyMultivariateGaussian,
        )
        from spflow.torch.structure.general.node.leaf.multivariate_gaussian import (
            MultivariateGaussian as TorchMultivariateGaussian,
        )

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyMultivariateGaussian.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchMultivariateGaussian.from_signatures(signatures)
        else:
            raise NotImplementedError("MultivariateGaussian is not implemented for this backend")
