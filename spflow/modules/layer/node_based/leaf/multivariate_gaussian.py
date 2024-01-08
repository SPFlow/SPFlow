from typing import List, Optional, Union

import tensorly as tl
from spflow.tensor.ops import Tensor
from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class MultivariateGaussianLayer:
    def __new__(
        cls,
        scope: Union[Scope, list[Scope]],
        mean: Optional[Union[list[float], list[list[float]], Tensor]] = None,
        cov: Optional[Union[list[list[float]], list[list[list[float]]], Tensor]] = None,
        n_nodes: int = 1,
        **kwargs,
    ):
        from spflow.base.structure.general.layer.leaf.multivariate_gaussian import (
            MultivariateGaussianLayer as TensorlyMultivariateGaussian,
        )
        from spflow.torch.structure.general.layer.leaf.multivariate_gaussian import (
            MultivariateGaussianLayer as TorchMultivariateGaussian,
        )

        """TODO"""
        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyMultivariateGaussian(scope=scope, mean=mean, cov=cov, n_nodes=n_nodes, **kwargs)
        elif backend == "pytorch":
            return TorchMultivariateGaussian(scope=scope, mean=mean, cov=cov, n_nodes=n_nodes, **kwargs)
        else:
            raise NotImplementedError("MultivariateGaussian is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        from spflow.base.structure.general.layer.leaf.multivariate_gaussian import (
            MultivariateGaussianLayer as TensorlyMultivariateGaussian,
        )
        from spflow.torch.structure.general.layer.leaf.multivariate_gaussian import (
            MultivariateGaussianLayer as TorchMultivariateGaussian,
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
        from spflow.base.structure.general.layer.leaf.multivariate_gaussian import (
            MultivariateGaussianLayer as TensorlyMultivariateGaussian,
        )
        from spflow.torch.structure.general.layer.leaf.multivariate_gaussian import (
            MultivariateGaussianLayer as TorchMultivariateGaussian,
        )

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyMultivariateGaussian.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchMultivariateGaussian.from_signatures(signatures)
        else:
            raise NotImplementedError("MultivariateGaussian is not implemented for this backend")
