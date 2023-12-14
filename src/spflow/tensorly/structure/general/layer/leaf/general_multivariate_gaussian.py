from typing import List, Optional, Union

import tensorly as tl
from spflow.tensorly.utils.helper_functions import T
from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class MultivariateGaussianLayer:
    def __new__(cls, scope: Union[Scope, List[Scope]],
        mean: Optional[Union[List[float], List[List[float]], T]] = None,
        cov: Optional[
            Union[
                List[List[float]],
                List[List[List[float]]],
                T
            ]
        ] = None,
        n_nodes: int = 1,
        **kwargs):
        from spflow.base.structure.general.layer.leaf.multivariate_gaussian import MultivariateGaussianLayer as TensorlyMultivariateGaussian
        from spflow.torch.structure.general.layer.leaf.multivariate_gaussian import MultivariateGaussianLayer as TorchMultivariateGaussian
        """TODO"""
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyMultivariateGaussian(scope=scope, mean=mean, cov=cov, n_nodes=n_nodes, **kwargs)
        elif backend == "pytorch":
            return TorchMultivariateGaussian(scope=scope, mean=mean, cov=cov, n_nodes=n_nodes, **kwargs)
        else:
            raise NotImplementedError("MultivariateGaussian is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        from spflow.base.structure.general.layer.leaf.multivariate_gaussian import MultivariateGaussianLayer as TensorlyMultivariateGaussian
        from spflow.torch.structure.general.layer.leaf.multivariate_gaussian import MultivariateGaussianLayer as TorchMultivariateGaussian
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyMultivariateGaussian.accepts(signatures)
        elif backend == "pytorch":
            return TorchMultivariateGaussian.accepts(signatures)
        else:
            raise NotImplementedError("MultivariateGaussian is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]):
        from spflow.base.structure.general.layer.leaf.multivariate_gaussian import MultivariateGaussianLayer as TensorlyMultivariateGaussian
        from spflow.torch.structure.general.layer.leaf.multivariate_gaussian import MultivariateGaussianLayer as TorchMultivariateGaussian
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyMultivariateGaussian.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchMultivariateGaussian.from_signatures(signatures)
        else:
            raise NotImplementedError("MultivariateGaussian is not implemented for this backend")
