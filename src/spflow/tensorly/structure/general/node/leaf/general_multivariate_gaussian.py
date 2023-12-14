from typing import List, Optional, Union

import tensorly as tl
from spflow.tensorly.utils.helper_functions import T
from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class MultivariateGaussian:
    def __new__(cls, scope: Scope, mean: Optional[Union[List[float], T]] = None,cov: Optional[Union[List[List[float]], T]] = None):
        from spflow.base.structure.general.node.leaf.multivariate_gaussian import MultivariateGaussian as TensorlyMultivariateGaussian
        from spflow.torch.structure.general.node.leaf.multivariate_gaussian import MultivariateGaussian as TorchMultivariateGaussian
        """TODO"""
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyMultivariateGaussian(scope=scope, mean=mean, cov=cov)
        elif backend == "pytorch":
            return TorchMultivariateGaussian(scope=scope, mean=mean, cov=cov)
        else:
            raise NotImplementedError("MultivariateGaussian is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        from spflow.base.structure.general.node.leaf.multivariate_gaussian import MultivariateGaussian as TensorlyMultivariateGaussian
        from spflow.torch.structure.general.node.leaf.multivariate_gaussian import MultivariateGaussian as TorchMultivariateGaussian
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyMultivariateGaussian.accepts(signatures)
        elif backend == "pytorch":
            return TorchMultivariateGaussian.accepts(signatures)
        else:
            raise NotImplementedError("MultivariateGaussian is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]):
        from spflow.base.structure.general.node.leaf.multivariate_gaussian import MultivariateGaussian as TensorlyMultivariateGaussian
        from spflow.torch.structure.general.node.leaf.multivariate_gaussian import MultivariateGaussian as TorchMultivariateGaussian
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyMultivariateGaussian.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchMultivariateGaussian.from_signatures(signatures)
        else:
            raise NotImplementedError("MultivariateGaussian is not implemented for this backend")
