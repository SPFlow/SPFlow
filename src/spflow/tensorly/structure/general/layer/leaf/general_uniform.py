from typing import List, Union

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope
from spflow.tensorly.utils.helper_functions import T

class UniformLayer:
    def __new__(cls, scope: Union[Scope, List[Scope]],
        start: Union[int, float, List[float], T],
        end: Union[int, float, List[float], T],
        support_outside: Union[bool, List[bool], T] = True,
        n_nodes: int = 1,
        **kwargs):
        from spflow.base.structure.general.layer.leaf.uniform import UniformLayer as TensorlyUniform
        from spflow.torch.structure.general.layer.leaf.uniform import UniformLayer as TorchUniform
        """TODO"""
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyUniform(scope=scope, start=start, end=end, support_outside=support_outside, n_nodes=n_nodes, **kwargs)
        elif backend == "pytorch":
            return TorchUniform(scope=scope, start=start, end=end, support_outside=support_outside, n_nodes=n_nodes, **kwargs)
        else:
            raise NotImplementedError("Uniform is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        from spflow.base.structure.general.layer.leaf.uniform import UniformLayer as TensorlyUniform
        from spflow.torch.structure.general.layer.leaf.uniform import UniformLayer as TorchUniform
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyUniform.accepts(signatures)
        elif backend == "pytorch":
            return TorchUniform.accepts(signatures)
        else:
            raise NotImplementedError("Uniform is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]):
        from spflow.base.structure.general.layer.leaf.uniform import UniformLayer as TensorlyUniform
        from spflow.torch.structure.general.layer.leaf.uniform import UniformLayer as TorchUniform
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyUniform.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchUniform.from_signatures(signatures)
        else:
            raise NotImplementedError("Uniform is not implemented for this backend")
