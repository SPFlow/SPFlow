from typing import List, Union

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope
from spflow.tensor.ops import Tensor


class UniformLayer:
    def __new__(
        cls,
        scope: Union[Scope, list[Scope]],
        start: Union[int, float, list[float], Tensor],
        end: Union[int, float, list[float], Tensor],
        support_outside: Union[bool, list[bool], Tensor] = True,
        n_nodes: int = 1,
        **kwargs,
    ):
        from spflow.base.structure.general.layer.leaf.uniform import UniformLayer as TensorlyUniform
        from spflow.torch.structure.general.layer.leaf.uniform import UniformLayer as TorchUniform

        """TODO"""
        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyUniform(
                scope=scope, start=start, end=end, support_outside=support_outside, n_nodes=n_nodes, **kwargs
            )
        elif backend == "pytorch":
            return TorchUniform(
                scope=scope, start=start, end=end, support_outside=support_outside, n_nodes=n_nodes, **kwargs
            )
        else:
            raise NotImplementedError("Uniform is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        from spflow.base.structure.general.layer.leaf.uniform import UniformLayer as TensorlyUniform
        from spflow.torch.structure.general.layer.leaf.uniform import UniformLayer as TorchUniform

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyUniform.accepts(signatures)
        elif backend == "pytorch":
            return TorchUniform.accepts(signatures)
        else:
            raise NotImplementedError("Uniform is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]):
        from spflow.base.structure.general.layer.leaf.uniform import UniformLayer as TensorlyUniform
        from spflow.torch.structure.general.layer.leaf.uniform import UniformLayer as TorchUniform

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyUniform.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchUniform.from_signatures(signatures)
        else:
            raise NotImplementedError("Uniform is not implemented for this backend")
