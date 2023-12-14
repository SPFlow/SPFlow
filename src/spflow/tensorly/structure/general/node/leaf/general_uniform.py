from typing import List

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class Uniform:
    def __new__(cls, scope: Scope, start: float, end: float, support_outside: bool = True):
        from spflow.base.structure.general.node.leaf.uniform import Uniform as TensorlyUniform
        from spflow.torch.structure.general.node.leaf.uniform import Uniform as TorchUniform
        """TODO"""
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyUniform(scope=scope, start=start, end=end, support_outside=support_outside)
        elif backend == "pytorch":
            return TorchUniform(scope=scope, start=start, end=end, support_outside=support_outside)
        else:
            raise NotImplementedError("Uniform is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        from spflow.base.structure.general.node.leaf.uniform import Uniform as TensorlyUniform
        from spflow.torch.structure.general.node.leaf.uniform import Uniform as TorchUniform
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyUniform.accepts(signatures)
        elif backend == "pytorch":
            return TorchUniform.accepts(signatures)
        else:
            raise NotImplementedError("Uniform is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]):
        from spflow.base.structure.general.node.leaf.uniform import Uniform as TensorlyUniform
        from spflow.torch.structure.general.node.leaf.uniform import Uniform as TorchUniform
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyUniform.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchUniform.from_signatures(signatures)
        else:
            raise NotImplementedError("Uniform is not implemented for this backend")
