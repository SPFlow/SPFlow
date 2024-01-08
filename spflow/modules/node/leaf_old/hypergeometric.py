from typing import List

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class Hypergeometric:
    def __new__(cls, scope: Scope, N: int, M: int, n: int):
        from spflow.base.structure.general.node.leaf.hypergeometric import (
            Hypergeometric as TensorlyHypergeometric,
        )
        from spflow.torch.structure.general.node.leaf.hypergeometric import (
            Hypergeometric as TorchHypergeometric,
        )

        """TODO"""
        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyHypergeometric(scope=scope, N=N, M=M, n=n)
        elif backend == "pytorch":
            return TorchHypergeometric(scope=scope, N=N, M=M, n=n)
        else:
            raise NotImplementedError("Hypergeometric is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        from spflow.base.structure.general.node.leaf.hypergeometric import (
            Hypergeometric as TensorlyHypergeometric,
        )
        from spflow.torch.structure.general.node.leaf.hypergeometric import (
            Hypergeometric as TorchHypergeometric,
        )

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyHypergeometric.accepts(signatures)
        elif backend == "pytorch":
            return TorchHypergeometric.accepts(signatures)
        else:
            raise NotImplementedError("Hypergeometric is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]):
        from spflow.base.structure.general.node.leaf.hypergeometric import (
            Hypergeometric as TensorlyHypergeometric,
        )
        from spflow.torch.structure.general.node.leaf.hypergeometric import (
            Hypergeometric as TorchHypergeometric,
        )

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyHypergeometric.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchHypergeometric.from_signatures(signatures)
        else:
            raise NotImplementedError("Hypergeometric is not implemented for this backend")