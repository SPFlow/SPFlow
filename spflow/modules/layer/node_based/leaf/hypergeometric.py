from typing import List, Union
import numpy as np
import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class HypergeometricLayer:
    def __new__(
        cls,
        scope: Union[Scope, list[Scope]],
        N: Union[int, list[int], np.ndarray],
        M: Union[int, list[int], np.ndarray],
        n: Union[int, list[int], np.ndarray],
        n_nodes: int = 1,
        **kwargs,
    ):
        from spflow.base.structure.general.layer.leaf.hypergeometric import (
            HypergeometricLayer as TensorlyHypergeometric,
        )
        from spflow.torch.structure.general.layer.leaf.hypergeometric import (
            HypergeometricLayer as TorchHypergeometric,
        )

        """TODO"""
        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyHypergeometric(scope=scope, N=N, M=M, n=n, n_nodes=n_nodes, **kwargs)
        elif backend == "pytorch":
            return TorchHypergeometric(scope=scope, N=N, M=M, n=n, n_nodes=n_nodes, **kwargs)
        else:
            raise NotImplementedError("Hypergeometric is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        from spflow.base.structure.general.layer.leaf.hypergeometric import (
            HypergeometricLayer as TensorlyHypergeometric,
        )
        from spflow.torch.structure.general.layer.leaf.hypergeometric import (
            HypergeometricLayer as TorchHypergeometric,
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
        from spflow.base.structure.general.layer.leaf.hypergeometric import (
            HypergeometricLayer as TensorlyHypergeometric,
        )
        from spflow.torch.structure.general.layer.leaf.hypergeometric import (
            HypergeometricLayer as TorchHypergeometric,
        )

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyHypergeometric.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchHypergeometric.from_signatures(signatures)
        else:
            raise NotImplementedError("Hypergeometric is not implemented for this backend")