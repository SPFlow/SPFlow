from torch import Tensor
from spflow.modules.layer.sum import SumLayer
from spflow.modules.module import Module


class Sum(SumLayer):
    def __init__(
            self,
            n_nodes: int,
            inputs: list[Module],
            weights: Tensor = None,
            **kwargs,
    ) -> None:
        super().__init__(n_nodes=n_nodes, inputs=inputs, weights=weights, **kwargs)
        assert self.event_shape[1] == self.event_shape[2] == 1, "SumNode event_shape must be (n, 1, 1)"