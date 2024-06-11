from spflow.modules.layer.product import ProductLayer
from spflow.modules.module import Module


class Product(ProductLayer):
    def __init__(
            self,
            inputs: list[Module],
            **kwargs,
    ) -> None:
        super().__init__(inputs=inputs, **kwargs)
        assert self.event_shape[1] == 1 or self.event_shape[2] == 1, "ProductNode event_shape must be (n, m, 1) or (n, 1, m)"