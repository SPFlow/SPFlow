from spflow.meta.data import Scope
from spflow.modules.base import Module


class AbstractWrapper(Module):
    def __init__(self, module: Module):
        super().__init__()
        self.module = module
        self.num_repetitions = module.num_repetitions
        self.scope = module.scope

    @property
    def out_features(self) -> int:
        """Returns the number of output features of the module."""
        return self.module.out_features

    @property
    def out_channels(self) -> int:
        """Returns the number of output channels of the module."""
        return self.module.out_channels

    @property
    def feature_to_scope(self) -> list[Scope]:
        """Returns the mapping from features to scopes."""
        return self.module.feature_to_scope

    @property
    def device(self):
        """
        Returns the device of the model. If the model parameters are on different devices,
        it returns the device of the first parameter. If the model has no parameters,
        it returns 'cpu' as the default device.
        """
        return next(iter(self.module.parameters())).device

    def extra_repr(self) -> str:
        return f"D={self.out_features}, C={self.out_channels}, R={self.num_repetitions}"
