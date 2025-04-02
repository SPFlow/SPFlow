from torch import Tensor

from spflow import distributions as D
from spflow.meta.data import Scope
from spflow.modules.leaf.leaf_module import LeafModule
from spflow.utils.leaf import parse_leaf_args


class Uniform(LeafModule):
    def __init__(
        self,
        scope: Scope,
        out_channels: int = None,
        num_repetitions: int = None,
        start: Tensor = None,
        end: Tensor = None,
        support_outside: bool = True,
    ):
        """
        Initialize a Uniform distribution leaf module.

        Args:
            scope (Scope): The scope of the distribution.
            out_channels (int, optional): The number of output channels. If None, it is determined by the parameter tensors.
            start (Tensor, optional): The start of the interval.
            end (Tensor, optional): The end of the interval.
            support_outside (bool, optional): Whether to support values outside the interval. Default is True.
        """
        event_shape = parse_leaf_args(scope=scope, out_channels=out_channels, params=[start, end], num_repetitions=num_repetitions)
        super().__init__(scope, out_channels=event_shape[1])
        self.distribution = D.Uniform(start, end, support_outside, event_shape=event_shape)

    @property
    def device(self):
        """
        Get the device of the module. Necessary hack since this module has no parameters.

        Returns:
            torch.device: The device on which the module's buffers are located.
        """
        return next(iter(self.buffers())).device
