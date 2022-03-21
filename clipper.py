import torch as th
import torch.nn.functional as F

LOWER_BOUND = th.tensor(1e-1)
DEFAULT_DEVICE = th.device("cpu")


class DistributionClipper(object):
    """
    Clip distribution parameters.
    """

    def __init__(self, device=DEFAULT_DEVICE, lower_bound=1e-5):
        """
        Args:
            device (th.device): Torch target device (defaults to CPU).
            lower_bound (float): Lower bound for distribution parameters with a condition of ">= 0".
        """
        self.lower_bound = th.tensor(lower_bound).to(device)

    def to(self, device) -> "DistributionClipper":
        """
        Move the clipper to the specified device.
        Returns a copy of the current object.

        Args:
            device (th.device): Target device to which the clipper should be moved.

        Returns:
            DistributionClipper: Copy of this distribution clipper on the target device.
        """
        return DistributionClipper(device=device, lower_bound=self.lower_bound)

    def __call__(self, module):
        """
        Clip the distribution parameters of the given leaf module.

        Args:
            module (Leaf): Module to clip.
        """
        if hasattr(module, "stds"):
            param = module.stds.data
            param.clamp_(self.lower_bound)

        if hasattr(module, "concentration0"):
            param = module.concentration0.data
            param.clamp_(self.lower_bound)

        if hasattr(module, "concentration1"):
            param = module.concentration1.data
            param.clamp_(self.lower_bound)

        if hasattr(module, "concentration"):
            param = module.concentration.data
            param.clamp_(self.lower_bound)

        if hasattr(module, "rate"):
            param = module.rate.data
            param.clamp_(self.lower_bound)

        if hasattr(module, "df"):
            param = module.df.data
            param.clamp_(self.lower_bound)

        if hasattr(module, "triangular"):
            param = module.triangular.data
            # Only clamp diagonal values for each
            mult_times_ndist = param.shape[0]
            for m in range(mult_times_ndist):
                param[m, :, :].diagonal().clamp_(self.lower_bound)
