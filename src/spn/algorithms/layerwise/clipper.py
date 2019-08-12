import torch
import torch.nn.functional as F

LOWER_BOUND = torch.tensor(1e-1)


class DistributionClipper(object):
    """
    Clip distribution parameters.
    """

    def __init__(self, device, lower_bound=1e-5):
        """
        Args:
            device: Torch device.
        """
        self.lower_bound = torch.tensor(lower_bound).to(device)

    def __call__(self, module):
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
