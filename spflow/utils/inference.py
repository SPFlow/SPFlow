import torch
from torch import Tensor


def log_posterior(log_likelihood: Tensor, log_prior: Tensor) -> Tensor:
    """Compute log posterior as sum of log likelihood and log prior.

    Args:
        log_likelihood: Log likelihood tensor.
        log_prior: Log prior tensor.

    Returns:
        Tensor: Log posterior tensor.
    """

    # posterior = likelihood * prior / marginal
    # in log space:
    #
    # logp(y | x) = logp(x, y) - logp(x)
    #             = logp(x | y) + logp(y) - logp(x)
    #             = logp(x | y) + logp(y) - logsumexp(logp(x,y), dim=y)

    ll = log_likelihood
    ll_y = log_prior
    ll_y = ll_y.view(1, -1)

    ll_x_and_y = ll + ll_y
    ll_x = torch.logsumexp(ll_x_and_y, dim=1, keepdim=True)
    ll_y_given_x = ll_x_and_y - ll_x

    return ll_y_given_x
