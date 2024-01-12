from spflow import log_likelihood
import torch


def evaluate_log_likelihood(node, data):
    lls = log_likelihood(node, data, check_support=True)
    assert lls.shape == (data.shape[0], 1)
    assert torch.isfinite(lls).all()
