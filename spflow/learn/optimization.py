import torch

from spflow import log_likelihood
from spflow.modules.module import Module


def gradient_descent(
    model: Module,
    dataloader: torch.utils.data.DataLoader,
    epochs: int = -1,
    verbose: bool = False,
    optimizer: torch.optim.Optimizer = None,
    lr: float = 1e-3,
):
    model.train()

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for (data,) in dataloader:
            # Reset gradients
            optimizer.zero_grad()

            # Compute negative log likelihood
            nll = -1 * log_likelihood(model, data).sum()

            # Compute gradients
            nll.backward()

            # Update weights
            optimizer.step()

        if verbose:
            print(f"Epoch {epoch + 1}: Loss {nll.item()}")
