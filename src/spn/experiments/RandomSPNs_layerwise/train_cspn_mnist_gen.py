import os
import random
import sys
import time

import imageio
import numpy as np
import skimage
import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms

from spn.experiments.RandomSPNs_layerwise.distributions import RatNormal
from spn.experiments.RandomSPNs_layerwise.cspn import CSPN, CspnConfig

from train_mnist import one_hot, time_delta_now, count_params, get_mnist_loaders, ensure_dir, set_seed


def make_spn(S, I, R, D, dropout, device) -> CSPN:
    """Construct the RatSpn"""

    # Setup RatSpnConfig
    config = CspnConfig()
    config.F = 28 * 28
    config.R = R
    config.D = D
    config.I = I
    config.S = S
    config.C = 10
    config.dropout = dropout
    config.leaf_base_class = RatNormal
    config.leaf_base_kwargs = {}

    # Construct RatSpn from config
    model = CSPN(config, [10])

    model = model.to(device)
    model.train()

    print("Using device:", device)
    return model


def run_torch(n_epochs=100, batch_size=256):
    """Run the torch code.

    Args:
        n_epochs (int, optional): Number of epochs.
        batch_size (int, optional): Batch size.
    """
    from torch import optim
    from torch import nn

    assert len(sys.argv) == 2, "Usage: train.mnist cuda/cpu"
    dev = sys.argv[1]

    if dev == "cpu":
        device = torch.device("cpu")
        use_cuda = False
    else:
        device = torch.device("cuda:0")
        use_cuda = True
        torch.cuda.benchmark = True

    model: CSPN = make_spn(S=10, I=10, D=3, R=5, device=dev, dropout=0.0)

    model.train()
    print(model)
    print("Number of pytorch parameters: ", count_params(model))

    # Define optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_loader, test_loader = get_mnist_loaders(use_cuda, batch_size=batch_size, device=device)

    log_interval = 100

    # lmbda = 1.0

    for epoch in range(n_epochs):
        t_start = time.time()
        running_loss = 0.0
        label = None
        for batch_index, (image, label) in enumerate(train_loader):
            # Send data to correct device
            image, label = image.to(device), label.to(device)
            # TODO make label one-hot
            image = image.view(image.shape[0], -1)

            # Reset gradients
            optimizer.zero_grad()

            # Inference
            output = model(x=image, condition=label)

            # Compute loss
            loss = loss_fn(output, image)

            # Backprop
            loss.backward()
            optimizer.step()
            # scheduler.step()

            # Log stuff
            running_loss += loss.item()
            if batch_index % log_interval == (log_interval - 1):
                print(
                    "Train Epoch: {} [{: >5}/{: <5} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_index * len(image),
                        60000,
                        100.0 * batch_index / len(train_loader),
                        running_loss / log_interval,
                    ),
                    end="\r",
                )
                running_loss = 0.0

        with torch.no_grad():
            set_seed(0)
            # samples = model.sample(cond, n=30)
            sample_cond = label[:30]
            samples = model.sample(condition=sample_cond).view(-1, 1, 28, 14)
            save_samples(samples, iteration=epoch)

        t_delta = time_delta_now(t_start)
        print("Train Epoch: {} took {}".format(epoch, t_delta))


def plot_samples(x: torch.Tensor, path):
    """
    Plot a single sample witht the target and prediction in the title.

    Args:
        x (torch.Tensor): Batch of input images. Has to be shape: [N, C, H, W].
    """
    # Normalize in valid range
    for i in range(x.shape[0]):
        x[i, :] = (x[i, :] - x[i, :].min()) / (x[i, :].max() - x[i, :].min())

    tensors = torchvision.utils.make_grid(x, nrow=10, padding=1).cpu()
    arr = tensors.permute(1, 2, 0).numpy()
    arr = skimage.img_as_ubyte(arr)
    imageio.imwrite(path, arr)


def save_samples(samples, iteration: int):
    d = "results_10012022_cspn/samples/"
    ensure_dir(d)
    plot_samples(samples.view(-1, 1, 28, 28), path=os.path.join(d, f"mnist-{iteration:03}.png"))


if __name__ == "__main__":
    torch.cuda.benchmark = True
    run_torch(100, 256)
