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

from distributions import RatNormal
from cspn import CSPN, CspnConfig

from train_mnist import one_hot, time_delta_now, count_params, get_mnist_loaders, ensure_dir, set_seed


def make_spn(S, I, R, D, dropout, device) -> CSPN:
    """Construct the RatSpn"""

    # Setup RatSpnConfig
    config = CspnConfig()
    config.F = 28 * 14
    config.R = R
    config.D = D
    config.I = I
    config.S = S
    config.C = 10
    config.dropout = dropout
    config.leaf_base_class = RatNormal
    config.leaf_base_kwargs = {}

    # Construct RatSpn from config
    model = CSPN(config, [28, 14])

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
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_loader, test_loader = get_mnist_loaders(use_cuda, batch_size=batch_size, device=device)

    log_interval = 100

    lmbda = 1.0

    for epoch in range(n_epochs):
        if epoch > 20:
            # lmbda = lmbda_0 + lmbda_rel * (0.95 ** (epoch - 20))
            lmbda = 0.5
        t_start = time.time()
        running_loss = 0.0
        running_loss_ce = 0.0
        running_loss_nll = 0.0
        cond = None
        target = None
        for batch_index, (data, target) in enumerate(train_loader):
            # Send data to correct device
            image, target = data.to(device), target.to(device)
            cond = image[:, :, :, :14].to(device)
            data = image[:, :, :, 14:].to(device)
            data = data.reshape(data.shape[0], -1)
            # samples = model.sample(cond[:30], class_index=target[:30].tolist())

            # Reset gradients
            optimizer.zero_grad()

            # Inference
            output = model(data, cond)

            # Compute loss
            loss_ce = loss_fn(output, target)
            loss_nll = -output.sum() / (data.shape[0] * 28 ** 2)
            loss = (1 - lmbda) * loss_nll + lmbda * loss_ce

            # Backprop
            loss.backward()
            optimizer.step()
            # scheduler.step()

            # Log stuff
            running_loss += loss.item()
            running_loss_ce += loss_ce.item()
            running_loss_nll += loss_nll.item()
            if batch_index % log_interval == (log_interval - 1):
                pred = output.argmax(1).eq(target).sum().cpu().numpy() / data.shape[0] * 100
                print(
                    "Train Epoch: {} [{: >5}/{: <5} ({:.0f}%)]\tLoss_ce: {:.6f}\tLoss_nll: {:.6f}\tAccuracy: {:.0f}%".format(
                        epoch,
                        batch_index * len(data),
                        60000,
                        100.0 * batch_index / len(train_loader),
                        running_loss_ce / log_interval,
                        running_loss_nll / log_interval,
                        pred,
                    ),
                    end="\r",
                )
                running_loss = 0.0
                running_loss_ce = 0.0
                running_loss_nll = 0.0

        with torch.no_grad():
            set_seed(0)
            # samples = model.sample(cond, n=30)
            sample_cond = cond[:30]
            sample_target = target[:30]
            samples = model.sample(sample_cond, class_index=sample_target.tolist())
            full_img = torch.cat((sample_cond, samples.view(-1, 1, 28, 14)), dim=3)
            save_samples(full_img, iteration=epoch)

        t_delta = time_delta_now(t_start)
        print("Train Epoch: {} took {}".format(epoch, t_delta))
        if epoch % 5 == 4:
            print("Evaluating model ...")
            evaluate_model(model, device, train_loader, "Train")
            evaluate_model(model, device, test_loader, "Test")


def evaluate_model(model: torch.nn.Module, device, loader, tag) -> float:
    """
    Description for method evaluate_model.

    Args:
        model (nn.Module): PyTorch module.
        device: Execution device.
        loader: Data loader.
        tag (str): Tag for information.

    Returns:
        float: Tuple of loss and accuracy.
    """
    model.eval()
    loss_ce = 0
    loss_nll = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for data, target in loader:
            image, target = data.to(device), target.to(device)
            cond = image[:, :, :, :14].to(device)
            data = image[:, :, :, 14:].to(device)
            data = data.reshape(data.shape[0], -1)
            output = model(data, cond)
            loss_ce += criterion(output, target).item()  # sum up batch loss
            loss_nll += -output.sum()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()

    loss_ce /= len(loader.dataset)
    loss_nll /= len(loader.dataset) + 28 ** 2
    accuracy = 100.0 * correct / len(loader.dataset)

    print(
        "{} set: Average loss_ce: {:.4f} Average loss_nll: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            tag, loss_ce, loss_nll, correct, len(loader.dataset), accuracy
        )
    )


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
    d = "results/samples/"
    ensure_dir(d)
    plot_samples(samples.view(-1, 1, 28, 28), path=os.path.join(d, f"mnist-{iteration:03}.png"))


if __name__ == "__main__":
    torch.cuda.benchmark = True
    run_torch(100, 256)
