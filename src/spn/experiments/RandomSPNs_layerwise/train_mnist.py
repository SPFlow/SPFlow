import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np
from torch import nn
from torch.nn import functional as F
import time
import sys

import spn.algorithms.Inference as inference
import spn.io.Graphics as graphics


def one_hot(vector):
    result = np.zeros((vector.size, vector.max() + 1))
    result[np.arange(vector.size), vector] = 1
    return result


def time_delta_now(t_start: float) -> str:
    """
    Convert a timestamp into a human readable timestring.
    Args:
        t_start (float): Timestamp.

    Returns:
        Human readable timestring.
    """
    a = t_start
    b = time.time()  # current epoch time
    c = b - a  # seconds
    days = round(c // 86400)
    hours = round(c // 3600 % 24)
    minutes = round(c // 60 % 60)
    seconds = round(c % 60)
    millisecs = round(c % 1 * 1000)
    return f"{days} days, {hours} hours, {minutes} minutes, {seconds} seconds, {millisecs} milliseconds"


def count_params(model: torch.nn.Module) -> int:
    """
    Count the number of parameters in a model.

    Args:
        model (torch.nn.Module): PyTorch model.

    Returns:
        int: Number of learnable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_mnist_loaders(use_cuda, device, batch_size):
    """
    Get the MNIST pytorch data loader.
    
    Args:
        use_cuda: Use cuda flag.

    """
    kwargs = {"num_workers": 8, "pin_memory": True} if use_cuda else {}

    test_batch_size = batch_size

    transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # Train data loader
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("../data", train=True, download=True, transform=transformer),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    # Test data loader
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("../data", train=False, transform=transformer),
        batch_size=test_batch_size,
        shuffle=True,
        **kwargs,
    )
    return train_loader, test_loader


def run_torch(n_epochs=100, batch_size=256):
    """Run the torch code.

    Args:
        n_epochs (int, optional): Number of epochs.
        batch_size (int, optional): Batch size.
    """
    from spn.experiments.RandomSPNs_layerwise.rat_spn import RatSpnConstructor
    from torch import optim
    from torch import nn

    assert len(sys.argv) == 2, "Usage: train.mnist cuda/cpu"
    dev = sys.argv[1]

    rg = RatSpnConstructor(in_features=28 * 28, C=10, S=10, I=20, dropout=0.0)
    n_splits = 2
    for _ in range(0, n_splits):
        rg.random_split(2, 1)

    if dev == "cpu":
        device = torch.device("cpu")
        use_cuda = False
    else:
        device = torch.device("cuda:0")
        use_cuda = True
        torch.cuda.benchmark = True

    print("Using device:", device)

    model = rg.build().to(device)
    model.train()
    print(model)
    print(f"Layer 0: {count_params(model.region_spns[0]._leaf) * n_splits}")
    for i in range(1, len(model.region_spns[0]._inner_layers) + 1):
        print(f"Layer {i}: {count_params(model.region_spns[0]._inner_layers[i - 1]) * n_splits}")
    print("Number of pytorch parameters: ", count_params(model))

    # Define optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_loader, test_loader = get_mnist_loaders(use_cuda, batch_size=batch_size, device=device)

    log_interval = 100

    for epoch in range(n_epochs):
        t_start = time.time()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            # Send data to correct device
            data, target = data.to(device), target.to(device)
            data = data.view(data.shape[0], -1)

            # Reset gradients
            optimizer.zero_grad()

            # Inference
            output = model(data)

            # Compute loss
            loss = loss_fn(output, target)

            # Backprop
            loss.backward()
            optimizer.step()

            # Log stuff
            running_loss += loss.item()
            if batch_idx % log_interval == (log_interval - 1):
                pred = output.argmax(1).eq(target).sum().cpu().numpy() / data.shape[0] * 100
                print(
                    "Train Epoch: {} [{: >5}/{: <5} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.0f}%".format(
                        epoch,
                        batch_idx * len(data),
                        60000,
                        100.0 * batch_idx / len(train_loader),
                        running_loss / log_interval,
                        pred,
                    ),
                    end="\r",
                )
                running_loss = 0.0

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
    loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.shape[0], -1)
            output = model(data)
            loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()

    loss /= len(loader.dataset)
    accuracy = 100.0 * correct / len(loader.dataset)

    print(
        "{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            tag, loss, correct, len(loader.dataset), accuracy
        )
    )
    return (loss, accuracy)


if __name__ == "__main__":
    torch.cuda.benchmark = True
    run_torch(100, 100)
