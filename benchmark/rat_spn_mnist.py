from torch.utils.data import DataLoader

from spflow.modules.leaves import Normal, Binomial, Bernoulli
from spflow.modules.rat.rat_spn import RatSPN

from spflow.meta.data import Scope

from spflow import log_likelihood, sample

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import os
import random
import time

import numpy as np
import torch
from torchvision import datasets, transforms



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


# def get_mnist_loaders(use_cuda, device, batch_size):
#     """
#     Get the MNIST pytorch data loader.
#
#     Args:
#         use_cuda: Use cuda flag.
#
#     """
#     kwargs = {"num_workers": 8, "pin_memory": True} if use_cuda else {}
#
#     test_batch_size = batch_size
#
#     transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
#     # Train data loader
#     train_loader = torch.utils.data.DataLoader(
#         datasets.MNIST("../data", train=True, download=True, transform=transformer),
#         batch_size=batch_size,
#         shuffle=True,
#         **kwargs,
#     )
#
#     # Test data loader
#     test_loader = torch.utils.data.DataLoader(
#         datasets.MNIST("../data", train=False, transform=transformer),
#         batch_size=test_batch_size,
#         shuffle=True,
#         **kwargs,
#     )
#     return train_loader, test_loader

def get_mnist_loaders(use_cuda, batch_size):
    """
    Get the MNIST PyTorch data loaders, filtered to only include selected classes.

    Args:
        use_cuda (bool): Use CUDA if available.
        device (torch.device): Target device.
        batch_size (int): Batch size.
        selected_classes (list): List of integer class labels to include.
    """
    #generator = torch.Generator()
    selected_classes = [0,1]#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    kwargs = {"num_workers": 8, "pin_memory": True} if use_cuda else {}
    #kwargs["generator"] = generator
    test_batch_size = batch_size

    transformer = transforms.Compose([
        transforms.ToTensor(),
    ])

    def filter_classes(dataset, classes):
        idx = np.isin(dataset.targets, classes)
        dataset.data = dataset.data[idx]
        dataset.targets = dataset.targets[idx]
        return dataset



    # Load and filter training data
    train_dataset = datasets.MNIST("../tests/data", train=True, download=True, transform=transformer)
    train_dataset = filter_classes(train_dataset, selected_classes)

    # Load and filter test data
    test_dataset = datasets.MNIST("../tests/data", train=False, transform=transformer)
    test_dataset = filter_classes(test_dataset, selected_classes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,**kwargs)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader


def make_spn(device) -> RatSPN:
    """Construct the RatSpn"""

    depth = 5
    n_region_nodes = 16
    num_leaves = 16
    num_repetitions = 10
    n_root_nodes = 1
    num_features = 28 ** 2


    random_variables = list(range(num_features))
    scope = Scope(random_variables)

    normal_layer = Normal(scope=scope, out_channels=num_leaves, num_repetitions=num_repetitions)

    model = RatSPN(
        leaf_modules=[normal_layer],
        n_root_nodes=n_root_nodes,
        n_region_nodes=n_region_nodes,
        num_repetitions=num_repetitions,
        depth=depth,
        outer_product=True,
        split_halves=False,
    )

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

    device = os.getenv("SPFLOW_TEST_DEVICE", "cpu")
    assert device == "cpu" or "cuda" in device, "SPFLOW_TEST_DEVICE must be 'cpu' or 'cuda' or 'cuda:<id>'"

    if device == "cuda":
        use_cuda = True
        torch.backends.cudnn.benchmark = True
    else:
        use_cuda = False
    device = torch.device(device)

    model = make_spn(device)

    model.train()
    print(model)
    print("Number of pytorch parameters: ", count_params(model))


    # Define optimizer

    lr = 0.1
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loader, test_loader = get_mnist_loaders(use_cuda=use_cuda, batch_size=batch_size)
    print("Number of training data points:", len(train_loader.dataset))
    print("number of batches per epoch:", len(train_loader))

    total_time = time.time()
    for epoch in range(n_epochs):

        t_start = time.time()
        running_loss = 0.0
        for batch_index, (data, target) in enumerate(train_loader):
            # Send data to correct device
            data, target = data.to(device), target.to(device)
            data = data.view(data.shape[0], -1)


            # Reset gradients
            optimizer.zero_grad()

            output = log_likelihood(model, data)
            loss = -1 * output.sum()

            # Backprop
            loss.backward()
            optimizer.step()


        t_delta = time_delta_now(t_start)
        print("Train Epoch: {} took {}".format(epoch, t_delta))

    end_time = time.time() - total_time
    print("Total training time: {:.2f} seconds".format(end_time))

    model.eval()
    #sampling_ctx = init_default_sampling_context(None,num_samples=10, device=device)
    samples = sample(model)
    samples = samples.view(samples.shape[0], 28, 28)
    for i in range(samples.shape[0]):
        plt.imshow(samples[0].cpu().detach().numpy(), cmap='gray')
        plt.show()



def set_seed(seed: int):
    """
    Set the seed globally for python, numpy and torch.

    Args:
        seed (int): Seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



if __name__ == "__main__":
    torch.cuda.benchmark = True
    run_torch(2, 128)


"""
SPFLOW Test Results:
Results for 5 epochs, batch size 128, cuda:0, Classes: [0, 1]

Spn Config:
    depth = 5
    n_region_nodes = 16
    num_leaves = 16
    num_repetitions = 10
    n_root_nodes = 2
    num_features = 28 ** 2
    outer_product = True
    split_halves = True

Number of pytorch parameters:  1482250
Number of training data points: 12665
number of batches per epoch: 99
Train Epoch: 0 took 0 days, 0 hours, 0 minutes, 25 seconds, 958 milliseconds
Train Epoch: 1 took 0 days, 0 hours, 0 minutes, 25 seconds, 749 milliseconds
Train Epoch: 2 took 0 days, 0 hours, 0 minutes, 25 seconds, 740 milliseconds
Train Epoch: 3 took 0 days, 0 hours, 0 minutes, 25 seconds, 19 milliseconds
Train Epoch: 4 took 0 days, 0 hours, 0 minutes, 25 seconds, 821 milliseconds
Total training time: 124.29 seconds


SPFlow old Test Results:

Number of pytorch parameters:  1482240
Number of training data points: 12665
number of batches per epoch: 99
Train Epoch: 0 took 0 days, 0 hours, 0 minutes, 33 seconds, 764 milliseconds
Train Epoch: 1 took 0 days, 0 hours, 0 minutes, 32 seconds, 352 milliseconds
Train Epoch: 2 took 0 days, 0 hours, 0 minutes, 32 seconds, 902 milliseconds
Train Epoch: 3 took 0 days, 0 hours, 0 minutes, 32 seconds, 814 milliseconds
Train Epoch: 4 took 0 days, 0 hours, 0 minutes, 31 seconds, 391 milliseconds
Total training time: 160.22 seconds

"""