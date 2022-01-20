import os
import random
import sys
import time

import imageio
import numpy as np
import skimage
import torch
import torchvision
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from spn.experiments.RandomSPNs_layerwise.distributions import RatNormal
from spn.experiments.RandomSPNs_layerwise.cspn import CSPN, CspnConfig

from train_mnist import count_params, ensure_dir, set_seed


def print_cspn_params(cspn: CSPN):
    print(f"Total params in CSPN: {count_params(cspn)}")
    print(f"Params to extract features from the conditional: {count_params(cspn.feat_layers)}")
    print(f"Params in MLP for the sum params, excluding the heads: {count_params(cspn.sum_layers)}")
    print(f"Params in the heads of the sum param MLPs: {sum([count_params(head) for head in cspn.sum_param_heads])}")
    print(f"Params in MLP for the dist params, excluding the heads: {count_params(cspn.dist_layers)}")
    print(f"Params in the heads of the dist param MLPs: "
          f"{count_params(cspn.dist_mean_head) + count_params(cspn.dist_std_head)}")


def time_delta(t_delta: float) -> str:
    """
    Convert a timestamp into a human readable timestring.
    Args:
        t_delta (float): Difference between two timestamps of time.time()

    Returns:
        Human readable timestring.
    """
    hours = round(t_delta // 3600)
    minutes = round(t_delta // 60 % 60)
    seconds = round(t_delta % 60)
    millisecs = round(t_delta % 1 * 1000)
    return f"{hours} hours, {minutes} minutes, {seconds} seconds, {millisecs} milliseconds"


def get_mnist_loaders(dataset_dir, use_cuda, device, batch_size):
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
        datasets.MNIST(dataset_dir, train=True, download=True, transform=transformer),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    # Test data loader
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(dataset_dir, train=False, transform=transformer),
        batch_size=test_batch_size,
        shuffle=True,
        **kwargs,
    )
    return train_loader, test_loader


def evaluate_model(model, save_dir, device, loader, tag):
    """
    Description for method evaluate_model.

    Args:
        model: PyTorch module or a list of modules, one for each image channel
        device: Execution device.
        loader: Data loader.
        tag (str): Tag for information.

    Returns:
        float: Tuple of loss and accuracy.
    """
    model.eval()
    log_like = []
    label = torch.as_tensor(np.arange(10)).repeat_interleave(10)
    label = F.one_hot(label, 10).float().to(device)
    samples = model.sample(condition=label).view(-1, *img_size[1:])
    plot_samples(samples, save_dir)
    with torch.no_grad():
        for image, label in loader:
            label = F.one_hot(label, 10).float().to(device)
            image = image.flatten(start_dim=1).to(device)
            log_like.append(model(x=image, condition=label).mean().tolist())
    print("{} set: Average log-likelihood: {:.4f}".format(tag, np.mean(log_like)))


def plot_samples(x: torch.Tensor, path):
    """
    Plot a single sample with the target and prediction in the title.

    Args:
        x (torch.Tensor): Batch of input images. Has to be shape: [N, C, H, W].
    """
    x.unsqueeze_(1)
    # Clip to valid range
    x[x < 0.0] = 0.0
    x[x > 1.0] = 1.0

    tensors = torchvision.utils.make_grid(x, nrow=10, padding=1).cpu()
    arr = tensors.permute(1, 2, 0).numpy()
    arr = skimage.img_as_ubyte(arr)
    imageio.imwrite(path, arr)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-dev', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', '-ep', type=int, default=100)
    parser.add_argument('--batch_size', '-bs', type=int, default=256)
    parser.add_argument('--results_dir', type=str, default='.',
                        help='The base directory where the directory containing the results will be saved to.')
    parser.add_argument('--dataset_dir', type=str, default='../data',
                        help='The base directory to provide to the PyTorch Dataloader.')
    parser.add_argument('--exp_name', type=str, default='stl', help='Experiment name. The results dir will contain it.')
    parser.add_argument('--repetitions', '-R', type=int, default=5, help='Number of parallel CSPNs to learn at once. ')
    parser.add_argument('--cspn_depth', '-D', type=int, default=3, help='Depth of the CSPN.')
    parser.add_argument('--num_dist', '-I', type=int, default=5, help='Number of Gauss dists per pixel.')
    parser.add_argument('--num_sums', '-S', type=int, default=5, help='Number of sums per RV in each sum layer.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout to apply')
    parser.add_argument('--nr_feat_layers', type=int, default=1, help='Number of fully connected layers that take the'
                                                                      'labels as input and have the sum_param_layers'
                                                                      'and dist_param_layers as heads.')
    parser.add_argument('--nr_sum_param_layers', type=int, default=1,
                        help='Number of fully connected hidden layers in the MLP that '
                             'provides the weights for the sum nodes.')
    parser.add_argument('--nr_dist_param_layers', type=int, default=1,
                        help='Number of fully connected hidden layers in the MLP that '
                             'provides the params for the dist nodes')
    parser.add_argument('--verbose', '-V', action='store_true', help='Output more debugging information when running.')
    parser.add_argument('--inspect', action='store_true', help='Enter inspection mode')
    parser.add_argument('--sumfirst', action='store_true', help='Make first layer after dists a sum layer.')
    args = parser.parse_args()

    results_dir = os.path.join(args.results_dir, f"results_{args.exp_name}")
    model_dir = os.path.join(results_dir, "models")
    sample_dir = os.path.join(results_dir, "samples")

    for d in [results_dir, model_dir, sample_dir, args.dataset_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    if args.device == "cpu":
        device = torch.device("cpu")
        use_cuda = False
    else:
        device = torch.device("cuda:0")
        use_cuda = True
        torch.cuda.benchmark = True

    batch_size = args.batch_size

    # The task is to do image in-painting - to fill in a cut-out square in the image.
    # The CSPN needs to learn the distribution of the cut-out given the image with the cut-out part set to 0 as
    # the conditional.

    img_size = (1, 28, 28)  # 3 channels
    cond_size = 10

    inspect = args.inspect
    if inspect:
        def plot_img(image: torch.Tensor, title: str = None, path=None):
            # Tensor shape N x channels x rows x cols
            tensors = torchvision.utils.make_grid(image, nrow=image.shape[0], padding=1)
            arr = tensors.permute(1, 2, 0).cpu().numpy()
            arr = skimage.img_as_ubyte(arr)
            imageio.imwrite(path, arr)
            return
            plt.imshow(arr)
            plt.title(title, fontdict={'fontsize': 10})
            plt.show()

        top_5_ll = torch.ones(5) * -10e6
        top_5_ll_img = torch.zeros(5, *img_size)
        low_5_ll = torch.ones(5) * 10e6
        low_5_ll_img = torch.zeros(5, *img_size)

        # path = 'results_stl_1/models/epoch-079.pt'
        # path = 'results_stl_2/models/epoch-029.pt'
        # path = 'results_stl_3/models/epoch-049.pt'
        # path = 'results_stl7/models/epoch-099.pt'
        path = 'results_stl10/models/epoch-099.pt'
        # path = [f"results_stl6/models/epoch-099-chan{ch}.pt" for ch in range(3)]
        results_dir = os.path.join(args.results_dir, f"results_stl10")

        models = []
        model = None
        if isinstance(path, list):
            models = [torch.load(p).cpu() for p in path]
            spn_per_channel = True
        else:
            model = torch.load(path).cpu()
            spn_per_channel = False

        show_all = True
        find_top_low_LL = False
        if show_all:
            batch_size = 5
        else:
            batch_size = 256

        train_loader, test_loader = get_mnist_loaders(args.dataset_dir, args.grayscale,
                                                      batch_size=batch_size, device=device)
        for i, (image, cond) in enumerate(train_loader):
            cond = F.one_hot(cond, cond_size).float()
            data_ll = model(x=image.flatten(start_dim=1), condition=cond).flatten()
            sample = model.sample(condition=None)
            sample_ll = model(x=sample, condition=None).flatten()

            sample[sample < 0.0] = 0.0
            sample[sample > 1.0] = 1.0

            if find_top_low_LL:
                top_5_ll, indices = torch.cat((top_5_ll, sample_ll), dim=0).sort(descending=True)
                top_5_ll = top_5_ll[:5]
                indices = indices[:5]
                imgs = []
                for ind in indices:
                    img = top_5_ll_img[ind] if ind < 5 else cond[ind-5]
                    imgs.append(img.unsqueeze(0))
                top_5_ll_img = torch.cat(imgs, dim=0)

                low_5_ll, indices = torch.cat((low_5_ll, sample_ll), dim=0).sort(descending=False)
                low_5_ll = low_5_ll[:5]
                indices = indices[:5]
                imgs = []
                for ind in indices:
                    img = low_5_ll_img[ind] if ind < 5 else cond[ind-5]
                    imgs.append(img.unsqueeze(0))
                low_5_ll_img = torch.cat(imgs, dim=0)

                if True:
                    lls = [[f"{n:.2f}" for n in ll.tolist()] for ll in [top_5_ll, low_5_ll]]
                    title = f"Samples of highest LL:\nLLs of sampled center boxes: [{', '.join(lls[0])}]"
                    plot_img(top_5_ll_img, title)
                    title = f"Samples of lowest LL:\nLLs of sampled center boxes: [{', '.join(lls[1])}]"
                    plot_img(low_5_ll_img, title)
                    print("Set breakpoint here")

            if show_all:
                lls = [[f"{n:.2f}" for n in ll.tolist()] for ll in [data_ll, sample_ll]]
                title = f"LLs of original center boxes: [{', '.join(lls[0])}]\n" \
                        f"LLs of sampled center boxes: [{', '.join(lls[1])}]"
                plot_img(sample, title, path=os.path.join(results_dir, f"inspect{i+1}.png"))
                print("Set breakpoint here")

            print(i)
            if i > 9:
                exit()
        exit()

    # Construct Cspn from config
    train_loader, test_loader = get_mnist_loaders(args.dataset_dir, use_cuda, batch_size=batch_size, device=device)
    config = CspnConfig()
    config.F = int(np.prod(img_size))
    config.F_cond = (cond_size,)
    config.R = args.repetitions
    config.D = args.cspn_depth
    config.I = args.num_dist
    config.S = args.num_sums
    config.C = 1
    config.dropout = 0.0
    config.leaf_base_class = RatNormal
    config.leaf_base_kwargs = {'min_sigma': 0.1, 'max_sigma': 1.0, 'min_mean': 0.0, 'max_mean': 1.0}
    config.first_layer_sum = args.sumfirst

    config.nr_feat_layers = args.nr_feat_layers
    config.fc_sum_param_layers = args.nr_sum_param_layers
    config.fc_dist_param_layers = args.nr_dist_param_layers

    print("Using device:", device)
    print("Config:", config)
    model = CSPN(config)
    model = model.to(device)
    model.train()
    print(model)
    print_cspn_params(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    sample_interval = 1 if args.verbose else 10  # number of epochs
    for epoch in range(args.epochs):
        t_start = time.time()
        running_loss = []
        running_ent = []
        cond = None
        for batch_index, (image, label) in enumerate(train_loader):
            # Send data to correct device
            image = image.to(device)
            label = F.one_hot(label, cond_size).float().to(device)
            # plt.imshow(data[0].permute(1, 2, 0))
            # plt.show()

            # Inference
            # evaluate_model(model, cut_out_center, insert_center, "test.png", device, train_loader, "Train")
            # model.entropy_lb(cond)
            optimizer.zero_grad()
            data = image.reshape(image.shape[0], -1)
            output: torch.Tensor = model(x=data, condition=label)
            loss = -output.mean()
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())

            # with torch.no_grad():
            #     ent = model.log_entropy(condition=None).mean()
            #     running_ent.append(ent.item())

            # Log stuff
            if args.verbose:
                batch_delta = time_delta((time.time()-t_start)/(batch_index+1))
                print(f"Epoch {epoch} ({100.0 * batch_index / len(train_loader):.1f}%) "
                      f"Avg. loss: {np.mean(running_loss):.2f} - Batch {batch_index} - "
                      f"Avg. batch time {batch_delta}",
                      end="\r")

        t_delta = time_delta(time.time()-t_start)
        print("Train Epoch: {} took {}".format(epoch, t_delta))
        if epoch % sample_interval == (sample_interval-1):
            print("Saving and evaluating model ...")
            torch.save(model, os.path.join(model_dir, f"epoch-{epoch:03}.pt"))
            save_path = os.path.join(sample_dir, f"epoch-{epoch:03}.png")
            evaluate_model(model, save_path, device, train_loader, "Train")
            evaluate_model(model, save_path, device, test_loader, "Test")


