import os
import random
import sys
import time
import csv

import imageio
import numpy as np
import skimage
import torch
import torchvision
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from distributions import RatNormal
from cspn import CSPN, CspnConfig
from rat_spn import RatSpn, RatSpnConfig

from train_mnist import count_params
from train_cspn_mnist_gen import print_cspn_params, time_delta, get_mnist_loaders, eval_root_sum_override, plot_samples

def evaluate_sampling(model, save_dir, device, img_size):
    model.eval()
    log_like = []
    label = torch.as_tensor(np.arange(10)).repeat_interleave(10).to(device)
    with torch.no_grad():
        if isinstance(model, CSPN):
            label = F.one_hot(label, 10).float().to(device)
            samples = model.sample(condition=label)
            log_like.append(model(x=samples, condition=label).mean().tolist())
        else:
            samples = model.sample(class_index=label.tolist())
            log_like = np.nan
        samples = samples.view(-1, *img_size[1:])
        plot_samples(samples, save_dir)
    result_str = f"Samples: Average log-likelihood: {np.mean(log_like):.2f}"
    print(result_str)


def evaluate_model(model, device, loader, tag):
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
    with torch.no_grad():
        for image, label in loader:
            image = image.flatten(start_dim=1).to(device)
            if isinstance(model, CSPN):
                label = F.one_hot(label, 10).float().to(device)
                log_like.append(model(x=image, condition=label).mean().tolist())
            else:
                log_like.append(model(x=image).mean().tolist())
    mean_ll = np.mean(log_like)
    print(f"{tag} set: Average log-likelihood: {mean_ll:.2f}")
    return mean_ll

class CsvLogger(dict):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.other_keys = ['epoch', 'time']
        self.keys_to_avg = ['mnist_test_ll', 'nll_loss', 'ent_loss', 'gmm_ent_lb', 'loss']
        self.no_log_dict = {'batch': None}
        self.reset()
        with open(self.path, 'w') as f:
            w = csv.DictWriter(f, self.keys())
            w.writeheader()

    def add_to_avg_keys(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if k not in self.keys_to_avg:
                self.keys_to_avg += [k]
                self[k] = [v]
            else:
                self[k].append(v)

    def reset(self, epoch: int = None):
        self.update({k: None for k in self.other_keys})
        self.update({k: [] for k in self.keys_to_avg})
        self.no_log_dict.update({k: None for k in self.no_log_dict.keys()})
        if epoch is not None:
            self['epoch'] = epoch

    def average(self):
        self.update({k: np.around(self.mean(k), 2) for k in self.keys_to_avg})

    def write(self):
        with open(self.path, 'a') as f:
            w = csv.DictWriter(f, self.keys())
            w.writerow(self)

    def mean(self, key):
        assert key in self.keys_to_avg, f"key {key} to take mean of is not in keys_to_avg"
        val = self[key]
        if isinstance(val, list):
            if len(val) > 0:
                return np.mean(val)
            else:
                return 0.0
        return val

    def _valid(self, key):
        if key in self.keys() and (mean := self.mean(key)) != 0.0:
            return mean

    def __str__(self):
        return_str = f"Train Epoch: {self['epoch']} took {time_delta(self['time'])}"
        if self.no_log_dict['batch'] is not None:
            return_str += f" @ batch {self.no_log_dict['batch']}"
        if mean := self._valid('ll_loss'):
            return_str += f" - NLL loss: {mean:.2f}"
        if mean := self._valid('ent_loss'):
            return_str += f" - Entropy loss: {mean:.2f}"
        if mean := self._valid('mnist_test_ll'):
            return_str += f" - LL orig mnist test set: {mean:.2f}"
        if mean := self._valid('gmm_ent_lb'):
            return_str += f" - GMM ent lower bound: {mean:.4f}"
        if mean := self._valid('gmm_ent_tayl_appr'):
            return_str += f" - GMM ent taylor approx.: {mean:.4f}"
        if mean := self._valid('gmm_H_0'):
            return_str += f" - 1. Taylor: {mean:.4f}"
        if mean := self._valid('gmm_H_2'):
            return_str += f" - 2. Taylor: {mean:.4f}"
        if mean := self._valid('gmm_H_3'):
            return_str += f" - 3. Taylor: {mean:.4f}"
        if mean := self._valid('inner_ent'):
            return_str += f" - Entropy of inner sums: {mean:.4f}|{self.mean('norm_inner_ent'):.2f}%"
        if mean := self._valid('root_ent'):
            return_str += f" - Entropy of root sum: {mean:.4f}|{self.mean('norm_root_ent'):.2f}%"
        return return_str

    def __setitem__(self, key, value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        if key in self.no_log_dict.keys():
            self.no_log_dict[key] = value
        else:
            super().__setitem__(key, value)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-dev', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', '-ep', type=int, default=100)
    parser.add_argument('--batch_size', '-bs', type=int, default=256)
    parser.add_argument('--results_dir', type=str, default='../spn_experiments/',
                        help='The base directory where the directory containing the results will be saved to.')
    parser.add_argument('--dataset_dir', type=str, default='../data',
                        help='The base directory to provide to the PyTorch Dataloader.')
    parser.add_argument('--model_path', type=str,
                        help='Path to the pretrained model. If it is given, '
                             'all other SPN config parameters are ignored.')
    parser.add_argument('--exp_name', type=str, default='cspn_test',
                        help='Experiment name. The results dir will contain it.')
    parser.add_argument('--repetitions', '-R', type=int, default=5, help='Number of parallel CSPNs to learn at once. ')
    parser.add_argument('--cspn_depth', '-D', type=int, default=3, help='Depth of the CSPN.')
    parser.add_argument('--num_dist', '-I', type=int, default=5, help='Number of Gauss dists per pixel.')
    parser.add_argument('--num_sums', '-S', type=int, default=5, help='Number of sums per RV in each sum layer.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout to apply')
    parser.add_argument('--nr_feat_layers', type=int, default=2, help='Number of fully connected layers that take the'
                                                                      'labels as input and have the sum_param_layers'
                                                                      'and dist_param_layers as heads.')
    parser.add_argument('--nr_sum_param_layers', type=int, default=2,
                        help='Number of fully connected hidden layers in the MLP that '
                             'provides the weights for the sum nodes.')
    parser.add_argument('--nr_dist_param_layers', type=int, default=2,
                        help='Number of fully connected hidden layers in the MLP that '
                             'provides the params for the dist nodes')
    parser.add_argument('--save_interval', type=int, default=50, help='Epoch interval to save model')
    parser.add_argument('--eval_interval', type=int, default=10, help='Epoch interval to evaluate model')
    parser.add_argument('--sample_override_root', action='store_true',
                        help='When evaluating, also sample all input channels of root sum node.')
    parser.add_argument('--verbose', '-V', action='store_true', help='Output more debugging information when running.')
    parser.add_argument('--inspect', action='store_true', help='Enter inspection mode')
    parser.add_argument('--ratspn', action='store_true', help='Use a RATSPN and not a CSPN')
    parser.add_argument('--ent_loss_alpha', '-alpha', type=float, default=0.0,
                        help='Factor for entropy loss at GMM leaves. Default 0.0. '
                             'If 0.0, no gradients are calculated w.r.t. the entropy.')
    parser.add_argument('--adamw', action='store_true', help='Use AdamW optimizer (incorporates weight decay)')
    parser.add_argument('--invert', type=float, default=0.0, help='Probability of an MNIST image being inverted.')
    parser.add_argument('--no_eval_at_start', action='store_true', help='Don\'t evaluate model at the beginning')
    args = parser.parse_args()

    if args.model_path:
        assert os.path.exists(args.model_path), f"The model_path doesn't exist! {args.model_path}"

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
    print("Using device:", device)
    batch_size = args.batch_size

    # The task is to do image in-painting - to fill in a cut-out square in the image.
    # The CSPN needs to learn the distribution of the cut-out given the image with the cut-out part set to 0 as
    # the conditional.

    img_size = (1, 28, 28)  # 3 channels
    cond_size = 10

    csv_log = os.path.join(results_dir, f"log_{args.exp_name}.csv")
    info = CsvLogger(csv_log)
    # Construct Cspn from config
    train_loader, test_loader = get_mnist_loaders(args.dataset_dir, use_cuda, batch_size=batch_size, device=device,
                                                  invert=args.invert, debug_mode=args.verbose)

    if not args.model_path:
        if args.ratspn:
            config = RatSpnConfig()
            config.C = 10
        else:
            config = CspnConfig()
            config.F_cond = (cond_size,)
            config.C = 1
            config.nr_feat_layers = args.nr_feat_layers
            config.fc_sum_param_layers = args.nr_sum_param_layers
            config.fc_dist_param_layers = args.nr_dist_param_layers
        config.F = int(np.prod(img_size))
        config.R = args.repetitions
        config.D = args.cspn_depth
        config.I = args.num_dist
        config.S = args.num_sums
        config.dropout = args.dropout
        config.leaf_base_class = RatNormal
        config.leaf_base_kwargs = {'tanh_factor': 1.0}
        if args.ratspn:
            model = RatSpn(config)
            count_params(model)
        else:
            model = CSPN(config)
            print_cspn_params(model)
        model = model.to(device)
    else:
        print(f"Using pretrained model under {args.model_path}")
        model = torch.load(args.model_path, map_location=device)
    model.train()
    print("Config:", model.config)
    print(model)
    if args.adamw:
        print("Using special optimizer AdamW!")
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    else:
        print("Using regular optimizer Adam.")
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

    lmbda = 1.0
    sample_interval = 1 if args.verbose else args.eval_interval  # number of epochs
    save_interval = 1 if args.verbose else args.save_interval  # number of epochs

    epoch = 0
    for epoch in range(args.epochs):
        if epoch > 20:
            lmbda = 0.5
        t_start = time.time()
        info.reset(epoch)
        for batch_index, (image, label) in enumerate(train_loader):
            # Send data to correct device
            label = label.to(device)
            image = image.to(device)
            # plt.imshow(image[0].permute(1, 2, 0), cmap='Greys)
            # plt.show()

            # Inference
            optimizer.zero_grad()
            data = image.reshape(image.shape[0], -1)
            ent_loss = leaf_entropy_lb = leaf_entropy = torch.zeros(1).to(device)
            gmm_H_0 = gmm_H_2 = gmm_H_3 = torch.zeros(1).to(device)
            if args.ratspn:
                output: torch.Tensor = model(x=data)
                loss_ce = F.cross_entropy(output, label)
                ll_loss = -output.mean()
                loss = (1 - lmbda) * ll_loss + lmbda * loss_ce
            else:
                label = F.one_hot(label, cond_size).float().to(device)
                model.vi_entropy_approx(sample_size=10, condition=label)
                output: torch.Tensor = model(x=data, condition=label)
                ll_loss = -output.mean()
                model.sample(n=3)
                if args.ent_loss_alpha > 0.0:
                    ent_loss = -args.ent_loss_alpha * leaf_entropy_lb
                loss = ll_loss + ent_loss

            loss.backward()
            optimizer.step()
            info.add_to_avg_keys(nll_loss=ll_loss, ent_loss=ent_loss, loss=loss,
                                 gmm_ent_lb=leaf_entropy_lb,
                                 # gmm_ent_tayl_appr=leaf_entropy, gmm_H_0=gmm_H_0, gmm_H_2=gmm_H_2, gmm_H_3=gmm_H_3
                                 )

            # Log stuff
            if args.verbose:
                info['time'] = time.time()-t_start
                info['batch'] = batch_index
                print(info)
                # print(info, end="\r")

        t_delta = np.around(time.time()-t_start, 2)
        if epoch % save_interval == (save_interval-1):
            print("Saving model ...")
            torch.save(model, os.path.join(model_dir, f"epoch-{epoch:03}_{args.exp_name}.pt"))

        if epoch % sample_interval == (sample_interval-1):
            if args.sample_override_root:
                print("Sampling from all input channels to root sum node ...")
                root_sum_override_dir = os.path.join(sample_dir, f"epoch-{epoch:03}_root_sum_override")
                eval_root_sum_override(model, root_sum_override_dir, device, img_size)
            print("Evaluating model ...")
            save_path = os.path.join(sample_dir, f"epoch-{epoch:03}_{args.exp_name}.png")
            evaluate_sampling(model, save_path, device, img_size)
            info['mnist_test_ll'] = evaluate_model(model, device, test_loader, "MNIST test")

        info.average()
        info['time'] = t_delta
        info['batch'] = None
        info.write()
        print(info)



