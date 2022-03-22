import os
import random
import sys
import time
import csv

import imageio
import numpy as np
import skimage
import torch as th
import torchvision
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from distributions import RatNormal
from cspn import CSPN, CspnConfig, print_cspn_params
from rat_spn import RatSpn, RatSpnConfig


def time_delta(t_delta: float) -> str:
    """
    Convert a timestamp into a human readable timestring.
    Args:
        t_delta (float): Difference between two timestamps of time.time()

    Returns:
        Human readable timestring.
    """
    if t_delta is None:
        return ""
    hours = round(t_delta // 3600)
    minutes = round(t_delta // 60 % 60)
    seconds = round(t_delta % 60)
    return f"{hours}h, {minutes}min, {seconds}s"


def get_mnist_loaders(dataset_dir, use_cuda, device, batch_size, img_side_len, invert=0.0, debug_mode=False):
    """
    Get the MNIST pytorch data loader.

    Args:
        use_cuda: Use cuda flag.

    """
    kwargs = {"num_workers": 8, "pin_memory": True} if use_cuda and not debug_mode else {}

    test_batch_size = batch_size

    # Transform from interval (0.0, 1.0) to (0.01, 0.99) so tanh squash experiments converge better
    transformer = transforms.Compose([transforms.ToTensor(),
                                      transforms.Resize(img_side_len, antialias=False),
                                      transforms.Normalize((-0.010204,), (1.0204,)),
                                      transforms.RandomInvert(p=invert)])
    # Train data loader
    train_loader = th.utils.data.DataLoader(
        datasets.MNIST(dataset_dir, train=True, download=True, transform=transformer),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    # Test data loader
    test_loader = th.utils.data.DataLoader(
        datasets.MNIST(dataset_dir, train=False, transform=transformer),
        batch_size=test_batch_size,
        shuffle=True,
        **kwargs,
    )
    return train_loader, test_loader


def evaluate_sampling(model, save_dir, device, img_size, mpe=False, eval_ll=True, style='index'):
    model.eval()
    log_like = []
    label = th.as_tensor(np.arange(10)).to(device)
    samples_per_label = 10
    with th.no_grad():
        if isinstance(model, CSPN):
            label = F.one_hot(label, 10).float().to(device)
            if style == 'index':
                samples = model.sample_index_style(n=samples_per_label, condition=label, is_mpe=mpe)
            else:
                samples = model.sample_onehot_style(n=samples_per_label, condition=label, is_mpe=mpe)
            if eval_ll:
                log_like.append(model(x=samples.atanh(), condition=None).mean().tolist())
        else:
            samples = model.sample(n=samples_per_label, class_index=label)
            if eval_ll:
                log_like.append(model(x=samples).mean().tolist())
        if model.config.tanh_squash:
            samples.mul_(0.5).add_(0.5)
        samples = samples.view(-1, *img_size[1:])
        plot_samples(samples, save_dir)
    result_str = f"{'MPE sample' if mpe else 'Sample'} average log-likelihood: {np.mean(log_like):.2f}"
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
    with th.no_grad():
        for image, label in loader:
            image = image.flatten(start_dim=1).to(device)
            if model.config.tanh_squash:
                image.sub_(0.5).mul_(2).atanh_()
            if isinstance(model, CSPN):
                label = F.one_hot(label, 10).float().to(device)
                log_like.append(model(x=image, condition=label).mean().tolist())
            else:
                log_like.append(model(x=image).mean().tolist())
    mean_ll = np.mean(log_like)
    print(f"{tag} set: Average log-likelihood: {mean_ll:.2f}")
    return mean_ll


def plot_samples(x: th.Tensor, path):
    """
    Plot a single sample with the target and prediction in the title.

    Args:
        x (th.Tensor): Batch of input images. Has to be shape: [N, C, H, W].
    """
    x.unsqueeze_(1)
    # Clip to valid range
    x[x < 0.0] = 0.0
    x[x > 1.0] = 1.0

    tensors = torchvision.utils.make_grid(x, nrow=10, padding=1).cpu()
    arr = tensors.permute(1, 2, 0).numpy()
    arr = skimage.img_as_ubyte(arr)
    imageio.imwrite(path, arr)


class CsvLogger(dict):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.other_keys = ['epoch', 'time']
        self.keys_to_avg = [
            'mnist_test_ll', 'nll_loss', 'mse_loss', 'ent_loss',
            'vi_ent_approx', 'loss'
        ]
        for i in range(20):
            self.keys_to_avg.append(f"{i}/weight_entropy")
            self.keys_to_avg.append(f"{i}/weighted_child_ent")
            self.keys_to_avg.append(f"{i}/weighted_aux_resp")
        self.no_log_dict = {'batch': None}
        self.reset()
        with open(self.path, 'w') as f:
            w = csv.DictWriter(f, self.keys())
            w.writeheader()

    def add_to_avg_keys(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, th.Tensor):
                v = v.item()
            if k not in self.keys_to_avg:
                raise KeyError(f"{k} not in keys_to_avg")
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
        if mean := self._valid('nll_loss'):
            return_str += f" - NLL loss: {mean:.2f}"
        if mean := self._valid('mse_loss'):
            return_str += f" - MSE loss: {mean:.4f}"
        if mean := self._valid('ent_loss'):
            return_str += f" - Entropy loss: {mean:.2f}"
        if mean := self._valid('mnist_test_ll'):
            return_str += f" - LL orig mnist test set: {mean:.2f}"
        if mean := self._valid('vi_ent_approx'):
            return_str += f" - VI ent. approx.: {mean:.4f}"
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
        for i in range(10):
            if self._valid(f'{i}/weight_entropy') or self._valid(f'{i}/weighted_child_ent') \
                    or self._valid(f'{i}/weighted_aux_resp'):
                return_str += f" - Sum layer {i}: "
                if mean := self._valid(f'{i}/weight_entropy'):
                    return_str += f"Weight ent {mean:.4f} "
                if mean := self._valid(f'{i}/weighted_child_ent'):
                    return_str += f"Weighted child ent {mean:.4f} "
                if mean := self._valid(f'{i}/weighted_aux_resp'):
                    return_str += f"Weighted aux. responsib. {mean:.4f}"

        return return_str

    def __setitem__(self, key, value):
        if isinstance(value, th.Tensor):
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
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', '-bs', type=int, default=256)
    parser.add_argument('--results_dir', type=str, default='../../spn_experiments',
                        help='The base directory where the directory containing the results will be saved to.')
    parser.add_argument('--dataset_dir', type=str, default='../../spn_experiments/data',
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
    parser.add_argument('--feat_layers', type=int, nargs='+',
                        help='List of sizes of the CSPN feature layers.')
    parser.add_argument('--sum_param_layers', type=int, nargs='+',
                        help='List of sizes of the CSPN sum param layers.')
    parser.add_argument('--dist_param_layers', type=int, nargs='+',
                        help='List of sizes of the CSPN dist param layers.')
    parser.add_argument('--save_interval', type=int, default=50, help='Epoch interval to save model')
    parser.add_argument('--eval_interval', type=int, default=10, help='Epoch interval to evaluate model')
    parser.add_argument('--verbose', '-V', action='store_true', help='Output more debugging information when running.')
    parser.add_argument('--ratspn', action='store_true', help='Use a RATSPN and not a CSPN')
    parser.add_argument('--ent_approx_separate_samples', action='store_true',
                        help='When approximating the entropy, don\'t share child samples for each sum in a layer.')
    parser.add_argument('--ent_approx__leaf_ent_closed_form', action='store_true',
                        help='Approximate the leaf entropies in closed form instead of approximating it.')
    parser.add_argument('--ent_approx__sample_size', type=int, default=5,
                        help='When approximating entropy, use this sample size. ')
    parser.add_argument('--ent_approx_sample_with_grad', action='store_true',
                        help='When approximating the entropy, sample children in a differentiable way.')
    parser.add_argument('--ent_loss_coef', type=float, default=0.0,
                        help='Factor for entropy loss. Default 0.0. '
                             'If 0.0, no gradients are calculated w.r.t. the entropy.')
    parser.add_argument('--invert', type=float, default=0.0, help='Probability of an MNIST image being inverted.')
    parser.add_argument('--no_eval_at_start', action='store_true', help='Don\'t evaluate model at the beginning')
    parser.add_argument('--learn_by_sampling', action='store_true', help='Learn in sampling mode.')
    parser.add_argument('--learn_by_sampling__sample_size', type=int, default=10,
                        help='When learning by sampling, this arg sets the number of samples generated for each label.')
    parser.add_argument('--tanh', action='store_true', help='Apply tanh squashing to leaves.')
    parser.add_argument('--sigmoid_std', action='store_true', help='Use sigmoid to set std.')
    parser.add_argument('--no_correction_term', action='store_true', help='Don\'t apply tanh correction term to logprob')
    parser.add_argument('--plot_vi_log', action='store_true',
                        help='Collect information from variational inference entropy '
                             'approx. and plot it at the end of the epoch.')
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
        device = th.device("cpu")
        use_cuda = False
    else:
        device = th.device("cuda:0")
        use_cuda = True
        th.cuda.benchmark = True
    print("Using device:", device)
    batch_size = args.batch_size

    img_size = (1, 28, 28)  # 3 channels
    cond_size = 10

    csv_log = os.path.join(results_dir, f"log_{args.exp_name}.csv")
    logger = CsvLogger(csv_log)
    # Construct Cspn from config
    train_loader, test_loader = get_mnist_loaders(args.dataset_dir, use_cuda, batch_size=batch_size, device=device,
                                                  img_side_len=img_size[1], invert=args.invert, debug_mode=args.verbose)
    print(f"There are {len(train_loader)} batches per epoch")

    if not args.model_path:
        if args.ratspn:
            config = RatSpnConfig()
            config.C = 10
        else:
            config = CspnConfig()
            config.F_cond = (cond_size,)
            config.C = 1
            config.feat_layers = args.feat_layers
            config.sum_param_layers = args.sum_param_layers
            config.dist_param_layers = args.dist_param_layers
        config.F = int(np.prod(img_size))
        config.R = args.repetitions
        config.D = args.cspn_depth
        config.I = args.num_dist
        config.S = args.num_sums
        config.dropout = args.dropout
        config.leaf_base_class = RatNormal
        if args.tanh:
            config.tanh_squash = True
            config.leaf_base_kwargs = {'no_tanh_log_prob_correction': args.no_correction_term}
            # config.leaf_base_kwargs = {'min_mean': -5.0, 'max_mean': 5.0}
        else:
            config.leaf_base_kwargs = {'min_mean': 0.0, 'max_mean': 1.0}
        if args.sigmoid_std:
            config.leaf_base_kwargs['min_sigma'] = 0.1
            config.leaf_base_kwargs['max_sigma'] = 1.0
        if args.ratspn:
            model = RatSpn(config)
            raise NotImplementedError("The log normalization of parameters in the layers "
                                      "themselves was removed and must be redone.")
            count_params(model)
        else:
            model = CSPN(config)
            print_cspn_params(model)
        model = model.to(device)
    else:
        print(f"Using pretrained model under {args.model_path}")
        model = th.load(args.model_path, map_location=device)
        model.create_one_hot_in_channel_mapping()
        model.set_no_tanh_log_prob_correction()
    model.train()
    print("Config:", model.config)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    print(f"Optimizer: {optimizer}")

    lmbda = 1.0
    sample_interval = 1 if args.verbose else args.eval_interval  # number of epochs
    save_interval = 1 if args.verbose else args.save_interval  # number of epochs

    epoch = 0
    if not args.no_eval_at_start:
        print("Evaluating model ...")
        save_path = os.path.join(sample_dir, f"epoch-{epoch:03}_{args.exp_name}.png")
        evaluate_sampling(model, save_path, device, img_size)
        save_path = os.path.join(sample_dir, f"mpe-epoch-{epoch:03}_{args.exp_name}.png")
        evaluate_sampling(model, save_path, device, img_size, mpe=True)
        logger.reset(epoch)
        logger['mnist_test_ll'] = evaluate_model(model, device, test_loader, "MNIST test")
        logger.average()
        logger.write()
    for epoch in range(args.epochs):
        if epoch > 20:
            lmbda = 0.5
        t_start = time.time()
        logger.reset(epoch)
        temp_log = []
        temp_log_separate_samples = []
        for batch_index, (image, label) in enumerate(train_loader):
            # Send data to correct device
            label = label.to(device)
            image = image.to(device)
            if model.config.tanh_squash:
                image.sub_(0.5).mul_(2).atanh_()
            # plt.imshow(image[0].permute(1, 2, 0), cmap='Greys)
            # plt.show()

            # Inference
            optimizer.zero_grad()
            data = image.reshape(image.shape[0], -1)
            mse_loss = ll_loss = ent_loss = vi_ent_approx_separate_samples = vi_ent_approx = th.zeros(1).to(device)
            if args.ratspn:
                output: th.Tensor = model(x=data)
                loss_ce = F.cross_entropy(output.squeeze(1), label)
                ll_loss = -output.mean()
                loss = (1 - lmbda) * ll_loss + lmbda * loss_ce
                vi_ent_approx = model.vi_entropy_approx(sample_size=10).mean()
            else:
                label = F.one_hot(label, cond_size).float().to(device)
                if args.learn_by_sampling:
                    sample: th.Tensor = model.sample_onehot_style(condition=label, n=args.learn_by_sampling__sample_size)
                    if model.config.tanh_squash:
                        sample = sample.clamp(-0.99999, 0.99999).atanh()
                    mse_loss: th.Tensor = ((data - sample) ** 2).mean()
                else:
                    output: th.Tensor = model(x=data, condition=label)
                    ll_loss = -output.mean()
                vi_ent_approx, batch_ent_log = model.vi_entropy_approx(
                    sample_size=args.ent_approx__sample_size, condition=label, verbose=True,
                    aux_resp_ll_with_grad=True, aux_resp_sample_with_grad=True,
                )
                vi_ent_approx = vi_ent_approx.mean()
                # vi_ent_approx_separate_samples = vi_ent_approx_separate_samples.mean()
                if args.plot_vi_log:
                    temp_log.append(batch_ent_log)
                # temp_log_separate_samples.append(batch_ent_log_separate_samples)
                # sample = model.sample(n=3, condition=None)
                if args.ent_loss_coef > 0.0:
                    ent_loss = -args.ent_loss_coef * vi_ent_approx
                loss = mse_loss + ll_loss + ent_loss

            loss.backward()
            optimizer.step()
            logger.add_to_avg_keys(
                nll_loss=ll_loss, mse_loss=mse_loss, ent_loss=ent_loss, loss=loss,
                vi_ent_approx=vi_ent_approx,
            )
            for lay_nr, lay_dict in batch_ent_log.items():
                for key, val in lay_dict.items():
                    logger.add_to_avg_keys(**{f"{lay_nr}/{key}": val})

            if False and batch_index > 0 and batch_index % 1000 == 0:
                exit()
                logger['time'] = time.time() - t_start
                logger['batch'] = batch_index
                print(logger)
            # Log stuff
            if args.verbose:
                logger['time'] = time.time()-t_start
                logger['batch'] = batch_index
                print(logger)
                # print(logger, end="\r")

        t_delta = np.around(time.time()-t_start, 2)
        if epoch % save_interval == (save_interval-1):
            print("Saving model ...")
            th.save(model, os.path.join(model_dir, f"epoch-{epoch:03}_{args.exp_name}.pt"))

        if epoch % sample_interval == (sample_interval-1):
            print("Evaluating model ...")
            save_path = os.path.join(sample_dir, f"epoch-{epoch:03}_{args.exp_name}.png")
            evaluate_sampling(model, save_path, device, img_size)
            save_path = os.path.join(sample_dir, f"mpe-epoch-{epoch:03}_{args.exp_name}.png")
            evaluate_sampling(model, save_path, device, img_size, mpe=True)
            logger['mnist_test_ll'] = evaluate_model(model, device, test_loader, "MNIST test")

        logger.average()
        logger['time'] = t_delta
        logger['batch'] = None
        logger.write()
        print(logger)

        if args.plot_vi_log:  # epoch level
            print("Plotting log data from variational inference entropy approximation...")
            save_dir = os.path.join(results_dir, f"plots_epoch{epoch}")
            os.makedirs(save_dir)
            def make_data_struct(source_struct, target_struct):
                for key, item in source_struct.items():
                    if isinstance(item, dict):
                        if key in target_struct.keys():
                            target_struct[key] = make_data_struct(item, target_struct[key])
                        else:
                            target_struct[key] = make_data_struct(item, {})
                    else:
                        if key in target_struct.keys():
                            target_struct[key] = np.concatenate((target_struct[key], np.asarray(item)),
                                                                axis=None)
                        else:
                            target_struct[key] = np.asarray(item)
                return target_struct

            # for tag, log in {'no_shared_samples': temp_log, 'separate_samples': temp_log_separate_samples}.items():
            for tag, log in {' ': temp_log}.items():
                log_struct = {}
                for i in range(len(log)):
                    log_struct: dict = make_data_struct(log[i], log_struct)

                for log_key in log_struct[1].keys():
                    plt.figure()
                    title_str = f"{log_key} over all sum layers ({len(log_struct.keys())} is root) - {tag}"
                    plt.title(title_str)
                    color_counter = 0
                    for lay_nr, lay_dict in reversed(log_struct.items()):
                        log_item = lay_dict[log_key]
                        if isinstance(log_item, dict):
                            plt.plot(
                                range(len(log_item['mean'])),
                                log_item['mean'],
                                label=f"Sum layer {(lay_nr+1)//2}",
                                color=f'C{color_counter}',
                            )
                            plt.fill_between(
                                range(len(log_item['mean'])),
                                log_item['min'],
                                log_item['max'],
                                # log_item['mean'] - log_item['std'],
                                # log_item['mean'] + log_item['std'],
                                color=f'C{color_counter}',
                                alpha=0.2,
                                )
                        else:
                            plt.plot(
                                range(len(log_item)),
                                log_item,
                                label=f"Sum layer {(lay_nr+1)//2}",
                                color=f'C{color_counter}',
                            )
                        color_counter += 1
                    plt.legend()
                    plt.savefig(os.path.join(save_dir, f"{tag}_{log_key}_over_all_sum_layers.png"))

                color_counter = 0
                for lay_nr, lay_dict in reversed(log_struct.items()):
                    for log_key, log_item in lay_dict.items():
                        if log_key in [
                            'weight_ent', 'weighted_child_entropies', 'entropy',
                        ]:
                            continue
                        plt.figure()
                        title_str = f"Sum layer #{(lay_nr+1)//2} of {len(log_struct.keys())} " \
                                    f"({len(log_struct.keys())} is root) - {tag}"
                        plt.title(title_str)
                        if isinstance(log_item, dict):
                            plt.plot(
                                range(len(log_item['mean'])),
                                log_item['mean'],
                                label=log_key,
                                color=f'C{color_counter}',
                            )
                            plt.fill_between(
                                range(len(log_item['mean'])),
                                log_item['min'],
                                log_item['max'],
                                # log_item['mean'] - log_item['std'],
                                # log_item['mean'] + log_item['std'],
                                color=f'C{color_counter}',
                                alpha=0.2,
                                )
                        else:
                            plt.plot(
                                range(len(log_item)),
                                log_item,
                                label=log_key,
                                color=f'C{color_counter}',
                            )
                        plt.legend()
                        plt.savefig(os.path.join(save_dir, f"{tag}_{log_key}_sum_layer_{(lay_nr+1)//2}.png"))
                    color_counter += 1



