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

from spn.experiments.RandomSPNs_layerwise.distributions import RatNormal
from spn.experiments.RandomSPNs_layerwise.cspn import CSPN, CspnConfig
from spn.experiments.RandomSPNs_layerwise.rat_spn import RatSpn, RatSpnConfig

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
    if t_delta is None:
        return ""
    hours = round(t_delta // 3600)
    minutes = round(t_delta // 60 % 60)
    seconds = round(t_delta % 60)
    return f"{hours}h, {minutes}min, {seconds}s"


def get_mnist_loaders(dataset_dir, use_cuda, device, batch_size, invert=False):
    """
    Get the MNIST pytorch data loader.

    Args:
        use_cuda: Use cuda flag.

    """
    kwargs = {"num_workers": 8, "pin_memory": True} if use_cuda else {}

    test_batch_size = batch_size

    if invert:
        transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
                                          transforms.RandomInvert(p=1.0)])
    else:
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


def evaluate_sampling(model, save_dir, device, img_size):
    model.eval()
    log_like = []
    label = torch.as_tensor(np.arange(10)).repeat_interleave(10).to(device)
    with torch.no_grad():
        if isinstance(model, RatSpn):
            samples = model.sample(class_index=label.tolist())
            log_like = np.nan
        else:
            label = F.one_hot(label, 10).float().to(device)
            samples = model.sample(condition=label)
            log_like.append(model(x=samples, condition=label).mean().tolist())
        samples = samples.view(-1, *img_size[1:])
        plot_samples(samples, save_dir)
    print(f"Samples: Average log-likelihood: {np.mean(log_like):.4f}")


def eval_root_sum_override(model, save_dir, device, img_size):
    for d in range(10):
        if not os.path.exists(os.path.join(save_dir, f'cond_{d}')):
            os.makedirs(os.path.join(save_dir, f'cond_{d}'))
        cond = torch.ones(model.config.R * model.config.S ** 2).long().to(device) * d
        with torch.no_grad():
            if isinstance(model, RatSpn):
                sample = model.sample(class_index=cond.tolist(), override_root=True)
            else:
                cond = F.one_hot(cond, 10).float().to(device)
                sample = model.sample(condition=cond, override_root=True)
        sample[sample < 0.0] = 0.0
        sample[sample > 1.0] = 1.0
        sample = sample.view(-1, *img_size)

        for i in range(model.config.R):
            b = sample[(i * model.config.S ** 2):((i + 1) * model.config.S ** 2), :, :, :]
            tensors = torchvision.utils.make_grid(b, nrow=10, padding=1)
            arr = tensors.permute(1, 2, 0).cpu().numpy()
            arr = skimage.img_as_ubyte(arr)
            imageio.imwrite(os.path.join(save_dir, f"cond_{d}", f'rep{i}.png'), arr)


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
            if isinstance(model, RatSpn):
                log_like.append(model(x=image).mean().tolist())
            else:
                label = F.one_hot(label, 10).float().to(device)
                log_like.append(model(x=image, condition=label).mean().tolist())
    mean_ll = np.mean(log_like)
    print(f"{tag} set: Average log-likelihood: {mean_ll}")
    return mean_ll


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


class CsvLogger(dict):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.other_keys = ['epoch', 'time']
        self.keys_to_avg = ['mnist_test_ll', 'll_loss', 'ent_loss', 'gmm_ent', 'inner_ent', 'norm_inner_ent',
                            'root_ent', 'norm_root_ent', 'loss']
        self.no_log_dict = {'batch': None}
        self.reset()
        with open(self.path, 'w') as f:
            w = csv.DictWriter(f, self.keys())
            w.writeheader()

    def add_to_avg_keys(self, **kwargs):
        for k, v in kwargs.items():
            assert k in self.keys_to_avg, f"{k} is not in keys_to_avg!"
            if isinstance(v, torch.Tensor):
                v = v.item()
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

    def __str__(self):
        if self.no_log_dict['batch'] is not None:
            batch_str = f" @ batch {self.no_log_dict['batch']}"
        else:
            batch_str = ""
        if 'inv_mnist_test_ll' in self.keys():
            inv_str = f"LL inv mnist test set: {self.mean('inv_mnist_test_ll'):.2f} - "
        else:
            inv_str = ""
        return f"Train Epoch: {self['epoch']} took {time_delta(self['time'])}{batch_str} - " \
               f"NLL loss: {self.mean('ll_loss'):.2f} - " \
               f"LL orig mnist test set: {self.mean('mnist_test_ll'):.2f} - " \
               f"{inv_str}" \
               f"Entropy of GMMs: {self.mean('gmm_ent'):.2f} - " \
               f"Entropy of inner sums: {self.mean('inner_ent'):.2f}|{self.mean('norm_inner_ent'):.2f}% - " \
               f"Entropy of root sum: {self.mean('root_ent'):.2f}|{self.mean('norm_root_ent'):.2f}% - " \
               f"Entropy loss: {self.mean('ent_loss'):.2f}"

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
    parser.add_argument('--results_dir', type=str, default='.',
                        help='The base directory where the directory containing the results will be saved to.')
    parser.add_argument('--dataset_dir', type=str, default='../data',
                        help='The base directory to provide to the PyTorch Dataloader.')
    parser.add_argument('--model_path', type=str,
                        help='Path to the pretrained model. If it is given, '
                             'all other SPN config parameters are ignored.')
    parser.add_argument('--exp_name', type=str, default='stl', help='Experiment name. The results dir will contain it.')
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
    parser.add_argument('--first_layer_sum', action='store_true', help='Make the first layer a sum layer')
    parser.add_argument('--no_ent', action='store_true', help='Don\'t calculate entropy.')
    parser.add_argument('--gmm_ent_factor', '-alpha', type=float, default=0.0,
                        help='Factor for the entropy loss of the Gaussian mixtures at the leaves.')
    parser.add_argument('--inner_sum_ent_factor', '-beta', type=float, default=0.0,
                        help='Factor for the entropy loss of the other sums in the inner layers.')
    parser.add_argument('--root_sum_ent_factor', '-gamma', type=float, default=0.0,
                        help='Factor for the entropy loss of the root sum.')
    parser.add_argument('--adamw', action='store_true', help='Use AdamW optimizer (incorporates weight decay)')
    parser.add_argument('--invert', action='store_true', help='Invert MNIST images')
    args = parser.parse_args()

    if not args.no_ent:
        args.first_layer_sum = True

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

    inspect = args.inspect
    if inspect:
        i = 5
        epoch = '099'
        exp_name = f"mnistgen{i}"
        base_path = os.path.join('mnist_gen_exp', f"results_{exp_name}")
        model_name = f"epoch-{epoch}_{exp_name}"
        path = os.path.join(base_path, 'models', f"{model_name}.pt")
        model = torch.load(path, map_location=torch.device('cpu'))

        exp = 1
        if exp == 1:
            # Here, the choices of the root sum node are overridden and instead all output channels of its children
            # are sampled.
            results_dir = os.path.join(base_path, f'all_root_in_channels_{model_name}')
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            eval_root_sum_override(model, results_dir, torch.device("cpu"), img_size)
        elif exp == 2:
            # Here, the sampling evaluation is redone for all model files in a given directory
            models_dir = os.path.join(base_path, 'models')
            onlyfiles = [f for f in os.listdir(models_dir) if os.path.isfile(os.path.join(models_dir, f))]
            samples_dir = os.path.join(base_path, 'new_samples')
            if not os.path.exists(samples_dir):
                os.makedirs(samples_dir)
            for f in onlyfiles:
                model_path = os.path.join(models_dir, f)
                save_path = os.path.join(samples_dir, f"{f.split('.')[0]}.png")
                model = torch.load(model_path, map_location=torch.device('cpu'))
                evaluate_sampling(model, save_path, torch.device('cpu'), img_size)
        else:
            results_dir = base_path
            def plot_img(image: torch.Tensor, title: str = None, path=None):
                # Tensor shape N x channels x rows x cols
                tensors = torchvision.utils.make_grid(image, nrow=5, padding=1)
                arr = tensors.permute(1, 2, 0).cpu().numpy()
                arr = skimage.img_as_ubyte(arr)
                # imageio.imwrite(path, arr)
                # return
                plt.imshow(arr)
                plt.title(title, fontdict={'fontsize': 10})
                plt.show()

            top_5_ll = torch.ones(5) * -10e6
            top_5_ll_img = torch.zeros(5, *img_size)
            low_5_ll = torch.ones(5) * 10e6
            low_5_ll_img = torch.zeros(5, *img_size)


            show_all = True
            find_top_low_LL = False
            if show_all:
                batch_size = 5
            else:
                batch_size = 256

            # train_loader, test_loader = get_mnist_loaders(args.dataset_dir, args.grayscale, batch_size=batch_size, device=device)
            for cond in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                cond = torch.ones(1000).long() * cond
                cond = F.one_hot(cond, cond_size).float()
                sample = model.sample(condition=cond)
                sample[sample < 0.0] = 0.0
                sample[sample > 1.0] = 1.0
                sample_ll = model(x=sample, condition=None).flatten()
                sample = sample.view(-1, *img_size)

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
                    lls = [[f"{n:.2f}" for n in ll.tolist()] for ll in [sample_ll]]
                    title = f"LLs of samples: [{', '.join(lls[0][:5])},\n{', '.join(lls[0][5:])}]"
                    plot_img(sample, title, path=None)
                print("Set breakpoint here")
        exit()

    csv_log = os.path.join(results_dir, f"log_{args.exp_name}.csv")
    info = CsvLogger(csv_log)
    # Construct Cspn from config
    orig_train_loader, orig_test_loader = get_mnist_loaders(args.dataset_dir, use_cuda, batch_size=batch_size, device=device)
    if args.invert:
        print("Training on inverted MNIST!")
        inv_train_loader, inv_test_loader = get_mnist_loaders(args.dataset_dir, use_cuda,
                                                              batch_size=batch_size, device=device, invert=True)
        train_loader = inv_train_loader
        csv_log = os.path.join(results_dir, f"log_{args.exp_name}.csv")
        info.keys_to_avg = ['inv_mnist_test_ll'] + info.keys_to_avg
        info.reset()
    else:
        inv_train_loader = inv_test_loader = None
        train_loader = orig_train_loader

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
        config.first_layer_sum = args.first_layer_sum
        config.leaf_base_class = RatNormal
        config.leaf_base_kwargs = {'min_sigma': 0.1, 'max_sigma': 1.0, 'min_mean': 0.0, 'max_mean': 1.0}
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
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

    lmbda = 1.0
    sample_interval = 1 if args.verbose else args.eval_interval  # number of epochs
    save_interval = 1 if args.verbose else args.save_interval  # number of epochs

    epoch = 0
    if args.sample_override_root:
        print("Sampling from all input channels to root sum node ...")
        root_sum_override_dir = os.path.join(sample_dir, f"epoch-{epoch:03}_root_sum_override")
        eval_root_sum_override(model, root_sum_override_dir, device, img_size)
    print("Evaluating model ...")
    save_path = os.path.join(sample_dir, f"epoch-{epoch:03}_{args.exp_name}.png")
    evaluate_sampling(model, save_path, device, img_size)
    info.reset(epoch)
    if inv_test_loader is not None:
        info['inv_mnist_test_ll'] = evaluate_model(model, device, inv_test_loader, "Inverted MNIST test")
    info['mnist_test_ll'] = evaluate_model(model, device, orig_test_loader, "Original MNIST test")
    info.average()
    info.write()
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
            ent_loss = leaf_gmm_ent_lb = inner_ents = norm_inner_ents = root_ent = norm_root_ent = torch.zeros(1).to(device)
            if args.ratspn:
                output: torch.Tensor = model(x=data)
                loss_ce = F.cross_entropy(output, label)
                ll_loss = -output.mean()
                loss = (1 - lmbda) * ll_loss + lmbda * loss_ce
            else:
                label = F.one_hot(label, cond_size).float().to(device)
                output: torch.Tensor = model(x=data, condition=label)
                ll_loss = -output.mean()
                if not args.no_ent:
                    with torch.no_grad():
                        leaf_gmm_ent_lb = model.gmm_entropy_lb(reduction='mean')
                        inner_ents, norm_inner_ents, root_ent, norm_root_ent = model.sum_node_entropies(reduction='mean')
                    ent_mix = leaf_gmm_ent_lb * args.gmm_ent_factor + args.inner_sum_ent_factor * norm_inner_ents + \
                              args.root_sum_ent_factor * norm_root_ent
                    ent_loss = -ent_mix
                loss = ll_loss + ent_loss

            loss.backward()
            optimizer.step()
            info.add_to_avg_keys(ll_loss=ll_loss, ent_loss=ent_loss, gmm_ent=leaf_gmm_ent_lb, inner_ent=inner_ents,
                                 norm_inner_ent=norm_inner_ents, root_ent=root_ent, norm_root_ent=norm_root_ent,
                                 loss=loss)

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
            if inv_test_loader is not None:
                info['inv_mnist_test_ll'] = evaluate_model(model, device, inv_test_loader, "Inverted MNIST test")
            info['mnist_test_ll'] = evaluate_model(model, device, orig_test_loader, "Original MNIST test")

        info.average()
        info['time'] = t_delta
        info['batch'] = None
        info.write()
        print(info)



