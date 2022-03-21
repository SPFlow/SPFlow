import math
import os
import imageio
import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import skimage
from experiments.train_cspn_mnist_gen import evaluate_sampling
from cspn import CSPN
from utils import SamplingContext
import matplotlib.pyplot as plt


def plot_img(image: th.Tensor, nrow: int, title: str = None, x_label: str = None, y_label: str = None,
             normalize: bool = False, save_dir: str = None):
    # Tensor shape N x channels x rows x cols
    tensors = torchvision.utils.make_grid(image, nrow=nrow, padding=1, normalize=normalize)
    arr = tensors.T.cpu().numpy()
    # arr = tensors.permute(1, 2, 0).cpu().numpy()
    arr = skimage.img_as_ubyte(arr)
    plt.imshow(arr)
    if title:
        plt.title(title, fontdict={'fontsize': 10})
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{title.replace(' ', '_').lower()}.png"))
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-dev', type=str, default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()

    if args.device == "cpu":
        device = th.device("cpu")
        use_cuda = False
    else:
        device = th.device("cuda:0")
        use_cuda = True
        th.cuda.benchmark = True
    print("Using device:", device)

    img_size = (1, 28, 28)  # 3 channels
    cond_size = 10

    # exp_name = f"learn_by_sampling_1_low_lr"
    exp_name = f"learn_by_sampling_large"
    epoch = '049'
    base_path = "."
    path_parts = ['..', '..', 'spn_experiments', 'sample_learning_leaf_ent_closed_form', f"results_{exp_name}"]
    for part in path_parts:
        base_path = os.path.join(base_path, part)
        assert os.path.exists(base_path), f"Path {base_path} doesn't exist!"

    model_name = f"epoch-{epoch}_{exp_name}"
    path = os.path.join(base_path, 'models', f"{model_name}.pt")
    model: CSPN = th.load(path, map_location=device)

    def means_of_RVs(model, save_dir: str = None):
        with th.no_grad():
            label = th.as_tensor(np.arange(10)).to(device)
            label = F.one_hot(label, 10).float().to(device)
            model.set_weights(label)
            means = model._leaf.base_leaf.means
            stds = model._leaf.base_leaf.stds
            w, f, oc, r = means.shape
            inv_perm = model.inv_permutation.unsqueeze(1).unsqueeze(0)
            inv_perm = inv_perm.expand(label.size(0), -1, means.size(2), -1)
            means = th.gather(means, dim=1, index=inv_perm)
            stds = th.gather(stds, dim=1, index=inv_perm)
            means = means.tanh().add_(1.0).div_(2.0)

            means_to_plot = means.permute(0, 2, 1, 3).reshape(w * oc, np.sqrt(f).astype(int), np.sqrt(f).astype(int), r)
            means_to_plot = means_to_plot.permute(0, 2, 1, 3)
            stds_to_plot = stds.permute(0, 2, 1, 3).reshape(w * oc, np.sqrt(f).astype(int), np.sqrt(f).astype(int), r)
            stds_to_plot = stds_to_plot.permute(0, 2, 1, 3)
            plot_img(
                means_to_plot.max(dim=-1).values.unsqueeze(1),
                nrow=oc, title=f"Max of means", x_label="Digits 0-9", y_label="Output channels",
                save_dir=save_dir,
            )
            plot_img(
                means_to_plot.min(dim=-1).values.unsqueeze(1),
                nrow=oc, title=f"Min of means", x_label="Digits 0-9", y_label="Output channels",
                save_dir=save_dir,
            )
            for i in range(r):
                if False:
                    plot_img(
                        means_to_plot[:, :, :, i].unsqueeze(1),
                        nrow=oc, title=f"Means of repetition {i}", x_label="Digits 0-9", y_label="Output channels",
                        save_dir=save_dir,
                    )
                plot_img(
                    stds_to_plot[:, :, :, i].unsqueeze(1),
                    nrow=oc, title=f"Stds of repetition {i}", x_label="Digits 0-9", y_label="Output channels",
                    save_dir=save_dir,
                    normalize=True,
                )

    def eval_root_sum_override(model, save_dir, device):
        with th.no_grad():
            label = th.as_tensor(np.arange(10)).to(device)
            label = F.one_hot(label, 10).float().to(device)
            samples = model.sample_index_style(condition=label, start_at_layer=1)
            samples.add_(1.0).div_(2.0).squeeze_(1)
            nr_nodes, w, f, r = samples.shape
            samples = samples.view(nr_nodes * w, np.sqrt(f).astype(int), np.sqrt(f).astype(int), r)
            samples = samples.permute(0, 2, 1, 3)
            for i in range(model.config.R):
                plot_img(
                    samples[:, :, :, i].unsqueeze(1),
                    nrow=w,
                    title=f"Samples of repetition {i}",
                    x_label="Product nodes in this rep",
                    y_label="Digits 0-9",
                    save_dir=save_dir,
                )

    exp = -2
    if exp == -2:
        results_dir = os.path.join(base_path, f'means_and_stds_{model_name}')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        means_of_RVs(model, results_dir)
    elif exp == -1:
        # Sample grad test
        label = th.as_tensor(np.arange(10)).to(device)
        label = F.one_hot(label, 10).float().to(device)
        # sample = model.sample_index_style(condition=label, is_mpe=False)
        sample = model.sample_onehot_style(condition=label, is_mpe=False)
        sample.mean().backward()
        print(f"dist std head weight grads max {model.dist_std_head.weight.grad.max()}")
        print(f"dist std head bias grads max {model.dist_std_head.bias.grad.max()}")
        print(
            f"sum param heads weight grads max {[lay.weight.grad.max() if lay.weight.grad is not None else 0 for lay in model.sum_param_heads]}")
        print(
            f"sum param heads bias grads max {[lay.bias.grad.max() if lay.bias.grad is not None else 0 for lay in model.sum_param_heads]}")
        for lay in model.dist_layers:
            if isinstance(lay, nn.Linear):
                print(f"Dist layer weight grads max {lay.weight.grad.max()}")
                print(f"Dist layer bias grads max {lay.bias.grad.max()}")
        for lay in model.sum_layers:
            if isinstance(lay, nn.Linear):
                print(
                    f"Sum layer weight grads max {lay.weight.grad.max() if lay.weight.grad is not None else 0}")
                print(f"Sum layer bias grads max {lay.bias.grad.max() if lay.bias.grad is not None else 0}")
        for lay in model.feat_layers:
            if isinstance(lay, nn.Linear):
                print(f"Feat layer weight grads max {lay.weight.grad.max()}")
                print(f"Feat layer bias grads max {lay.bias.grad.max()}")
        print('done')

    elif exp == 0:
        samples_dir = os.path.join(base_path, 'new_samples')
        if not os.path.exists(samples_dir):
            os.makedirs(samples_dir)
        save_path = os.path.join(samples_dir, f"sample_mpe_index_style.png")
        evaluate_sampling(model, save_path, device, img_size, mpe=True, style='index')
        save_path = os.path.join(samples_dir, f"sample_mpe_onehot_style.png")
        evaluate_sampling(model, save_path, device, img_size, mpe=True, style='onehot')
        save_path = os.path.join(samples_dir, f"sample_index_style.png")
        evaluate_sampling(model, save_path, device, img_size, style='index')
        save_path = os.path.join(samples_dir, f"sample_onehot_style.png")
        evaluate_sampling(model, save_path, device, img_size, style='onehot')
        print(1)
    elif exp == 1:
        # Here, the choices of the root sum node are overridden and instead all output channels of its children
        # are sampled.
        results_dir = os.path.join(base_path, f'all_root_in_channels_{model_name}')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        eval_root_sum_override(model, results_dir, th.device("cpu"))
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
            model = th.load(model_path, map_location=th.device('cpu'))
            evaluate_sampling(model, save_path, th.device('cpu'), img_size)
    else:
        results_dir = base_path

        top_5_ll = th.ones(5) * -10e6
        top_5_ll_img = th.zeros(5, *img_size)
        low_5_ll = th.ones(5) * 10e6
        low_5_ll_img = th.zeros(5, *img_size)

        show_all = True
        find_top_low_LL = False
        if show_all:
            batch_size = 5
        else:
            batch_size = 256

        # train_loader, test_loader = get_mnist_loaders(args.dataset_dir, args.grayscale, batch_size=batch_size, device=device)
        for cond in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            cond = th.ones(1000).long() * cond
            cond = F.one_hot(cond, cond_size).float()
            sample = model.sample(condition=cond)
            sample[sample < 0.0] = 0.0
            sample[sample > 1.0] = 1.0
            sample_ll = model(x=sample, condition=None).flatten()
            sample = sample.view(-1, *img_size)

            if find_top_low_LL:
                top_5_ll, indices = th.cat((top_5_ll, sample_ll), dim=0).sort(descending=True)
                top_5_ll = top_5_ll[:5]
                indices = indices[:5]
                imgs = []
                for ind in indices:
                    img = top_5_ll_img[ind] if ind < 5 else cond[ind-5]
                    imgs.append(img.unsqueeze(0))
                    top_5_ll_img = th.cat(imgs, dim=0)

                low_5_ll, indices = th.cat((low_5_ll, sample_ll), dim=0).sort(descending=False)
                low_5_ll = low_5_ll[:5]
                indices = indices[:5]
                imgs = []
                for ind in indices:
                    img = low_5_ll_img[ind] if ind < 5 else cond[ind-5]
                    imgs.append(img.unsqueeze(0))
                    low_5_ll_img = th.cat(imgs, dim=0)

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
