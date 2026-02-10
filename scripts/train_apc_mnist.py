"""Train an Autoencoding Probabilistic Circuit (APC) on MNIST.

This script provides a lightweight training entrypoint inspired by the
reference APC project main script, but using SPFlow's in-repo APC APIs.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch
from torch import nn
from torch.optim import RMSprop, SGD
from torch.optim import Adam
from torch.optim import AdamW
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import MultiStepLR, OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from spflow.modules.leaves import Bernoulli, Binomial, Normal
from spflow.modules.leaves.leaf import LeafModule
from spflow.zoo.apc.config import ApcConfig, ApcLossWeights
from spflow.zoo.apc.decoders import ConvDecoder2D, MLPDecoder1D, NeuralDecoder2D
from spflow.zoo.apc.debug_trace import configure_trace, trace_tensor, trace_value
from spflow.zoo.apc.encoders.convpc_joint_encoder import ConvPcJointEncoder
from spflow.zoo.apc.encoders.einet_joint_encoder import EinetJointEncoder
from spflow.zoo.apc.model import AutoencodingPC
from spflow.zoo.apc.train import evaluate_apc

if TYPE_CHECKING:
    from lightning import Fabric

LeafFactory = Callable[[list[int], int, int], LeafModule]


class _ReconstructionSummaryWrapper(nn.Module):
    """Wrapper exposing reconstruction forward for architecture summaries."""

    def __init__(self, model: AutoencodingPC) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.reconstruct(x)


def flatten_image_tensor(x: torch.Tensor) -> torch.Tensor:
    """Flatten image tensor from ``(C, H, W)`` to ``(C*H*W,)``."""
    return x.view(-1)


def scale_to_n_bits(x: torch.Tensor, *, n_bits: int) -> torch.Tensor:
    """Quantize tensor to integer counts in ``[0, 2^n_bits - 1]``.

    This mirrors the reference APC preprocessing (`to_255_int` for 8-bit MNIST),
    i.e. multiply then integer-cast.
    """
    max_value = float(2**n_bits - 1)
    return (x * max_value).to(dtype=torch.int32)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train an APC model on MNIST.")

    parser.add_argument(
        "--data-dir", type=Path, default=Path("./data"), help="Directory used for MNIST data."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./results/apc_mnist"),
        help="Root directory where run outputs are stored.",
    )
    parser.add_argument("--download", action="store_true", help="Download MNIST if not present.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=32,
        help="MNIST spatial size. Reference APC setup uses 32 (mnist-32).",
    )
    parser.add_argument(
        "--dist-data",
        type=str,
        default="binomial",
        choices=("normal", "bernoulli", "binomial"),
        help="Data leaf distribution. Reference MNIST APC uses binomial.",
    )
    parser.add_argument(
        "--dist-latent",
        type=str,
        default="normal",
        choices=("normal", "bernoulli", "binomial"),
        help="Latent leaf distribution. Reference MNIST APC uses normal.",
    )
    parser.add_argument(
        "--n-bits",
        type=int,
        default=8,
        help="Quantization bits for binomial leaves (total_count = 2^n_bits - 1).",
    )
    parser.add_argument(
        "--normalize-data",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Whether to normalize pixels to [0, 1]. Default auto: false for "
            "binomial/bernoulli data leaves, true for normal."
        ),
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Debug mode (reference-style): no dataset filtering; useful for tracing and quick runs.",
    )
    parser.add_argument(
        "--trace",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable APC tensor tracing (defaults to enabled when --debug is on).",
    )
    parser.add_argument(
        "--trace-max-events",
        type=int,
        default=800,
        help="Maximum number of trace events to print.",
    )
    parser.add_argument(
        "--trace-max-values",
        type=int,
        default=8,
        help="How many leading values to include in each trace line.",
    )

    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Execution device.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32-true",
        help="Fabric precision mode (e.g. 32-true, 16-mixed, bf16-mixed).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count (default 0 for sandbox-safe portability).",
    )

    parser.add_argument(
        "--iters",
        type=int,
        default=5000,
        help="Training iterations/optimizer steps (reference APC style).",
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Fallback optimizer learning rate.")
    parser.add_argument(
        "--learning-rate-encoder",
        type=float,
        default=0.1,
        help="Encoder learning rate (reference APC MNIST default).",
    )
    parser.add_argument(
        "--learning-rate-decoder",
        type=float,
        default=1e-3,
        help="Decoder learning rate (reference APC MNIST default).",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Optimizer weight decay.")
    parser.add_argument(
        "--optim",
        type=str,
        default="adam",
        choices=("adam", "adamw", "sgd", "rmsprop"),
        help="Optimizer type. Reference APC defaults to Adam.",
    )
    parser.add_argument(
        "--amsgrad",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use AMSGrad variant for Adam/AdamW.",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="multistep",
        choices=("none", "multistep", "onecycle", "plateau"),
        help="Learning-rate scheduler. Reference APC defaults to MultiStepLR.",
    )
    parser.add_argument(
        "--lr-gamma",
        type=float,
        default=0.1,
        help="Gamma used by MultiStepLR.",
    )
    parser.add_argument(
        "--warmup-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable learning-rate warmup.",
    )
    parser.add_argument(
        "--warmup-pct",
        type=float,
        default=2.0,
        help="Warmup length as percentage of total iterations.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=None,
        help="Optional gradient clipping norm.",
    )

    parser.add_argument("--val-size", type=int, default=10_000, help="Validation split size from train set.")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap on train subset size for quick experiments.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Optional cap on validation subset size for quick experiments.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional cap on test subset size for quick experiments.",
    )

    parser.add_argument("--latent-dim", type=int, default=64, help="Latent dimension.")
    parser.add_argument("--num-sums", type=int, default=16, help="Einet sum channels.")
    parser.add_argument("--num-leaves", type=int, default=16, help="Einet leaf channels.")
    parser.add_argument("--depth", type=int, default=6, help="Einet depth.")
    parser.add_argument("--num-repetitions", type=int, default=1, help="Einet repetition count.")
    parser.add_argument(
        "--layer-type",
        type=str,
        default="linsum",
        choices=("einsum", "linsum"),
        help="Einet layer type.",
    )
    parser.add_argument(
        "--structure",
        type=str,
        default="top-down",
        choices=("top-down", "bottom-up"),
        help="Einet structure mode.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="convpc",
        choices=("convpc", "einet"),
        help="APC encoder family. Defaults to reference MNIST Conv-PC.",
    )

    parser.add_argument("--conv-channels", type=int, default=64, help="Conv-PC channels.")
    parser.add_argument("--conv-depth", type=int, default=4, help="Conv-PC depth.")
    parser.add_argument(
        "--conv-latent-depth",
        type=int,
        default=0,
        help="Conv-PC latent injection depth.",
    )
    parser.add_argument(
        "--conv-architecture",
        type=str,
        default="reference",
        choices=("reference", "legacy"),
        help="Conv-PC architecture mode.",
    )
    parser.add_argument(
        "--conv-perm-latents",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable latent permutation packing in reference Conv-PC architecture.",
    )
    parser.add_argument(
        "--conv-use-sum-conv",
        action="store_true",
        help="Use SumConv layers in Conv-PC encoder.",
    )

    parser.add_argument(
        "--decoder-hidden-dims",
        type=int,
        nargs="+",
        default=[64, 16],
        help="MLP decoder hidden dimensions.",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default="auto",
        choices=("auto", "mlp", "conv", "nn"),
        help="Decoder type. 'auto' chooses nn for convpc encoder and mlp for einet.",
    )
    parser.add_argument(
        "--conv-decoder-base-channels", type=int, default=64, help="Conv decoder base channels."
    )
    parser.add_argument("--conv-decoder-upsamples", type=int, default=2, help="Conv decoder upsample blocks.")
    parser.add_argument("--nn-decoder-num-hidden", type=int, default=64, help="NN decoder hidden channels.")
    parser.add_argument(
        "--nn-decoder-num-res-hidden", type=int, default=16, help="NN decoder residual width."
    )
    parser.add_argument(
        "--nn-decoder-num-res-layers", type=int, default=2, help="NN decoder residual blocks."
    )
    parser.add_argument("--nn-decoder-num-scales", type=int, default=2, help="NN decoder upscale depth.")
    parser.add_argument(
        "--nn-decoder-bn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable batch norm inside NN decoder residual blocks.",
    )
    parser.add_argument(
        "--nn-decoder-out-activation",
        type=str,
        default="tanh",
        choices=("identity", "linear", "tanh", "sigmoid"),
        help="NN decoder output activation (reference MNIST default: linear).",
    )
    parser.add_argument(
        "--rec-loss",
        type=str,
        default="mse",
        choices=("mse", "bce"),
        help="Reconstruction term type.",
    )
    parser.add_argument("--sample-tau", type=float, default=1.0, help="Differentiable sampling temperature.")
    parser.add_argument("--w-rec", type=float, default=1.0, help="Weight for reconstruction term.")
    parser.add_argument("--w-kld", type=float, default=1.0, help="Weight for KL term.")
    parser.add_argument("--w-nll", type=float, default=1.0, help="Weight for NLL term.")
    parser.add_argument(
        "--num-vis",
        type=int,
        default=16,
        help="Number of examples to visualize (top row data, bottom row reconstructions).",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=500,
        help="Validation interval in iterations. Set <=0 to validate only at the end.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Train-metric logging interval in iterations.",
    )

    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    """Resolve the desired torch device."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_fabric(*, device: torch.device, precision: str) -> "Fabric":
    """Create a Lightning Fabric runtime matching the selected device."""
    try:
        import lightning as L
    except ImportError as exc:
        raise RuntimeError(
            "lightning is required for this training script. Install it with `pip install lightning`."
        ) from exc

    accelerator = "cuda" if device.type == "cuda" else "cpu"
    return L.Fabric(accelerator=accelerator, devices=1, precision=precision)


def seed_everything(seed: int) -> None:
    """Seed torch RNGs for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cap_subset(subset: torch.utils.data.Subset, max_samples: int | None) -> torch.utils.data.Subset:
    """Return a capped subset if ``max_samples`` is provided."""
    if max_samples is None or max_samples >= len(subset):
        return subset
    indices = subset.indices[:max_samples]
    return torch.utils.data.Subset(subset.dataset, indices)


def _subset_mnist_by_classes(
    dataset: torch.utils.data.Dataset, classes: list[int]
) -> torch.utils.data.Subset:
    """Filter a torchvision MNIST dataset to a target class set."""
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise RuntimeError("MNIST dataset does not expose 'targets'; cannot apply class filtering.")
    if not isinstance(targets, torch.Tensor):
        targets = torch.as_tensor(targets)

    keep_classes = torch.as_tensor(classes, dtype=targets.dtype, device=targets.device)
    keep_mask = (targets.unsqueeze(1) == keep_classes.unsqueeze(0)).any(dim=1)
    keep_indices = keep_mask.nonzero(as_tuple=True)[0].tolist()
    if len(keep_indices) == 0:
        raise RuntimeError(f"Class filtering kept zero samples for classes={classes}.")
    return torch.utils.data.Subset(dataset, keep_indices)


def build_mnist_loaders(
    args: argparse.Namespace, *, flatten: bool, normalize_data: bool
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/validation/test DataLoaders for MNIST."""
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError(
            "torchvision is required for MNIST training script. Install it in the environment first."
        ) from exc

    transforms_list: list[Any] = [
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ]
    if normalize_data:
        transforms_list.append(transforms.Normalize([0.5], [0.5]))
    else:
        transforms_list.append(transforms.Lambda(lambda tensor: scale_to_n_bits(tensor, n_bits=args.n_bits)))

    if flatten:
        transforms_list.append(transforms.Lambda(flatten_image_tensor))

    transform = transforms.Compose(transforms_list)

    train_full = datasets.MNIST(
        root=str(args.data_dir),
        train=True,
        transform=transform,
        download=args.download,
    )
    test_full = datasets.MNIST(
        root=str(args.data_dir),
        train=False,
        transform=transform,
        download=args.download,
    )

    val_size = args.val_size
    if val_size <= 0:
        raise ValueError(f"--val-size must be >= 1, got {val_size}.")
    if val_size >= len(train_full):
        raise ValueError(f"--val-size must be in [1, {len(train_full) - 1}], got {val_size}.")

    split_gen = torch.Generator()
    split_gen.manual_seed(args.seed)
    train_subset, val_subset = random_split(
        train_full,
        [len(train_full) - val_size, val_size],
        generator=split_gen,
    )

    train_subset = _cap_subset(train_subset, args.max_train_samples)
    val_subset = _cap_subset(val_subset, args.max_val_samples)
    test_subset = _cap_subset(
        torch.utils.data.Subset(test_full, list(range(len(test_full)))), args.max_test_samples
    )

    pin_memory = torch.cuda.is_available() and resolve_device(args.device).type == "cuda"

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def make_leaf_factory(dist: str, *, total_count: float) -> LeafFactory:
    """Create a leaf factory for APC encoder blocks."""
    if dist == "normal":
        def _normal_factory(scope_indices: list[int], out_channels: int, num_repetitions: int) -> LeafModule:
            # Reference APC initializes Normal leaves via (mu, logvar), not via
            # uniformly sampled scales.
            event_shape = (len(scope_indices), out_channels, num_repetitions)
            loc = torch.randn(event_shape)
            logvar = torch.randn(event_shape)
            scale = torch.exp(0.5 * logvar)
            return Normal(
                scope=scope_indices,
                out_channels=out_channels,
                num_repetitions=num_repetitions,
                loc=loc,
                scale=scale,
            )

        return _normal_factory
    if dist == "bernoulli":
        return lambda scope_indices, out_channels, num_repetitions: Bernoulli(
            scope=scope_indices,
            out_channels=out_channels,
            num_repetitions=num_repetitions,
        )
    if dist == "binomial":
        total_count_tensor = torch.tensor(float(total_count))
        def _binomial_factory(scope_indices: list[int], out_channels: int, num_repetitions: int) -> LeafModule:
            # Match reference APC Binomial init: p ~ Uniform(0.4, 0.6).
            event_shape = (len(scope_indices), out_channels, num_repetitions)
            probs = 0.5 + (torch.rand(event_shape) - 0.5) * 0.2
            return Binomial(
                scope=scope_indices,
                out_channels=out_channels,
                num_repetitions=num_repetitions,
                total_count=total_count_tensor,
                probs=probs,
            )

        return _binomial_factory
    raise ValueError(f"Unsupported leaf distribution '{dist}'.")


def build_model(args: argparse.Namespace) -> AutoencodingPC:
    """Build APC model components for MNIST."""
    num_x_features = args.image_size * args.image_size
    use_conv_encoder = args.encoder == "convpc"
    total_count = float(2**args.n_bits - 1)
    x_leaf_factory = make_leaf_factory(args.dist_data, total_count=total_count)
    z_leaf_factory = make_leaf_factory(args.dist_latent, total_count=total_count)

    if use_conv_encoder:
        encoder = ConvPcJointEncoder(
            input_height=args.image_size,
            input_width=args.image_size,
            input_channels=1,
            latent_dim=args.latent_dim,
            channels=args.conv_channels,
            depth=args.conv_depth,
            kernel_size=2,
            num_repetitions=args.num_repetitions,
            use_sum_conv=args.conv_use_sum_conv,
            latent_depth=args.conv_latent_depth,
            architecture=args.conv_architecture,
            perm_latents=args.conv_perm_latents,
            x_leaf_factory=x_leaf_factory,
            z_leaf_factory=z_leaf_factory,
        )
    else:
        encoder = EinetJointEncoder(
            num_x_features=num_x_features,
            latent_dim=args.latent_dim,
            num_sums=args.num_sums,
            num_leaves=args.num_leaves,
            depth=args.depth,
            num_repetitions=args.num_repetitions,
            layer_type=args.layer_type,
            structure=args.structure,
            x_leaf_factory=x_leaf_factory,
            z_leaf_factory=z_leaf_factory,
        )

    if args.decoder == "auto":
        decoder_kind = "nn" if use_conv_encoder else "mlp"
    else:
        decoder_kind = args.decoder

    if decoder_kind == "conv":
        decoder = ConvDecoder2D(
            latent_dim=args.latent_dim,
            output_shape=(1, args.image_size, args.image_size),
            base_channels=args.conv_decoder_base_channels,
            num_upsamples=args.conv_decoder_upsamples,
            out_activation="sigmoid" if args.rec_loss == "bce" else "identity",
        )
    elif decoder_kind == "nn":
        decoder = NeuralDecoder2D(
            latent_dim=args.latent_dim,
            output_shape=(1, args.image_size, args.image_size),
            num_hidden=args.nn_decoder_num_hidden,
            num_res_hidden=args.nn_decoder_num_res_hidden,
            num_res_layers=args.nn_decoder_num_res_layers,
            num_scales=args.nn_decoder_num_scales,
            bn=args.nn_decoder_bn,
            out_activation=args.nn_decoder_out_activation,
        )
    else:
        decoder = MLPDecoder1D(
            latent_dim=args.latent_dim,
            output_dim=num_x_features,
            hidden_dims=tuple(args.decoder_hidden_dims),
            out_activation="sigmoid" if args.rec_loss == "bce" else "identity",
        )

    config = ApcConfig(
        latent_dim=args.latent_dim,
        rec_loss=args.rec_loss,
        sample_tau=args.sample_tau,
        loss_weights=ApcLossWeights(
            rec=args.w_rec,
            kld=args.w_kld,
            nll=args.w_nll,
        ),
    )
    return AutoencodingPC(encoder=encoder, decoder=decoder, config=config)


def build_optimizer(args: argparse.Namespace, model: AutoencodingPC) -> Optimizer:
    """Build optimizer with parameter groups for encoder/decoder."""
    param_groups = [
        {"params": model.encoder.parameters(), "lr": args.learning_rate_encoder},
    ]
    if model.decoder is not None:
        param_groups.append({"params": model.decoder.parameters(), "lr": args.learning_rate_decoder})

    if args.optim == "adam":
        return Adam(param_groups, weight_decay=args.weight_decay, amsgrad=args.amsgrad)
    if args.optim == "adamw":
        return AdamW(param_groups, weight_decay=args.weight_decay, amsgrad=args.amsgrad)
    if args.optim == "sgd":
        return SGD(param_groups, weight_decay=args.weight_decay)
    if args.optim == "rmsprop":
        return RMSprop(param_groups, weight_decay=args.weight_decay)
    raise ValueError(f"Unsupported optimizer '{args.optim}'.")


def build_scheduler(args: argparse.Namespace, optimizer: Optimizer) -> LRScheduler | ReduceLROnPlateau | None:
    """Build optional LR scheduler from CLI arguments."""
    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler == "multistep":
        milestones = sorted({max(1, int(0.66 * args.iters)), max(1, int(0.9 * args.iters))})
        return MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_gamma)
    if args.lr_scheduler == "onecycle":
        max_lr = [group["lr"] for group in optimizer.param_groups]
        return OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=args.iters + 1,
            div_factor=25.0,
            final_div_factor=1e4,
        )
    if args.lr_scheduler == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=max(1, int(0.05 * args.iters)),
            min_lr=args.learning_rate / 1000.0,
        )
    raise ValueError(f"Unsupported lr scheduler '{args.lr_scheduler}'.")


def _to_serializable(obj: Any) -> Any:
    """Convert objects to JSON-serializable forms."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    return obj


def _extract_batch_tensor(batch: torch.Tensor | tuple | list) -> torch.Tensor:
    """Extract the input tensor from a dataset batch."""
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, (tuple, list)) and len(batch) > 0 and isinstance(batch[0], torch.Tensor):
        return batch[0]
    raise TypeError(f"Unsupported batch type: {type(batch)}")


def _to_image_batch(x: torch.Tensor, *, image_size: int) -> torch.Tensor:
    """Convert MNIST data to image batch ``(B, 1, H, W)`` for visualization."""
    if x.dim() == 4 and x.shape[1:] == (1, image_size, image_size):
        return x
    if x.dim() == 2 and x.shape[1] == image_size * image_size:
        return x.view(-1, 1, image_size, image_size)
    raise ValueError(
        f"Expected MNIST batch as (B, {image_size * image_size}) or "
        f"(B, 1, {image_size}, {image_size}), got {tuple(x.shape)}."
    )


def save_reconstruction_visualization(
    *,
    model: AutoencodingPC,
    test_loader: DataLoader,
    device: torch.device,
    output_path: Path,
    num_vis: int,
    image_size: int,
) -> None:
    """Save a two-row reconstruction grid (data/reconstruction)."""
    if num_vis <= 0:
        raise ValueError(f"--num-vis must be >= 1, got {num_vis}.")

    try:
        from torchvision.utils import make_grid, save_image
    except ImportError as exc:
        raise RuntimeError("torchvision is required to save reconstruction visualizations.") from exc

    first_batch = next(iter(test_loader))
    x_batch = _extract_batch_tensor(first_batch).to(device)
    x_batch = x_batch[: min(num_vis, x_batch.shape[0])]

    model.eval()
    with torch.no_grad():
        x_rec = model.reconstruct(x_batch)
    x_batch_float = x_batch.float()
    print(
        "[APC] Recon stats: "
        f"data(min={x_batch_float.min().item():.6f}, max={x_batch_float.max().item():.6f}, mean={x_batch_float.mean().item():.6f}), "
        f"recon(min={x_rec.min().item():.6f}, max={x_rec.max().item():.6f}, mean={x_rec.mean().item():.6f})"
    )

    x_img = _to_image_batch(x_batch.detach().cpu(), image_size=image_size)
    x_rec_img = _to_image_batch(x_rec.detach().cpu(), image_size=image_size)
    grid = torch.cat([x_img, x_rec_img], dim=0)
    grid = make_grid(
        grid.float(),
        nrow=x_img.shape[0],
        normalize=True,
        padding=1,
        pad_value=1.0,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, fp=str(output_path))


def _human_readable_number(number: int) -> str:
    """Convert a count to human-readable ``K/M/B`` format."""
    suffixes = ["", "K", "M", "B", "T"]
    value = float(number)
    suffix_index = 0
    while value >= 1000.0 and suffix_index < len(suffixes) - 1:
        value /= 1000.0
        suffix_index += 1
    return f"{value:.2f}{suffixes[suffix_index]}"


def _print_param_counts(model: AutoencodingPC) -> None:
    """Print parameter counts in the same style as the reference script."""
    num_params_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params_encoder = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    num_params_decoder = (
        sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
        if model.decoder is not None
        else 0
    )
    print(f"[APC] Number of parameters Total:   {_human_readable_number(num_params_total)}")
    print(f"[APC] Number of parameters Encoder: {_human_readable_number(num_params_encoder)}")
    print(f"[APC] Number of parameters Decoder: {_human_readable_number(num_params_decoder)}")


def _print_model_summary(model: AutoencodingPC, val_loader: DataLoader, device: torch.device) -> None:
    """Print model architecture and layer summary using torchinfo (reference style)."""
    print("[APC] Model:")
    print(model)

    first_batch = next(iter(val_loader))
    summary_input = _extract_batch_tensor(first_batch)[:20].to(device)
    summary_model = _ReconstructionSummaryWrapper(model).to(device)
    summary_model.eval()

    try:
        from torchinfo import summary as torchinfo_summary

        model_summary = torchinfo_summary(
            summary_model,
            input_data=summary_input,
            depth=3,
            col_names=["input_size", "output_size", "num_params", "params_percent"],
            verbose=0,
        )
        print(model_summary)
        return
    except ImportError:
        pass
    except Exception as exc:
        print(f"[APC] torchinfo summary failed: {exc}")
        return

    try:
        from torchsummary import summary as torchsummary_summary
    except ImportError:
        print(
            "[APC] Model summary skipped (install torchinfo to match reference output, "
            "or install torchsummary as fallback)."
        )
        return

    print("[APC] torchinfo not found; using torchsummary fallback.")
    input_size = tuple(summary_input.shape[1:])
    torchsummary_summary(summary_model, input_size=input_size, device=device.type)


def _unwrap_fabric_module(model: nn.Module) -> nn.Module:
    """Return the underlying module when wrapped by Fabric."""
    inner = getattr(model, "module", None)
    return inner if isinstance(inner, nn.Module) else model


def train_apc_iters(
    *,
    fabric: "Fabric",
    model: AutoencodingPC,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    iters: int,
    batch_size: int,
    grad_clip_norm: float | None,
    val_every: int,
    log_every: int,
    scheduler: LRScheduler | ReduceLROnPlateau | None,
    warmup_enabled: bool,
    warmup_steps: int,
) -> list[dict[str, float]]:
    """Train APC for a fixed number of optimization iterations."""
    if iters <= 0:
        raise ValueError(f"--iters must be >= 1, got {iters}.")

    history: list[dict[str, float]] = []
    train_iter = iter(train_loader)

    window_totals = {"rec": 0.0, "kld": 0.0, "nll": 0.0, "total": 0.0}
    window_steps = 0

    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}.")

    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    progress = tqdm(range(1, iters + 1), desc="[APC] Training", unit="iter")
    for step in progress:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        model.train()
        x = _extract_batch_tensor(batch).to(fabric.device)

        optimizer.zero_grad(set_to_none=True)
        step_losses = model.loss_components(x)
        total = step_losses["total"]
        fabric.backward(total)

        if grad_clip_norm is not None:
            fabric.clip_gradients(model, optimizer, max_norm=grad_clip_norm)

        optimizer.step()
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(float(step_losses["rec"].item()))
            else:
                scheduler.step()

        if warmup_enabled and warmup_steps > 0 and step <= warmup_steps:
            factor = math.exp(-5.0 * (1.0 - (step / warmup_steps)) ** 2)
            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base_lr * factor

        current_lr = float(optimizer.param_groups[0]["lr"])
        for key in window_totals:
            window_totals[key] += float(step_losses[key].item())
        window_steps += 1
        progress.set_postfix(
            {
                "loss": f"{float(step_losses['total'].item()):.4f}",
                "rec": f"{float(step_losses['rec'].item()):.4f}",
                "kld": f"{float(step_losses['kld'].item()):.4f}",
                "nll": f"{float(step_losses['nll'].item()):.4f}",
                "lr": f"{current_lr:.3e}",
            }
        )

        if log_every > 0 and step % log_every == 0:
            print(
                "[APC] Iter {step}: train_total={total:.4f}, train_rec={rec:.4f}, "
                "train_kld={kld:.4f}, train_nll={nll:.4f}".format(
                    step=step,
                    total=window_totals["total"] / window_steps,
                    rec=window_totals["rec"] / window_steps,
                    kld=window_totals["kld"] / window_steps,
                    nll=window_totals["nll"] / window_steps,
                )
            )

        should_validate = (val_every > 0 and step % val_every == 0) or step == iters
        if not should_validate:
            continue

        train_means = {f"train_{k}": v / window_steps for k, v in window_totals.items()}
        val_metrics = evaluate_apc(model, val_loader, batch_size=batch_size)
        entry: dict[str, float] = {"iter": float(step), **train_means}
        entry.update({f"val_{k}": float(v) for k, v in val_metrics.items()})
        history.append(entry)

        print(
            "[APC] Eval @iter {iter:.0f}: train_total={train_total:.4f}, val_total={val_total:.4f}, "
            "train_rec={train_rec:.4f}, train_kld={train_kld:.4f}, train_nll={train_nll:.4f}".format(**entry)
        )

        window_totals = {"rec": 0.0, "kld": 0.0, "nll": 0.0, "total": 0.0}
        window_steps = 0

    return history


def main() -> None:
    """Run APC training on MNIST and persist artifacts."""
    raise RuntimeError(
        "APC KL-style training is unavailable after the differentiable-sampling rollback."
    )
    args = parse_args()
    seed_everything(args.seed)

    if args.normalize_data is None:
        normalize_data = args.dist_data == "normal"
    else:
        normalize_data = args.normalize_data

    trace_enabled = args.debug if args.trace is None else bool(args.trace)
    configure_trace(
        enabled=trace_enabled,
        prefix="SPFLOW",
        max_events=args.trace_max_events,
        max_values=args.trace_max_values,
    )
    trace_value("trace.enabled", trace_enabled)

    device = resolve_device(args.device)
    fabric = build_fabric(device=device, precision=args.precision)
    model = build_model(args)
    train_loader, val_loader, test_loader = build_mnist_loaders(
        args,
        flatten=(args.encoder == "einet"),
        normalize_data=normalize_data,
    )

    effective_iters = args.iters
    optimizer = build_optimizer(args, model)
    warmup_steps = int(args.iters * args.warmup_pct / 100.0)

    print(f"[APC] Device: {device}")
    print(
        f"[APC] Dataset sizes: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, "
        f"test={len(test_loader.dataset)}"
    )
    print(
        "[APC] Config: "
        f"encoder={args.encoder}, decoder={args.decoder}, latent_dim={args.latent_dim}, "
        f"image_size={args.image_size}, batch_size={args.batch_size}, iters={effective_iters}, rec_loss={args.rec_loss}, "
        f"dist_data={args.dist_data}, dist_latent={args.dist_latent}, "
        f"normalize_data={normalize_data}, debug={args.debug}, "
        f"optim={args.optim}, lr_scheduler={args.lr_scheduler}, "
        f"lr_encoder={args.learning_rate_encoder}, lr_decoder={args.learning_rate_decoder}, "
        f"warmup_enabled={args.warmup_enabled}, warmup_steps={warmup_steps}, precision={args.precision}"
    )
    if args.encoder == "convpc":
        print(
            "[APC] Conv-PC: "
            f"channels={args.conv_channels}, depth={args.conv_depth}, "
            f"latent_depth={args.conv_latent_depth}, use_sum_conv={args.conv_use_sum_conv}, "
            f"architecture={args.conv_architecture}, perm_latents={args.conv_perm_latents}"
        )
    else:
        print(
            "[APC] Einet: "
            f"depth={args.depth}, num_sums={args.num_sums}, num_leaves={args.num_leaves}, "
            f"layer_type={args.layer_type}, structure={args.structure}"
        )
    if args.decoder in {"auto", "nn"} and (args.decoder == "nn" or args.encoder == "convpc"):
        print(
            "[APC] NN decoder: "
            f"num_hidden={args.nn_decoder_num_hidden}, num_res_hidden={args.nn_decoder_num_res_hidden}, "
            f"num_res_layers={args.nn_decoder_num_res_layers}, num_scales={args.nn_decoder_num_scales}, "
            f"bn={args.nn_decoder_bn}, out_activation={args.nn_decoder_out_activation}"
        )
    _print_param_counts(model)
    _print_model_summary(model, val_loader, device)

    train_loader, val_loader, test_loader = fabric.setup_dataloaders(train_loader, val_loader, test_loader)
    first_batch = next(iter(train_loader))
    first_x = _extract_batch_tensor(first_batch)
    trace_tensor("main.first_train_batch.x", first_x)

    model, optimizer = fabric.setup(model, optimizer)
    scheduler = build_scheduler(args, optimizer)

    history = train_apc_iters(
        fabric=fabric,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        iters=effective_iters,
        batch_size=args.batch_size,
        grad_clip_norm=args.grad_clip_norm,
        val_every=args.val_every,
        log_every=args.log_every,
        scheduler=scheduler,
        warmup_enabled=args.warmup_enabled,
        warmup_steps=warmup_steps,
    )

    test_metrics = evaluate_apc(model, test_loader, batch_size=args.batch_size)
    print("[APC] Test: total={total:.4f}, rec={rec:.4f}, kld={kld:.4f}, nll={nll:.4f}".format(**test_metrics))

    model_unwrapped = _unwrap_fabric_module(model)
    if not isinstance(model_unwrapped, AutoencodingPC):
        raise RuntimeError(
            f"Expected Fabric-wrapped module to be AutoencodingPC, got {type(model_unwrapped)}."
        )

    run_dir = args.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    args_dict = _to_serializable(vars(args))
    config_payload = {
        "script_args": args_dict,
        "apc_config": asdict(model_unwrapped.config),
        "train_config": {
            "iters": effective_iters,
            "batch_size": args.batch_size,
            "learning_rate_encoder": args.learning_rate_encoder,
            "learning_rate_decoder": args.learning_rate_decoder,
            "optim": args.optim,
            "lr_scheduler": args.lr_scheduler,
            "warmup_enabled": args.warmup_enabled,
            "warmup_steps": warmup_steps,
            "weight_decay": args.weight_decay,
            "grad_clip_norm": args.grad_clip_norm,
            "val_every": args.val_every,
            "log_every": args.log_every,
        },
        "device": str(device),
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)
    with (run_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with (run_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "apc_config": asdict(model_unwrapped.config),
        "script_args": args_dict,
    }
    fabric.save(str(run_dir / "model.pt"), checkpoint)
    recon_path = run_dir / "reconstructions.png"
    save_reconstruction_visualization(
        model=model_unwrapped,
        test_loader=test_loader,
        device=device,
        output_path=recon_path,
        num_vis=args.num_vis,
        image_size=args.image_size,
    )

    print(f"[APC] Saved artifacts to {run_dir}")
    print(f"[APC] Saved reconstruction visualization to {recon_path}")


if __name__ == "__main__":
    main()
