"""Learning continuous mixtures of tractable probabilistic models.

Implements the core idea of:

    "Continuous Mixtures of Tractable Probabilistic Models"
    Correia et al., 2023

We learn a decoder network φ(z) that maps low-dimensional latent variables z to
parameters of a tractable model (either fully factorized or a Chow–Liu tree).
The marginal p(x) = E[p(x | φ(z))] is approximated with numerical integration
(Sobol-RQMC) and trained by maximizing the approximate log-likelihood.

The trained continuous mixture can be *compiled* into a standard SPFlow module
by fixing a set of integration points and returning a discrete mixture (Sum) of
tractable components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from spflow.exceptions import InvalidParameterError, UnsupportedOperationError
from spflow.meta.data.scope import Scope
from spflow.modules.leaves import Bernoulli, Categorical, CLTree, Normal
from spflow.modules.products.product import Product
from spflow.modules.sums.sum import Sum
from spflow.zoo.cms.joint import JointLogLikelihood
from spflow.zoo.cms.rqmc import rqmc_sobol_normal

FactorizedLeaf = Literal["bernoulli", "categorical", "normal"]
CltLeaf = Literal["bernoulli", "categorical"]


@dataclass(frozen=True)
class LatentOptimizationConfig:
    """Configuration for latent optimization (LO).

    LO optimizes integration points z after training, keeping the decoder fixed.
    """

    enabled: bool = True
    num_points: int = 32
    num_epochs: int = 150
    batch_size: int = 256
    lr: float = 1e-2
    patience: int = 10
    seed: int = 0


def _to_device_dtype(
    data: Tensor,
    *,
    device: torch.device | None,
    dtype: torch.dtype | None,
) -> Tensor:
    if device is not None:
        data = data.to(device=device)
    if dtype is not None:
        data = data.to(dtype=dtype)
    return data


def _iter_minibatches(data: Tensor, *, batch_size: int, generator: torch.Generator) -> Tensor:
    num_rows = int(data.shape[0])
    if batch_size >= num_rows:
        yield data
        return
    perm = torch.randperm(num_rows, generator=generator, device=data.device)
    for start in range(0, num_rows, batch_size):
        idx = perm[start : start + batch_size]
        yield data[idx]


def _make_sum_weights(
    *,
    num_components: int,
    num_features: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    w = torch.full(
        (num_features, num_components, 1, 1), 1.0 / float(num_components), device=device, dtype=dtype
    )
    return w


def _broadcast_component_weights(*, weights: Tensor, num_features: int) -> Tensor:
    """Broadcast a 1D mixture weight vector to Sum's expected weight tensor shape."""
    if weights.dim() != 1:
        raise InvalidParameterError("weights must be 1D.")
    if not torch.all(weights > 0):
        raise InvalidParameterError("weights must be strictly positive.")
    if not torch.allclose(weights.sum(), weights.new_tensor(1.0), atol=1e-6, rtol=0.0):
        raise InvalidParameterError("weights must sum to 1.")
    I = int(weights.shape[0])
    w = weights.view(1, I, 1, 1).expand(int(num_features), I, 1, 1)
    return w


class _MlpDecoder(nn.Module):
    def __init__(self, *, latent_dim: int, out_dim: int, hidden_dims: tuple[int, ...] = (256, 256)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = latent_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LeakyReLU(0.2))
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


def _factorized_component_ll(
    *,
    data: Tensor,  # (B,F) possibly with NaNs
    leaf: FactorizedLeaf,
    decoder_out: Tensor,
    num_cats: int | None,
    normal_eps: float,
) -> Tensor:
    """Return component log-likelihoods per integration point.

    Returns:
        Tensor of shape (I,B) with per-component joint log-likelihoods.
    """
    if data.dim() != 2:
        raise InvalidParameterError("data must be 2D (N,F).")
    B, F = int(data.shape[0]), int(data.shape[1])

    mask = torch.isfinite(data)  # (B,F)
    if leaf == "bernoulli":
        logits = decoder_out.view(-1, F)  # (I,F)
        x = torch.where(mask, data, torch.zeros_like(data))
        if not torch.allclose(x[mask], x[mask].round()):
            raise InvalidParameterError("Bernoulli data must be in {0,1} (or NaN).")
        if (x[mask] < 0).any() or (x[mask] > 1).any():
            raise InvalidParameterError("Bernoulli data must be in {0,1} (or NaN).")
        logits = logits.unsqueeze(1)  # (I,1,F)
        x = x.unsqueeze(0)  # (1,B,F)
        logp1 = -torch.nn.functional.softplus(-logits)
        logp0 = -torch.nn.functional.softplus(logits)
        per_feat = x * logp1 + (1.0 - x) * logp0
        per_feat = torch.where(mask.unsqueeze(0), per_feat, torch.zeros_like(per_feat))
        return per_feat.sum(dim=2)  # (I,B)

    if leaf == "categorical":
        if num_cats is None or num_cats < 2:
            raise InvalidParameterError("num_cats must be provided and >= 2 for categorical.")
        K = int(num_cats)
        logits = decoder_out.view(-1, F, K)  # (I,F,K)
        x = torch.where(mask, data, torch.zeros_like(data))
        if not torch.allclose(x[mask], x[mask].round()):
            raise InvalidParameterError("Categorical data must be integer-coded (or NaN).")
        if (x[mask] < 0).any() or (x[mask] >= K).any():
            raise InvalidParameterError(f"Categorical data must be in {{0,..,{K - 1}}} (or NaN).")
        x_long = x.to(dtype=torch.long)
        log_probs = torch.log_softmax(logits, dim=2)  # (I,F,K)
        # gather along K with broadcast over batch without materializing (I,B,F,K) storage.
        gathered = torch.gather(
            log_probs.unsqueeze(1).expand(-1, B, -1, -1),
            dim=3,
            index=x_long.unsqueeze(0).unsqueeze(-1).expand(log_probs.shape[0], -1, -1, 1),
        ).squeeze(-1)  # (I,B,F)
        gathered = torch.where(mask.unsqueeze(0), gathered, torch.zeros_like(gathered))
        return gathered.sum(dim=2)  # (I,B)

    if leaf == "normal":
        # decoder_out encodes loc and scale (raw) per feature.
        loc_raw, scale_raw = decoder_out.chunk(2, dim=1)
        loc = loc_raw.view(-1, F)  # (I,F)
        scale = torch.nn.functional.softplus(scale_raw.view(-1, F)) + float(normal_eps)
        x = torch.where(mask, data, torch.zeros_like(data))
        loc = loc.unsqueeze(1)  # (I,1,F)
        scale = scale.unsqueeze(1)  # (I,1,F)
        x = x.unsqueeze(0)  # (1,B,F)
        log_two_pi = torch.log(x.new_tensor(2.0 * torch.pi))
        z = (x - loc) / scale
        per_feat = -0.5 * (z * z + 2.0 * torch.log(scale) + log_two_pi)
        per_feat = torch.where(mask.unsqueeze(0), per_feat, torch.zeros_like(per_feat))
        return per_feat.sum(dim=2)  # (I,B)

    raise InvalidParameterError(f"Unsupported leaf type: {leaf}.")


def _mixture_log_likelihood_from_component_ll(component_ll: Tensor, weights: Tensor) -> Tensor:
    """Compute log p(x) from per-component joint log-likelihoods.

    Args:
        component_ll: (I,B) log p(x | component_i)
        weights: (I,) non-negative, sum to 1

    Returns:
        (B,) mixture log-likelihood.
    """
    if component_ll.dim() != 2:
        raise InvalidParameterError("component_ll must be 2D (I,B).")
    if weights.dim() != 1 or weights.shape[0] != component_ll.shape[0]:
        raise InvalidParameterError("weights must be 1D with length I.")
    log_w = torch.log(weights.clamp_min(1e-30)).view(-1, 1)
    return torch.logsumexp(log_w + component_ll, dim=0)


def learn_continuous_mixture_factorized(
    data: Tensor,
    *,
    leaf: FactorizedLeaf,
    latent_dim: int = 4,
    num_points_train: int = 128,
    num_points_eval: int | None = None,
    num_epochs: int = 300,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 0,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    num_cats: int | None = None,
    normal_eps: float = 1e-4,
    val_data: Tensor | None = None,
    patience: int = 15,
    lo: LatentOptimizationConfig | None = None,
) -> Sum:
    """Learn a continuous mixture with fully factorized structure S_F.

    Args:
        data: Training data of shape (N,F). NaNs are supported and treated as missing
            values (marginalized out).
        leaf: Leaf distribution family.
        latent_dim: Latent dimension d.
        num_points_train: Number of RQMC integration points during training.
        num_points_eval: Number of integration points for evaluation/early stopping.
            Defaults to num_points_train if None.
        num_epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate for Adam.
        seed: Random seed.
        device: Optional device for training.
        dtype: Optional dtype for training computations.
        num_cats: Number of categories K for categorical leaves.
        normal_eps: Minimum scale for Normal leaves.
        val_data: Optional validation data for early stopping.
        patience: Early stopping patience in epochs.
        lo: Latent optimization configuration. If None, LO is disabled.

    Returns:
        A compiled SPFlow module (discrete mixture / Sum) representing the trained model.
    """
    if data.dim() != 2:
        raise InvalidParameterError("data must be 2D (N,F).")
    if latent_dim < 1:
        raise InvalidParameterError("latent_dim must be >= 1.")
    if num_points_train < 1:
        raise InvalidParameterError("num_points_train must be >= 1.")
    if num_epochs < 1:
        raise InvalidParameterError("num_epochs must be >= 1.")
    if batch_size < 1:
        raise InvalidParameterError("batch_size must be >= 1.")
    if lr <= 0:
        raise InvalidParameterError("lr must be > 0.")
    if patience < 0:
        raise InvalidParameterError("patience must be >= 0.")

    num_points_eval = int(num_points_train if num_points_eval is None else num_points_eval)
    if num_points_eval < 1:
        raise InvalidParameterError("num_points_eval must be >= 1.")

    data = _to_device_dtype(data, device=device, dtype=dtype)
    if val_data is not None:
        val_data = _to_device_dtype(val_data, device=device, dtype=dtype)

    N, F = int(data.shape[0]), int(data.shape[1])
    device_eff = data.device
    dtype_eff = data.dtype

    if leaf == "bernoulli":
        out_dim = F
    elif leaf == "categorical":
        if num_cats is None:
            raise InvalidParameterError("num_cats must be provided for categorical leaves.")
        out_dim = F * int(num_cats)
    elif leaf == "normal":
        out_dim = 2 * F
    else:
        raise InvalidParameterError(f"Unsupported leaf type: {leaf}.")

    decoder = _MlpDecoder(latent_dim=latent_dim, out_dim=out_dim).to(device=device_eff, dtype=dtype_eff)
    opt = torch.optim.Adam(decoder.parameters(), lr=lr)

    gen = torch.Generator(device=device_eff)
    gen.manual_seed(int(seed))

    best_val = None
    best_state = None
    bad_epochs = 0

    def eval_ll_mean() -> float:
        points = rqmc_sobol_normal(
            num_points=num_points_eval,
            latent_dim=latent_dim,
            device=device_eff,
            dtype=dtype_eff,
            seed=42,
        )
        with torch.no_grad():
            z = points.z
            w = points.weights
            out = decoder(z)
            ll = _factorized_component_ll(
                data=val_data if val_data is not None else data,
                leaf=leaf,
                decoder_out=out,
                num_cats=num_cats,
                normal_eps=normal_eps,
            )
            mix_ll = _mixture_log_likelihood_from_component_ll(ll, w)
            return float(mix_ll.mean().item())

    for epoch in range(num_epochs):
        decoder.train()
        for batch in _iter_minibatches(data, batch_size=batch_size, generator=gen):
            # Fresh RQMC points per step (random shift via varying seed).
            step_seed = int(seed + epoch * 100000 + torch.randint(0, 10**9, (1,), generator=gen).item())
            points = rqmc_sobol_normal(
                num_points=num_points_train,
                latent_dim=latent_dim,
                device=device_eff,
                dtype=dtype_eff,
                seed=step_seed,
            )
            z = points.z
            w = points.weights
            decoder_out = decoder(z)
            component_ll = _factorized_component_ll(
                data=batch,
                leaf=leaf,
                decoder_out=decoder_out,
                num_cats=num_cats,
                normal_eps=normal_eps,
            )
            mix_ll = _mixture_log_likelihood_from_component_ll(component_ll, w)
            loss = -mix_ll.mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        if val_data is not None or patience > 0:
            decoder.eval()
            ll_mean = eval_ll_mean()
            if best_val is None or ll_mean > best_val:
                best_val = ll_mean
                best_state = {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if patience > 0 and bad_epochs >= patience:
                    break

    if best_state is not None:
        decoder.load_state_dict({k: v.to(device_eff) for k, v in best_state.items()})

    if lo is not None and lo.enabled:
        z_opt, w_opt = _latent_opt_factorized(
            data=data,
            val_data=val_data,
            leaf=leaf,
            decoder=decoder,
            latent_dim=latent_dim,
            num_cats=num_cats,
            normal_eps=normal_eps,
            cfg=lo,
        )
        return _compile_factorized(
            decoder=decoder,
            leaf=leaf,
            z=z_opt,
            weights=w_opt,
            num_features=F,
            num_cats=num_cats,
            normal_eps=normal_eps,
            device=device_eff,
            dtype=dtype_eff,
        )

    # Compile with a deterministic evaluation set of points.
    points = rqmc_sobol_normal(
        num_points=num_points_eval,
        latent_dim=latent_dim,
        device=device_eff,
        dtype=dtype_eff,
        seed=42,
    )
    return _compile_factorized(
        decoder=decoder,
        leaf=leaf,
        z=points.z,
        weights=points.weights,
        num_features=F,
        num_cats=num_cats,
        normal_eps=normal_eps,
        device=device_eff,
        dtype=dtype_eff,
    )


def _compile_factorized(
    *,
    decoder: nn.Module,
    leaf: FactorizedLeaf,
    z: Tensor,  # (I,d)
    weights: Tensor,  # (I,)
    num_features: int,
    num_cats: int | None,
    normal_eps: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Sum:
    decoder.eval()
    with torch.no_grad():
        out = decoder(z.to(device=device, dtype=dtype))

    components = []
    for i in range(int(z.shape[0])):
        if leaf == "bernoulli":
            logits = out[i].view(num_features)
            leaves = [
                Bernoulli(scope=Scope([j]), out_channels=1, logits=logits[j : j + 1].view(1, 1, 1))
                for j in range(num_features)
            ]
        elif leaf == "categorical":
            if num_cats is None:
                raise InvalidParameterError("num_cats must be provided for categorical compilation.")
            logits = out[i].view(num_features, int(num_cats))
            leaves = [
                Categorical(
                    scope=Scope([j]),
                    out_channels=1,
                    K=int(num_cats),
                    logits=logits[j : j + 1].view(1, 1, 1, int(num_cats)),
                )
                for j in range(num_features)
            ]
        elif leaf == "normal":
            loc_raw, scale_raw = out[i].chunk(2, dim=0)
            loc = loc_raw.view(num_features)
            scale = torch.nn.functional.softplus(scale_raw.view(num_features)) + float(normal_eps)
            leaves = [
                Normal(
                    scope=Scope([j]),
                    out_channels=1,
                    loc=loc[j : j + 1].view(1, 1, 1),
                    scale=scale[j : j + 1].view(1, 1, 1),
                )
                for j in range(num_features)
            ]
        else:
            raise InvalidParameterError(f"Unsupported leaf type: {leaf}.")

        comp = Product(inputs=leaves)
        components.append(comp)

    w = _make_sum_weights(
        num_components=len(components),
        num_features=components[0].out_shape.features,
        device=device,
        dtype=dtype,
    )
    w = _broadcast_component_weights(weights=weights.to(device=device, dtype=dtype), num_features=w.shape[0])
    return Sum(inputs=components, weights=w)


def _latent_opt_factorized(
    *,
    data: Tensor,
    val_data: Tensor | None,
    leaf: FactorizedLeaf,
    decoder: nn.Module,
    latent_dim: int,
    num_cats: int | None,
    normal_eps: float,
    cfg: LatentOptimizationConfig,
) -> tuple[Tensor, Tensor]:
    decoder.eval()
    device = data.device
    dtype = data.dtype

    points = rqmc_sobol_normal(
        num_points=cfg.num_points,
        latent_dim=latent_dim,
        device=device,
        dtype=dtype,
        seed=cfg.seed,
    )
    z = torch.nn.Parameter(points.z.clone())
    w = points.weights

    opt = torch.optim.Adam([z], lr=cfg.lr)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(cfg.seed))

    best_val = None
    best_z = None
    bad = 0

    eval_data = val_data if val_data is not None else data

    for epoch in range(cfg.num_epochs):
        for batch in _iter_minibatches(data, batch_size=cfg.batch_size, generator=gen):
            out = decoder(z)
            component_ll = _factorized_component_ll(
                data=batch,
                leaf=leaf,
                decoder_out=out,
                num_cats=num_cats,
                normal_eps=normal_eps,
            )
            mix_ll = _mixture_log_likelihood_from_component_ll(component_ll, w)
            loss = -mix_ll.mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        with torch.no_grad():
            out_eval = decoder(z)
            ll_eval = _factorized_component_ll(
                data=eval_data,
                leaf=leaf,
                decoder_out=out_eval,
                num_cats=num_cats,
                normal_eps=normal_eps,
            )
            mix_ll_eval = _mixture_log_likelihood_from_component_ll(ll_eval, w)
            score = float(mix_ll_eval.mean().item())

        if best_val is None or score > best_val:
            best_val = score
            best_z = z.detach().clone()
            bad = 0
        else:
            bad += 1
            if cfg.patience > 0 and bad >= cfg.patience:
                break

    return (best_z if best_z is not None else z.detach()), w.detach()


def learn_continuous_mixture_cltree(
    data: Tensor,
    *,
    leaf: CltLeaf,
    latent_dim: int = 4,
    num_points_train: int = 128,
    num_points_eval: int | None = None,
    num_epochs: int = 300,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 0,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    num_cats: int | None = None,
    val_data: Tensor | None = None,
    patience: int = 15,
    lo: LatentOptimizationConfig | None = None,
    alpha: float = 0.01,
) -> JointLogLikelihood:
    """Learn a continuous mixture with Chow–Liu tree structure S_CLT (discrete only).

    Notes:
        - This learner supports only discrete leaves (Bernoulli / Categorical).
        - Data must be complete (no NaNs) and integer-coded.

    Args:
        data: Training data of shape (N,F) with values in {0,..,K-1}.
        leaf: Discrete leaf family.
        latent_dim: Latent dimension d.
        num_points_train: Number of RQMC integration points during training.
        num_points_eval: Number of integration points for evaluation/early stopping.
            Defaults to num_points_train if None.
        num_epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate for Adam.
        seed: Random seed.
        device: Optional device for training.
        dtype: Optional dtype for training computations.
        num_cats: K for categorical leaves. Ignored for Bernoulli (K=2).
        val_data: Optional validation data for early stopping.
        patience: Early stopping patience in epochs.
        lo: Latent optimization configuration. If None, LO is disabled.
        alpha: CLTree pseudocount used at compile time.

    Returns:
        A compiled SPFlow module representing the trained model, wrapped so that
        log_likelihood returns a single feature (joint score).
    """
    if leaf not in ("bernoulli", "categorical"):
        raise UnsupportedOperationError("CLTree continuous mixtures support only discrete leaves.")
    if data.dim() != 2:
        raise InvalidParameterError("data must be 2D (N,F).")
    if torch.isnan(data).any():
        raise InvalidParameterError("CLTree continuous mixtures require complete data (no NaNs).")
    if not torch.allclose(data, data.round()):
        raise InvalidParameterError("CLTree data must be integer-coded.")

    num_points_eval = int(num_points_train if num_points_eval is None else num_points_eval)
    data = _to_device_dtype(data, device=device, dtype=dtype)
    if val_data is not None:
        val_data = _to_device_dtype(val_data, device=device, dtype=dtype)

    N, F = int(data.shape[0]), int(data.shape[1])
    device_eff = data.device
    dtype_eff = data.dtype

    K = 2 if leaf == "bernoulli" else int(num_cats or 0)
    if K < 2:
        raise InvalidParameterError("num_cats must be provided and >= 2 for categorical.")
    if (data < 0).any() or (data >= K).any():
        raise InvalidParameterError(f"CLTree data must be in {{0,..,{K - 1}}}.")

    # Learn the Chow–Liu structure once.
    tmp = CLTree(scope=Scope(list(range(F))), out_channels=1, num_repetitions=1, K=K)
    tmp = tmp.to(device=device_eff, dtype=dtype_eff)
    tmp.fit_structure(data)
    parents = tmp.parents.detach().clone()

    root = int((parents == -1).nonzero(as_tuple=False).view(-1)[0].item())

    # Decoder outputs: root logits (K) + conditional logits for every feature (F*K*K).
    out_dim = K + F * K * K
    decoder = _MlpDecoder(latent_dim=latent_dim, out_dim=out_dim).to(device=device_eff, dtype=dtype_eff)
    opt = torch.optim.Adam(decoder.parameters(), lr=lr)

    gen = torch.Generator(device=device_eff)
    gen.manual_seed(int(seed))

    best_val = None
    best_state = None
    bad_epochs = 0

    def decode_log_cpt(z_points: Tensor) -> Tensor:
        raw = decoder(z_points)  # (I, out_dim)
        root_logits = raw[:, :K]  # (I,K)
        cond_logits = raw[:, K:].view(-1, F, K, K)  # (I,F,K,K)
        log_cpt_all = torch.log_softmax(cond_logits, dim=2)  # normalize over x_i
        log_root = torch.log_softmax(root_logits, dim=1)  # (I,K)
        root_row = log_root.unsqueeze(-1).expand(-1, K, K)  # (I,K,K)
        root_mask = torch.zeros((F,), dtype=torch.bool, device=log_cpt_all.device)
        root_mask[root] = True
        log_cpt = torch.where(root_mask.view(1, F, 1, 1), root_row.unsqueeze(1), log_cpt_all)
        return log_cpt  # (I,F,K,K)

    def cltree_component_ll(*, x_batch: Tensor, log_cpt: Tensor) -> Tensor:
        # x_batch: (B,F) long-ish float
        x_long = x_batch.to(dtype=torch.long)
        I = int(log_cpt.shape[0])
        B = int(x_long.shape[0])
        ll = torch.zeros((I, B), dtype=log_cpt.dtype, device=log_cpt.device)

        # Root term.
        root_table = log_cpt[:, root, :, 0]  # (I,K)
        idx_root = x_long[:, root].view(1, B).expand(I, B)
        ll += torch.gather(root_table, dim=1, index=idx_root)

        # Conditionals.
        for i in range(F):
            p = int(parents[i].item())
            if p == -1:
                continue
            xi = x_long[:, i]
            xp = x_long[:, p]
            lin = (xi * K + xp).view(1, B).expand(I, B)
            table_flat = log_cpt[:, i].reshape(I, K * K)
            ll += torch.gather(table_flat, dim=1, index=lin)
        return ll

    def eval_ll_mean() -> float:
        points = rqmc_sobol_normal(
            num_points=num_points_eval,
            latent_dim=latent_dim,
            device=device_eff,
            dtype=dtype_eff,
            seed=42,
        )
        with torch.no_grad():
            log_cpt = decode_log_cpt(points.z)
            ll = cltree_component_ll(x_batch=val_data if val_data is not None else data, log_cpt=log_cpt)
            mix_ll = _mixture_log_likelihood_from_component_ll(ll, points.weights)
            return float(mix_ll.mean().item())

    for epoch in range(num_epochs):
        decoder.train()
        for batch in _iter_minibatches(data, batch_size=batch_size, generator=gen):
            step_seed = int(seed + epoch * 100000 + torch.randint(0, 10**9, (1,), generator=gen).item())
            points = rqmc_sobol_normal(
                num_points=num_points_train,
                latent_dim=latent_dim,
                device=device_eff,
                dtype=dtype_eff,
                seed=step_seed,
            )
            log_cpt = decode_log_cpt(points.z)
            component_ll = cltree_component_ll(x_batch=batch, log_cpt=log_cpt)
            mix_ll = _mixture_log_likelihood_from_component_ll(component_ll, points.weights)
            loss = -mix_ll.mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        if val_data is not None or patience > 0:
            decoder.eval()
            ll_mean = eval_ll_mean()
            if best_val is None or ll_mean > best_val:
                best_val = ll_mean
                best_state = {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if patience > 0 and bad_epochs >= patience:
                    break

    if best_state is not None:
        decoder.load_state_dict({k: v.to(device_eff) for k, v in best_state.items()})

    if lo is not None and lo.enabled:
        z_opt, w_opt = _latent_opt_cltree(
            data=data,
            val_data=val_data,
            decoder=decoder,
            decode_log_cpt=decode_log_cpt,
            parents=parents,
            root=root,
            K=K,
            latent_dim=latent_dim,
            cfg=lo,
        )
        return _compile_cltree(
            decoder=decoder,
            decode_log_cpt=decode_log_cpt,
            parents=parents,
            root=root,
            K=K,
            z=z_opt,
            weights=w_opt,
            alpha=alpha,
        )

    points = rqmc_sobol_normal(
        num_points=num_points_eval,
        latent_dim=latent_dim,
        device=device_eff,
        dtype=dtype_eff,
        seed=42,
    )
    return _compile_cltree(
        decoder=decoder,
        decode_log_cpt=decode_log_cpt,
        parents=parents,
        root=root,
        K=K,
        z=points.z,
        weights=points.weights,
        alpha=alpha,
    )


def _latent_opt_cltree(
    *,
    data: Tensor,
    val_data: Tensor | None,
    decoder: nn.Module,
    decode_log_cpt,
    parents: Tensor,
    root: int,
    K: int,
    latent_dim: int,
    cfg: LatentOptimizationConfig,
) -> tuple[Tensor, Tensor]:
    decoder.eval()
    device = data.device
    dtype = data.dtype
    F = int(data.shape[1])

    points = rqmc_sobol_normal(
        num_points=cfg.num_points,
        latent_dim=latent_dim,
        device=device,
        dtype=dtype,
        seed=cfg.seed,
    )
    z = torch.nn.Parameter(points.z.clone())
    w = points.weights

    opt = torch.optim.Adam([z], lr=cfg.lr)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(cfg.seed))

    best_val = None
    best_z = None
    bad = 0
    eval_data = val_data if val_data is not None else data

    def cltree_component_ll(*, x_batch: Tensor, log_cpt: Tensor) -> Tensor:
        x_long = x_batch.to(dtype=torch.long)
        I = int(log_cpt.shape[0])
        B = int(x_long.shape[0])
        ll = torch.zeros((I, B), dtype=log_cpt.dtype, device=log_cpt.device)
        root_table = log_cpt[:, root, :, 0]  # (I,K)
        idx_root = x_long[:, root].view(1, B).expand(I, B)
        ll += torch.gather(root_table, dim=1, index=idx_root)
        for i in range(F):
            p = int(parents[i].item())
            if p == -1:
                continue
            xi = x_long[:, i]
            xp = x_long[:, p]
            lin = (xi * K + xp).view(1, B).expand(I, B)
            table_flat = log_cpt[:, i].reshape(I, K * K)
            ll += torch.gather(table_flat, dim=1, index=lin)
        return ll

    for epoch in range(cfg.num_epochs):
        for batch in _iter_minibatches(data, batch_size=cfg.batch_size, generator=gen):
            log_cpt = decode_log_cpt(z)
            ll = cltree_component_ll(x_batch=batch, log_cpt=log_cpt)
            mix_ll = _mixture_log_likelihood_from_component_ll(ll, w)
            loss = -mix_ll.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        with torch.no_grad():
            log_cpt_eval = decode_log_cpt(z)
            ll_eval = cltree_component_ll(x_batch=eval_data, log_cpt=log_cpt_eval)
            mix_ll_eval = _mixture_log_likelihood_from_component_ll(ll_eval, w)
            score = float(mix_ll_eval.mean().item())

        if best_val is None or score > best_val:
            best_val = score
            best_z = z.detach().clone()
            bad = 0
        else:
            bad += 1
            if cfg.patience > 0 and bad >= cfg.patience:
                break

    return (best_z if best_z is not None else z.detach()), w.detach()


def _compile_cltree(
    *,
    decoder: nn.Module,
    decode_log_cpt,
    parents: Tensor,
    root: int,
    K: int,
    z: Tensor,
    weights: Tensor,
    alpha: float,
) -> JointLogLikelihood:
    decoder.eval()
    with torch.no_grad():
        log_cpt = decode_log_cpt(z)  # (I,F,K,K)

    I = int(z.shape[0])
    F = int(log_cpt.shape[1])
    scope = Scope(list(range(F)))

    components = []
    for i in range(I):
        log_cpt_i = log_cpt[i].unsqueeze(1).unsqueeze(1)  # (F,1,1,K,K)
        node = CLTree(
            scope=scope,
            out_channels=1,
            num_repetitions=1,
            K=K,
            alpha=float(alpha),
            parents=parents,
            log_cpt=log_cpt_i,
        )
        components.append(node)

    w_sum = _make_sum_weights(
        num_components=I,
        num_features=F,
        device=z.device,
        dtype=z.dtype,
    )
    w_sum = _broadcast_component_weights(weights=weights.to(device=z.device, dtype=z.dtype), num_features=F)
    mix = Sum(inputs=components, weights=w_sum)
    return JointLogLikelihood(mix)
