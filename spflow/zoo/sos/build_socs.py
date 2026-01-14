"""SOCS structure builder utilities.

SOCS (Σ2cmp) requires a set of compatible component circuits (same structure).
This helper provides a minimal, SPFlow-native way to construct such components
by cloning a template circuit and optionally converting `Sum` nodes into
`SignedSum` nodes with perturbed (possibly negative) weights.
"""

from __future__ import annotations

import copy

import torch

from spflow.exceptions import InvalidParameterError
from spflow.modules.module import Module
from spflow.zoo.sos.socs import SOCS
from spflow.zoo.sos.signed_sum import SignedSum
from spflow.modules.sums.sum import Sum
from spflow.zoo.sos.compatibility import check_compatible_components


def build_socs(
    template: Module,
    *,
    num_components: int,
    signed: bool = True,
    noise_scale: float = 0.05,
    flip_prob: float = 0.5,
    seed: int | None = None,
) -> SOCS:
    """Build a SOCS model from a compatible component template.

    Args:
        template: A SPFlow module representing a (typically scalar-output) circuit.
            This circuit is deep-copied `num_components` times to ensure all
            components share the same structure.
        num_components: Number of components r.
        signed: If True, convert all `Sum` nodes in each clone to `SignedSum`
            nodes with perturbed weights (allowing negative weights).
        noise_scale: Standard deviation of additive Gaussian noise applied to
            copied weights when `signed=True`.
        flip_prob: Probability of flipping the sign of each weight entry when
            `signed=True`. Must be in [0, 1].
        seed: Optional random seed used for weight perturbations.

    Returns:
        A `SOCS` module with `num_components` compatible components.
    """
    if num_components < 1:
        raise InvalidParameterError("num_components must be >= 1.")
    if noise_scale < 0.0:
        raise InvalidParameterError("noise_scale must be >= 0.")
    if not (0.0 <= flip_prob <= 1.0):
        raise InvalidParameterError("flip_prob must be in [0, 1].")

    gen = None
    if seed is not None:
        gen = torch.Generator(device=template.device)
        gen.manual_seed(int(seed))

    def _convert_sum(node: Sum) -> SignedSum:
        w = node.weights.detach().clone()
        if gen is None:
            flip = (torch.rand_like(w) < flip_prob).to(dtype=w.dtype)
        else:
            flip = (torch.rand(w.shape, dtype=w.dtype, device=w.device, generator=gen) < flip_prob).to(
                dtype=w.dtype
            )
        sign = 1.0 - 2.0 * flip  # {+1,-1}
        w = w * sign
        if noise_scale > 0.0:
            if gen is None:
                w = w + noise_scale * torch.randn_like(w)
            else:
                w = w + noise_scale * torch.randn(w.shape, dtype=w.dtype, device=w.device, generator=gen)
        return SignedSum(
            inputs=node.inputs,
            out_channels=node.out_shape.channels,
            num_repetitions=node.out_shape.repetitions,
            weights=w,
        )

    def _transform_in_place(root: torch.nn.Module) -> None:
        for name, child in list(root.named_children()):
            if isinstance(child, Sum) and signed:
                root._modules[name] = _convert_sum(child)
                child = root._modules[name]
            _transform_in_place(child)

    components: list[Module] = []
    for _i in range(num_components):
        comp = copy.deepcopy(template)
        if isinstance(comp, Sum) and signed:
            comp = _convert_sum(comp)
        _transform_in_place(comp)
        components.append(comp)

    check_compatible_components(components)
    return SOCS(components)


def build_abs_weight_proposal(component: Module, *, eps: float = 1e-8) -> Module:
    """Build a monotone proposal q(x) from a (possibly signed) component.

    Replaces each `SignedSum` with a standard `Sum` whose weights are proportional
    to `abs(weights)`, ensuring q is non-negative and normalized at each sum node.

    Args:
        component: Component circuit to convert.
        eps: Small additive constant to avoid all-zero abs weights.

    Returns:
        A new `Module` that supports `.sample()` and `.log_likelihood()` and
        can be used as an independence proposal.
    """
    if eps <= 0.0:
        raise InvalidParameterError("eps must be > 0.")

    prop = copy.deepcopy(component)

    def _convert_signed(node: SignedSum) -> Sum:
        w = node.weights.detach()
        w = torch.abs(w) + w.new_tensor(float(eps))
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return Sum(
            inputs=node.inputs,
            out_channels=node.out_shape.channels,
            num_repetitions=node.out_shape.repetitions,
            weights=w,
        )

    if isinstance(prop, SignedSum):
        prop = _convert_signed(prop)

    def _transform(root: torch.nn.Module) -> None:
        for name, child in list(root.named_children()):
            if isinstance(child, SignedSum):
                root._modules[name] = _convert_signed(child)
                child = root._modules[name]
            _transform(child)

    _transform(prop)
    return prop


def build_complex_socs(real: Module, imag: Module) -> SOCS:
    """Build a SOCS model equivalent to a complex squared circuit |c|^2.

    This implements the paper-aligned reduction:

        c(x) = a(x) + i b(x)  =>  |c(x)|^2 = a(x)^2 + b(x)^2

    by constructing a SOCS with two components `[a, b]`. This avoids introducing
    complex-valued parameters/semirings in SPFlow while still matching the
    squared-magnitude semantics used by complex SOS models.

    Args:
        real: Circuit computing a(x).
        imag: Circuit computing b(x).

    Returns:
        A `SOCS` module with two components.
    """
    if real.scope != imag.scope:
        raise InvalidParameterError(
            "build_complex_socs requires real and imag circuits to have identical scope."
        )
    if tuple(real.out_shape) != tuple(imag.out_shape):
        raise InvalidParameterError(
            "build_complex_socs requires real and imag circuits to have identical out_shape; "
            f"got {tuple(real.out_shape)} vs {tuple(imag.out_shape)}."
        )
    return SOCS([real, imag])
