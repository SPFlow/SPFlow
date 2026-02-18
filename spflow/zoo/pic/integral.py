from __future__ import annotations
from collections.abc import Callable, Iterable
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor, nn

from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.meta.data.scope import Scope


class Integral(Module):
    """Integral module representing a continuous latent variable integration.

    Computes:
        g_u(X, Z_u) = ∫ f_u(Z_u, Y_u) · g_in(X, Y_u) dY_u

    Where:
        - X are variables from the input branch (descendants).
        - Z_u are conditioned latent variables for this unit.
        - Y_u are latent variables integrated out by this unit.

    Attributes:
        inputs (Module): The child module whose output depends on Y_u.
        latent_scope (Scope): Conditioned latent variables Z_u for this unit.
        integrated_latent_scope (Scope): Integrated latent variables Y_u for this unit.
        function (Callable | nn.Module): The weighting function f_u(Z_u, Y_u).
    """

    def __init__(
        self,
        input_module: Module,
        latent_scope: Scope | int | Iterable[int] | None,
        integrated_latent_scope: Scope | int | Iterable[int] | None,
        function: Callable[[Tensor, Tensor], Tensor] | nn.Module | None,
        function_head_idx: Optional[int] = None,
    ) -> None:
        """Initialize the Integral module.

        Args:
            input_module: The child module.
            latent_scope: Scope of conditioned latent variables Z_u.
            integrated_latent_scope: Scope of integrated latent variables Y_u.
            function: Function f(Z, Y) parameterized by neural network or similar.
                Should accept broadcastable tensors `z` and `y` and return positive weights.
                Convention: `z.shape[-1] == |Z_u|` and `y.shape[-1] == |Y_u|`.
            function_head_idx: Optional head index when `function` is a multi-function group.
        """
        super().__init__()

        self.latent_scope = Scope.as_scope(latent_scope)
        self.integrated_latent_scope = Scope.as_scope(integrated_latent_scope)

        self.inputs = input_module
        self.scope = input_module.scope

        self.function = function
        self.function_head_idx = function_head_idx

        if not isinstance(input_module, Module):
            raise ValueError("Integral module expects a single input Module.")

        # PIC nodes are symbolic; the channel dimension depends on the chosen quadrature size K.
        self.in_shape = input_module.out_shape
        self.out_shape = ModuleShape(features=self.in_shape.features, channels=1, repetitions=1)

    @property
    def feature_to_scope(self) -> np.ndarray:
        # self.inputs is confirmed to be Module by __init__ check
        return self.inputs.feature_to_scope  # type: ignore

    def log_likelihood(self, data: Tensor, cache: Cache | None = None) -> Tensor:
        """Compute symbolic log-likelihood (Not Implemented for direct execution).

        Integral nodes are typically compiled to QPCs for inference.
        """
        raise NotImplementedError(
            "Exact inference on Integral nodes is not implemented. Please compile to QPC using `pic2qpc`."
        )

    def sample(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        is_mpe: bool = False,
        cache: Cache | None = None,
    ) -> Tensor:
        raise NotImplementedError("Sampling from Integral nodes is not implemented.")

    def _sample(
        self,
        data: Tensor,
        sampling_ctx: SamplingContext,
        cache: Cache,
        is_mpe: bool = False,
    ) -> Tensor:
        raise NotImplementedError("Sampling from Integral nodes is not implemented.")

    def marginalize(
        self, marg_rvs: list[int], prune: bool = True, cache: Cache | None = None
    ) -> Module | None:
        raise NotImplementedError("Marginalization on Integral nodes is not implemented.")
