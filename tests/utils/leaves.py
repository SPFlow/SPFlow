import torch

from spflow.meta import Scope
from spflow.modules import leaves
from spflow.modules.leaves import Normal
from spflow.modules.leaves.base import LeafModule

from torch import Tensor, nn

from spflow.utils.leaves import init_parameter, _handle_mle_edge_cases


def evaluate_log_likelihood(module: LeafModule, data: torch.Tensor):
    lls = module.log_likelihood(data)
    # Always expect 4D output with num_repetitions dimension
    assert lls.shape == (
        data.shape[0],
        len(module.scope.query),
        module.out_channels,
        module.num_repetitions,
    )
    assert torch.isfinite(lls).all()


def evaluate_samples(node: LeafModule, data: torch.Tensor, is_mpe: bool, sampling_ctx):
    samples = node.sample(data=data, is_mpe=is_mpe, sampling_ctx=sampling_ctx)
    assert samples.shape == data.shape
    s_query = samples[:, node.scope.query]
    assert s_query.shape == (data.shape[0], len(node.scope.query))
    assert torch.isfinite(s_query).all()


def make_normal_leaf(
    scope=None, out_features=None, out_channels=None, num_repetitions=None, mean=None, std=None
) -> "DummyLeaf":
    """Create a Normal leaves module.

    Args:
        mean: Mean of the distribution.
        std: Standard deviation of the distribution.
    """
    num_repetitions = 1 if num_repetitions is None else num_repetitions
    if mean is not None:
        out_features = mean.shape[0]
    # assert (scope is None) ^ (out_features is None), "Either scope or out_features must be given"

    if scope is None:
        scope = Scope(list(range(0, out_features)))
    elif isinstance(scope, int):
        scope = Scope([scope])
    elif isinstance(scope, list):
        scope = Scope(scope)
    elif isinstance(scope, Scope):
        pass
    else:
        out_features = len(scope.query)

    if out_features and scope:
        assert len(scope.query) == out_features, "scope and out_features must have the same length"

    # Always create 3D parameters with num_repetitions dimension
    mean = mean if mean is not None else torch.randn(len(scope.query), out_channels, num_repetitions)
    std = std if std is not None else torch.rand(len(scope.query), out_channels, num_repetitions) + 1e-8
    return DummyLeaf(scope=scope, loc=mean, scale=std, num_repetitions=num_repetitions)


def make_normal_data(mean=0.0, std=1.0, num_samples=10, out_features=2):
    torch.manual_seed(0)
    return torch.randn(num_samples, out_features) * std + mean


def make_leaf(
    cls, out_channels: int = None, out_features: int = None, scope: Scope = None, num_repetitions=None
) -> LeafModule:
    assert (out_features is None) ^ (scope is None), "Either out_features or scope must be provided"

    num_repetitions = 1 if num_repetitions is None else num_repetitions
    if scope is None:
        scope = Scope(list(range(0, out_features)))

    # Check special cases
    if cls == leaves.Binomial:
        return leaves.Binomial(
            scope=scope,
            out_channels=out_channels,
            total_count=torch.ones(1) * 3,
            num_repetitions=num_repetitions,
        )
    elif cls == leaves.NegativeBinomial:
        return leaves.NegativeBinomial(
            scope=scope,
            out_channels=out_channels,
            total_count=torch.ones(1) * 3,
            num_repetitions=num_repetitions,
        )
    elif cls == leaves.Categorical:
        return leaves.Categorical(
            scope=scope,
            out_channels=out_channels,
            K=3,
            num_repetitions=num_repetitions,
        )
    elif cls == leaves.Hypergeometric:
        return leaves.Hypergeometric(
            scope=scope,
            n=torch.ones((len(scope.query), out_channels, num_repetitions)) * 3,
            N=torch.ones((len(scope.query), out_channels, num_repetitions)) * 10,
            K=torch.ones((len(scope.query), out_channels, num_repetitions)) * 5,
            num_repetitions=num_repetitions,
        )
    elif cls == leaves.Uniform:
        return leaves.Uniform(
            scope=scope,
            low=torch.zeros((len(scope.query), out_channels, num_repetitions)),
            high=torch.ones((len(scope.query), out_channels, num_repetitions)),
            num_repetitions=num_repetitions,
        )
    else:
        # Default case: just call the class
        return cls(scope=scope, out_channels=out_channels, num_repetitions=num_repetitions)


def make_leaf_args(cls, out_channels: int = None, scope: Scope = None, num_repetitions=None) -> dict:
    num_repetitions = 1 if num_repetitions is None else num_repetitions
    # Check special cases
    if cls == leaves.Binomial or cls == leaves.NegativeBinomial:
        return {"total_count": torch.ones(1) * 3}
    elif cls == leaves.Categorical:
        return {"K": 3}
    elif cls == leaves.Hypergeometric:
        # Always create 3D tensors with num_repetitions dimension
        return {
            "n": torch.ones((len(scope.query), out_channels, num_repetitions)) * 3,
            "N": torch.ones((len(scope.query), out_channels, num_repetitions)) * 10,
            "K": torch.ones((len(scope.query), out_channels, num_repetitions)) * 5,
        }
    elif cls == leaves.Uniform:
        # Always create 3D tensors with num_repetitions dimension
        return {
            "low": torch.zeros((len(scope.query), out_channels, num_repetitions)),
            "high": torch.ones((len(scope.query), out_channels, num_repetitions)),
        }
    else:
        return {}


def make_cond_leaf(
    cls, out_channels: int = None, out_features: int = None, scope: Scope = None
) -> LeafModule:
    assert (out_features is None) ^ (scope is None), "Either out_features or scope must be provided"

    if scope is None:
        scope = Scope(list(range(0, out_features)))

    event_shape = (len(scope.query), out_channels)
    """
    # Check special cases
    if cls == leaves.CondBinomial:
        return leaves.CondBinomial(scope=scope, out_channels=out_channels, total_count=torch.ones(1) * 3)
    elif cls == leaves.NegativeBinomial:
        return leaves.NegativeBinomial(scope=scope, out_channels=out_channels, total_count=torch.ones(1) * 3)
    elif cls == leaves.CondCategorical:
        return leaves.CondCategorical(
            scope=scope,
            out_channels=out_channels,
            K=3,
        )
    elif cls == leaves.CondHypergeometric:
        return leaves.CondHypergeometric(
            scope=scope,
            n=torch.ones((len(scope.query), out_channels)) * 3,
            N=torch.ones((len(scope.query), out_channels)) * 10,
            K=torch.ones((len(scope.query), out_channels)) * 5,
        )
    elif cls == leaves.CondUniform:
        return leaves.CondUniform(
            scope=scope,
            low=torch.zeros((len(scope.query), out_channels)),
            high=torch.ones((len(scope.query), out_channels)),
        )
    else:
    """
    # Default case: just call the class
    mean = torch.randn(event_shape)
    std = torch.rand(event_shape)
    cond_f = lambda data: {"loc": mean, "scale": std}
    return cls(scope=scope, cond_f=cond_f)


def make_data(cls, out_features: int, n_samples: int = 5) -> torch.Tensor:
    scope = Scope(list(range(0, out_features)))
    return make_leaf(cls=cls, scope=scope, out_channels=1).distribution.sample((n_samples,)).squeeze(-1).squeeze(-1)


def make_cond_data(cls, out_features: int, n_samples: int = 5) -> torch.Tensor:
    scope = Scope(list(range(0, out_features)))
    return make_cond_leaf(cls=cls, scope=scope, out_channels=1).distribution.sample((n_samples,)).squeeze(-1)


class Constraint:
    NONE = "none"
    POSITIVE = "positive"
    UNIT_INTERVAL = "unit_interval"
    NEGATIVE = "negative"


class SimpleParameterNetwork(torch.nn.Module):
    """Simple parameter network for conditional distributions with optional fixed parameters."""

    def __init__(
            self,
            input_size: int,
            output_size: int,
            num_features: int,
            param_constraints: dict[str, str],
            fixed_params: dict[str, torch.Tensor] | None = None,
            num_repetitions: int | None = None,
    ):
        """
        Args:
            input_size: Number of evidence variables.
            output_size: Number of output channels.
            num_features: Number of features (query variables).
            param_constraints: Dictionary specifying constraints for trainable parameters.
            fixed_params: Optional dictionary of fixed (non-trainable) parameters.
            num_repetitions: Number of repetitions (optional, for multi-repetition distributions).
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_features = num_features
        self.param_constraints = param_constraints
        self.fixed_params = fixed_params if fixed_params is not None else {}
        self.num_repetitions = num_repetitions

        # Only create network for trainable parameters
        if len(param_constraints) > 0:
            # Always use 3D output with num_repetitions dimension
            output_features = num_features * output_size * num_repetitions * len(param_constraints)
            self.network = torch.nn.Sequential(
                torch.nn.Linear(input_size, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, output_features),
            )
        else:
            self.network = None

    def forward(self, evidence: torch.Tensor) -> dict:
        """Generate distribution parameters from evidence.

        Args:
            evidence: Shape (batch_size, input_size)

        Returns:
            Dictionary with both trainable and fixed parameters.
        """
        batch_size = evidence.shape[0]
        param_dict = {}

        # Generate trainable parameters
        if self.network is not None and len(self.param_constraints) > 0:
            params = self.network(evidence)
            # Always reshape to 5D with num_repetitions dimension
            params = params.reshape(
                batch_size, self.num_features, self.output_size, self.num_repetitions, len(self.param_constraints)
            )

            for i, (param_name, param_constraint) in enumerate(self.param_constraints.items()):
                p = params[..., i]
                if param_constraint == Constraint.NONE:
                    param_dict[param_name] = p
                elif param_constraint == Constraint.POSITIVE:
                    param_dict[param_name] = torch.nn.functional.softplus(p) + 1e-6
                elif param_constraint == Constraint.UNIT_INTERVAL:
                    param_dict[param_name] = torch.sigmoid(p)
                elif param_constraint == Constraint.NEGATIVE:
                    param_dict[param_name] = -torch.nn.functional.softplus(p) - 1e-6
                else:
                    raise ValueError(f"Unknown constraint: {param_constraint}")

        # Add fixed parameters (broadcast to batch size if needed)
        for param_name, param_value in self.fixed_params.items():
            # Expand fixed parameter to match batch size
            # Fixed params are typically shaped as (num_features, out_channels) or scalar
            # Always expand to 4D: (batch_size, num_features, out_channels, num_repetitions)
            if param_value.ndim == 0:
                # Scalar
                expanded = param_value.expand(batch_size, self.num_features, self.output_size, self.num_repetitions)
            elif param_value.ndim == 1:
                # Scalar or 1D - expand to 4D
                expanded = param_value.expand(batch_size, self.num_features, self.output_size, self.num_repetitions)
            elif param_value.ndim == 2:
                # (num_features, out_channels) - add batch and num_repetitions dimensions
                expanded = param_value.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, self.num_repetitions)
            elif param_value.ndim == 3:
                # (num_features, out_channels, num_repetitions) - add batch dimension
                expanded = param_value.unsqueeze(0).expand(batch_size, -1, -1, -1)
            else:
                expanded = param_value
            param_dict[param_name] = expanded

        return param_dict


def get_param_constraints(distribution_class) -> dict[str, str]:
    """Get parameter constraints for a distribution class.

    Args:
        distribution_class: The distribution class (e.g., leaves.Normal, leaves.Bernoulli).

    Returns:
        Dictionary mapping parameter names to constraints (NONE, POSITIVE, UNIT_INTERVAL, NEGATIVE).

    Note:
        Gamma uses NONE for both parameters because it applies torch.exp() in its
        conditional_distribution method, which is different from the base class behavior.
    """
    # Map each distribution to its trainable parameters and constraints
    if distribution_class == leaves.Bernoulli:
        return {"probs": Constraint.UNIT_INTERVAL}
    elif distribution_class == leaves.Binomial:
        return {"probs": Constraint.UNIT_INTERVAL}
    elif distribution_class == leaves.Categorical:
        return {"probs": Constraint.UNIT_INTERVAL}
    elif distribution_class == leaves.Exponential:
        return {"rate": Constraint.POSITIVE}
    elif distribution_class == leaves.Gamma:
        # Gamma applies torch.exp() itself in conditional_distribution
        return {"concentration": Constraint.NONE, "rate": Constraint.NONE}
    elif distribution_class == leaves.Geometric:
        return {"probs": Constraint.UNIT_INTERVAL}
    elif distribution_class == leaves.Hypergeometric:
        # Hypergeometric has no trainable parameters (n, N, K are all fixed)
        return {}
    elif distribution_class == leaves.LogNormal:
        return {"loc": Constraint.NONE, "scale": Constraint.POSITIVE}
    elif distribution_class == leaves.NegativeBinomial:
        return {"probs": Constraint.UNIT_INTERVAL}
    elif distribution_class == leaves.Normal:
        return {"loc": Constraint.NONE, "scale": Constraint.POSITIVE}
    elif distribution_class == leaves.Poisson:
        return {"rate": Constraint.POSITIVE}
    elif distribution_class == leaves.Uniform:
        return {"low": Constraint.NONE, "high": Constraint.NONE}
    else:
        raise ValueError(f"Unknown distribution class: {distribution_class}")


def create_conditional_parameter_fn(
        distribution_class, out_features: int, out_channels: int, evidence_size: int, num_repetitions: int | None = None
):
    """Create a parameter network for conditional distribution testing.

    Args:
        distribution_class: The distribution class to create a parameter network for.
        out_features: Number of features (query variables).
        out_channels: Number of output channels.
        evidence_size: Number of evidence variables.
        num_repetitions: Number of repetitions (optional, for multi-repetition distributions).

    Returns:
        A parameter network configured for the distribution.
    """
    param_constraints = get_param_constraints(distribution_class)

    # Handle distributions with fixed parameters that need to be passed to the torch distribution
    fixed_params = {}
    if distribution_class == leaves.Binomial or distribution_class == leaves.NegativeBinomial:
        fixed_params["total_count"] = torch.ones(1) * 3

    # Note: Categorical K is not included because torch.distributions.Categorical
    # infers K from the probs/logits shape. The K parameter is only for SPFlow's
    # Categorical class initialization, not for the torch distribution.

    return SimpleParameterNetwork(
        input_size=evidence_size,
        output_size=out_channels,
        num_features=out_features,
        param_constraints=param_constraints,
        fixed_params=fixed_params if fixed_params else None,
        num_repetitions=num_repetitions,
    )




class DummyLeaf(LeafModule):
    """Dummy leaf module for testing purposes."""
    def __init__(
        self,
        scope,
        out_channels: int = None,
        num_repetitions: int = 1,
        loc: Tensor = None,
        scale: Tensor = None,
        parameter_fn: nn.Module = None,
        validate_args: bool | None = True,
    ):
        super().__init__(
            scope=scope,
            out_channels=out_channels,
            num_repetitions=num_repetitions,
            params=[loc, scale],
            parameter_fn=parameter_fn,
            validate_args=validate_args,
        )

        loc = init_parameter(param=loc, event_shape=self._event_shape, init=torch.randn)
        scale = init_parameter(param=scale, event_shape=self._event_shape, init=torch.rand)

        self.loc = nn.Parameter(loc)
        self.log_scale = nn.Parameter(torch.log(scale))

    @property
    def scale(self) -> Tensor:
        """Standard deviation in natural space (read via exp of log_std)."""
        return torch.exp(self.log_scale)

    @scale.setter
    def scale(self, value: Tensor) -> None:
        """Set standard deviation (stores as log_std, no validation after init)."""
        self.log_scale.data = torch.log(
            torch.as_tensor(value, dtype=self.log_scale.dtype, device=self.log_scale.device)
        )

    @property
    def _supported_value(self):
        return 0.0

    @property
    def _torch_distribution_class(self) -> type[torch.distributions.Normal]:
        return torch.distributions.Normal

    def params(self):
        return {"loc": self.loc, "scale": self.scale}

    def _compute_parameter_estimates(
        self, data: Tensor, weights: Tensor, bias_correction: bool
    ) -> dict[str, Tensor]:
        """Compute raw MLE estimates for normal distribution (without broadcasting).

        Args:
            data: Input data tensor.
            weights: Weight tensor for each data point.
            bias_correction: Whether to apply bias correction to variance estimate.

        Returns:
            Dictionary with 'loc' and 'scale' estimates (shape: out_features).
        """
        return {"loc": torch.zeros_like(self.loc), "scale": torch.ones_like(self.scale)}

    def _set_mle_parameters(self, params_dict: dict[str, Tensor]) -> None:
        """Set MLE-estimated parameters for Normal distribution.

        Explicitly handles the two parameter types:
        - loc: Direct nn.Parameter, update .data attribute
        - scale: Property with setter, calls property setter which updates log_scale

        Args:
            params_dict: Dictionary with 'loc' and 'scale' parameter values.
        """
        self.loc.data = params_dict["loc"]
        self.scale = params_dict["scale"]  # Uses property setter


    def _mle_update_statistics(self, data: Tensor, weights: Tensor, bias_correction: bool) -> None:
        """Compute weighted mean and standard deviation.

        Args:
            data: Input data tensor.
            weights: Weight tensor for each data point.
            bias_correction: Whether to apply bias correction to variance estimate.
        """
        estimates = self._compute_parameter_estimates(data, weights, bias_correction)

        # Broadcast to event_shape and assign directly
        self.loc.data = self._broadcast_to_event_shape(estimates["loc"])
        self.scale = self._broadcast_to_event_shape(estimates["scale"])
