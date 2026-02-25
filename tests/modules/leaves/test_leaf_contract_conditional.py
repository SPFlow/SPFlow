import pytest
import torch

from spflow.meta import Scope
from spflow.modules import leaves
from tests.modules.leaves.leaf_contract_data import CONDITIONAL_LEAF_GRAD_PARAMS, CONDITIONAL_LEAF_PARAMS
from tests.utils.leaves import create_conditional_parameter_fn, make_leaf, make_leaf_args

pytestmark = pytest.mark.contract


class TestConditionalLeaves:
    @pytest.mark.parametrize(
        "leaf_cls,out_features,out_channels,num_reps",
        CONDITIONAL_LEAF_PARAMS,
    )
    def test_conditional_leaf_is_conditional(self, leaf_cls, out_features, out_channels, num_reps):
        query = list(range(out_features))
        evidence = [out_features, out_features + 1]

        scope = Scope(query)
        leaf = make_leaf(cls=leaf_cls, scope=scope, out_channels=out_channels, num_repetitions=num_reps)
        assert not leaf.is_conditional

        param_net = create_conditional_parameter_fn(
            distribution_class=leaf_cls,
            out_features=out_features,
            out_channels=out_channels,
            evidence_size=len(evidence),
            num_repetitions=num_reps,
        )
        scope_cond = Scope(query, evidence=evidence)
        leaf_args = make_leaf_args(
            cls=leaf_cls, out_channels=out_channels, scope=scope_cond, num_repetitions=num_reps
        )
        leaf_cond = leaf_cls(scope=scope_cond, out_channels=out_channels, parameter_fn=param_net, **leaf_args)
        assert leaf_cond.is_conditional

    @pytest.mark.parametrize(
        "leaf_cls,out_features,out_channels,num_reps",
        CONDITIONAL_LEAF_PARAMS,
    )
    def test_conditional_leaf_distribution_with_evidence(
        self, leaf_cls, out_features, out_channels: int, num_reps
    ):
        query = list(range(out_features))
        evidence = [out_features, out_features + 1]

        param_net = create_conditional_parameter_fn(
            distribution_class=leaf_cls,
            out_features=out_features,
            out_channels=out_channels,
            evidence_size=len(evidence),
            num_repetitions=num_reps,
        )
        scope = Scope(query, evidence=evidence)
        leaf_args = make_leaf_args(
            cls=leaf_cls, out_channels=out_channels, scope=scope, num_repetitions=num_reps
        )
        leaf = leaf_cls(scope=scope, out_channels=out_channels, parameter_fn=param_net, **leaf_args)

        batch_size = 4
        evidence_tensor = torch.randn(batch_size, len(evidence))

        dist = leaf.conditional_distribution(evidence=evidence_tensor)
        assert dist is not None

    @pytest.mark.parametrize(
        "leaf_cls,out_features,out_channels,num_reps",
        CONDITIONAL_LEAF_PARAMS,
    )
    def test_conditional_leaf_distribution_requires_evidence(
        self, leaf_cls, out_features, out_channels, num_reps
    ):
        query = list(range(out_features))
        evidence = [out_features, out_features + 1]

        param_net = create_conditional_parameter_fn(
            distribution_class=leaf_cls,
            out_features=out_features,
            out_channels=out_channels,
            evidence_size=len(evidence),
            num_repetitions=num_reps,
        )
        scope = Scope(query, evidence=evidence)
        leaf_args = make_leaf_args(
            cls=leaf_cls, out_channels=out_channels, scope=scope, num_repetitions=num_reps
        )
        leaf = leaf_cls(scope=scope, out_channels=out_channels, parameter_fn=param_net, **leaf_args)

        if leaf_cls == leaves.Gamma:
            # Gamma validates attribute access before the generic evidence guard triggers.
            with pytest.raises(AttributeError):
                leaf.conditional_distribution(evidence=None)
        else:
            with pytest.raises(ValueError):
                leaf.conditional_distribution(evidence=None)

    @pytest.mark.parametrize(
        "leaf_cls,out_features,out_channels,num_reps",
        CONDITIONAL_LEAF_PARAMS,
    )
    def test_conditional_leaf_likelihood(self, leaf_cls, out_features, out_channels, num_reps):
        query = list(range(out_features))
        evidence = [out_features, out_features + 1]

        param_net = create_conditional_parameter_fn(
            distribution_class=leaf_cls,
            out_features=out_features,
            out_channels=out_channels,
            evidence_size=len(evidence),
            num_repetitions=num_reps,
        )
        scope = Scope(query, evidence=evidence)
        leaf_args = make_leaf_args(
            cls=leaf_cls, out_channels=out_channels, scope=scope, num_repetitions=num_reps
        )
        leaf = leaf_cls(scope=scope, out_channels=out_channels, parameter_fn=param_net, **leaf_args)

        batch_size = 4
        evidence_data = torch.randn(batch_size, len(evidence))

        dist = leaf.conditional_distribution(evidence=evidence_data)
        query_data = dist.sample()

        assert query_data.shape[0] == batch_size
        assert query_data.shape[1] == len(query)
        assert torch.isfinite(query_data).all()

    @pytest.mark.parametrize(
        "leaf_cls,out_features,out_channels,num_reps",
        CONDITIONAL_LEAF_PARAMS,
    )
    def test_conditional_leaf_sampling(self, leaf_cls, out_features, out_channels, num_reps):
        query = list(range(out_features))
        evidence = [out_features, out_features + 1]

        param_net = create_conditional_parameter_fn(
            distribution_class=leaf_cls,
            out_features=out_features,
            out_channels=out_channels,
            evidence_size=len(evidence),
            num_repetitions=num_reps,
        )
        scope = Scope(query, evidence=evidence)
        leaf_args = make_leaf_args(
            cls=leaf_cls, out_channels=out_channels, scope=scope, num_repetitions=num_reps
        )
        leaf = leaf_cls(scope=scope, out_channels=out_channels, parameter_fn=param_net, **leaf_args)

        batch_size = 4
        evidence_data = torch.randn(batch_size, len(evidence))

        dist = leaf.conditional_distribution(evidence=evidence_data)
        assert dist is not None

    @pytest.mark.parametrize(
        "leaf_cls,out_features,out_channels,num_reps",
        CONDITIONAL_LEAF_GRAD_PARAMS,
    )
    def test_conditional_leaf_parameter_fn_gradients(self, leaf_cls, out_features, out_channels, num_reps):
        query = list(range(out_features))
        evidence = [out_features, out_features + 1]

        param_net = create_conditional_parameter_fn(
            distribution_class=leaf_cls,
            out_features=out_features,
            out_channels=out_channels,
            evidence_size=len(evidence),
            num_repetitions=num_reps,
        )
        scope = Scope(query, evidence=evidence)
        leaf_args = make_leaf_args(
            cls=leaf_cls, out_channels=out_channels, scope=scope, num_repetitions=num_reps
        )
        leaf = leaf_cls(scope=scope, out_channels=out_channels, parameter_fn=param_net, **leaf_args)

        batch_size = 3
        evidence_data = torch.randn(batch_size, len(evidence))
        dist = leaf.conditional_distribution(evidence=evidence_data)
        query_data = dist.sample().reshape(batch_size, -1).float()
        query_data = query_data[:, : len(query)]
        data = torch.cat([query_data, evidence_data], dim=1)

        optimizer = torch.optim.SGD(leaf.parameters(), lr=1e-2)
        optimizer.zero_grad()
        loss = -leaf.log_likelihood(data).sum()
        loss.backward()
        optimizer.step()

        grads = [param.grad for param in leaf.parameter_fn.parameters() if param.requires_grad]
        assert grads
        for grad in grads:
            assert grad is not None
            assert torch.isfinite(grad).all()

    @pytest.mark.parametrize(
        "leaf_cls,out_features,out_channels,num_reps",
        CONDITIONAL_LEAF_PARAMS,
    )
    def test_conditional_leaf_mle_not_supported(self, leaf_cls, out_features, out_channels, num_reps):
        query = list(range(out_features))
        evidence = [out_features, out_features + 1]

        param_net = create_conditional_parameter_fn(
            distribution_class=leaf_cls,
            out_features=out_features,
            out_channels=out_channels,
            evidence_size=len(evidence),
            num_repetitions=num_reps,
        )
        scope = Scope(query, evidence=evidence)
        leaf_args = make_leaf_args(
            cls=leaf_cls, out_channels=out_channels, scope=scope, num_repetitions=num_reps
        )
        leaf = leaf_cls(scope=scope, out_channels=out_channels, parameter_fn=param_net, **leaf_args)

        batch_size = 4
        n_vars = max(query + evidence) + 1
        data = torch.randn(batch_size, n_vars)

        with pytest.raises(RuntimeError):
            leaf.maximum_likelihood_estimation(data=data)

    @pytest.mark.parametrize(
        "leaf_cls,out_features,out_channels,num_reps",
        CONDITIONAL_LEAF_PARAMS,
    )
    def test_conditional_leaf_marginalization_not_supported(
        self, leaf_cls, out_features, out_channels, num_reps
    ):
        query = list(range(out_features))
        evidence = [out_features, out_features + 1]

        param_net = create_conditional_parameter_fn(
            distribution_class=leaf_cls,
            out_features=out_features,
            out_channels=out_channels,
            evidence_size=len(evidence),
            num_repetitions=num_reps,
        )
        scope = Scope(query, evidence=evidence)
        leaf_args = make_leaf_args(
            cls=leaf_cls, out_channels=out_channels, scope=scope, num_repetitions=num_reps
        )
        leaf = leaf_cls(scope=scope, out_channels=out_channels, parameter_fn=param_net, **leaf_args)

        with pytest.raises(RuntimeError):
            leaf.marginalize(marg_rvs=[0])
