"""Contract tests for EinsumLayer."""

import pytest

from spflow.modules.einsum import EinsumLayer
from tests.modules.einsum.contracts import EinsumLikeContractTests, make_two_inputs_for_contract
from tests.utils.leaves import make_normal_leaf


@pytest.mark.contract
class TestEinsumLayerContract(EinsumLikeContractTests):
    """Shared behavior contract implementation for EinsumLayer."""

    __test__ = True

    def layer_cls(self):
        return EinsumLayer

    def make_single_input(self, in_channels: int, out_channels: int, in_features: int, num_reps: int):
        inputs = make_normal_leaf(
            out_features=in_features,
            out_channels=in_channels,
            num_repetitions=num_reps,
        )
        return EinsumLayer(inputs=inputs, out_channels=out_channels, num_repetitions=num_reps)

    def make_two_inputs(self, in_channels: int, out_channels: int, in_features: int, num_reps: int):
        left, right = make_two_inputs_for_contract(
            in_channels=in_channels,
            in_features=in_features,
            num_repetitions=num_reps,
        )
        return EinsumLayer(inputs=[left, right], out_channels=out_channels, num_repetitions=num_reps)

    def expected_single_weight_shape(
        self, in_channels: int, out_channels: int, in_features: int, num_reps: int
    ) -> tuple[int, ...]:
        return (in_features // 2, out_channels, num_reps, in_channels, in_channels)

    def assert_module_specific_two_input_channel_behavior(self) -> None:
        left, right = make_two_inputs_for_contract(
            in_channels=2,
            in_features=2,
            num_repetitions=1,
            left_channels=2,
            right_channels=3,
        )
        module = EinsumLayer(inputs=[left, right], out_channels=4, num_repetitions=1)
        assert module.weights_shape == (2, 4, 1, 2, 3)

    def input_channel_reduce_dims(self) -> tuple[int, ...]:
        return (-2, -1)
