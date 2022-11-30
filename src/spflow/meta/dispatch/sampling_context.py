"""Contains the sampling context used in SPFlow

Typical usage example:

    sampling_ctx = SamplingDispatch(instance_ids, output_ids)
"""
from typing import List, Optional, Tuple, Union


class SamplingContext:
    """Class for storing context information during sampling.

    Keeps track of instance indices to sample and which output indices of a module to sample from (relevant for modules with multiple outputs).

    Attributes:
        instance_ids:
            List of integers representing the instances of a data set to sample.
            Required to correctly place sampled values into target data set and take potential evidence into account.
        output_ids:
            List of lists of integers representing the output ids for the corresponding instances to sample from (relevant for multi-output module).
            As a shorthand convention, ``[]`` implies to sample from all outputs for a given instance.
    """

    def __init__(
        self,
        instance_ids: List[int],
        output_ids: Optional[List[List[int]]] = None,
    ) -> None:
        """Initializes 'SamplingContext' object.

        Args:
            instance_ids:
                List of integers representing the instances of a data set to sample.
                Required to correctly place sampled values into target data set and take potential evidence into account.
            output_ids:
                Optional list of lists of integers representing the output ids for the corresponding instances to sample from (relevant for multi-output module).
                As a shorthand convention, an empty list (``[]``) implies to sample from all outputs for the corresponding instance.
                Defaults to None, in which case the output indices for all instances are set to an empty list (``[]``).

        Raises:
            ValueError: Number of instance indices does not match number of output indices.
        """
        if output_ids is None:
            # assume sampling over all outputs (i.e. [])
            output_ids = [[] for _ in instance_ids]

        if len(output_ids) != len(instance_ids):
            raise ValueError(
                f"Number of specified instance ids {len(instance_ids)} does not match specified number of output ids {len(output_ids)}."
            )

        self.instance_ids = instance_ids
        self.output_ids = output_ids

    def group_output_ids(
        self, n_total: int
    ) -> List[Tuple[Union[int, None], List[int]]]:
        """Groups instances in the sampling context by their output indices.

        Args:
            n_total:
                Integer indicating the number of outputs of the module in question.
                Required to initialize the output ids in case of an empty list of output ids (in which case all output ids are sampled from).

        Returns:
            List of pairs (tuples) of an output index (or None) and a list of all instance indices that sample from the corresponding output index.
        """
        output_id_dict = {}

        for instance_id, instance_output_ids in zip(
            self.instance_ids, self.output_ids
        ):
            if instance_output_ids == []:
                instance_output_ids = list(range(n_total))
            for output_id in instance_output_ids:
                if output_id in output_id_dict:
                    output_id_dict[output_id].append(instance_id)
                else:
                    output_id_dict[output_id] = [instance_id]

        return tuple(output_id_dict.items())

    def unique_outputs_ids(
        self, return_indices: bool = False
    ) -> Union[List[List[int]], Tuple[List[List[int]], List[List[int]]]]:
        """Return the list of unique lists of output indices, not the individual indices

        Args:
            return_indices:
                Boolean indicating whether or not to additionally return the indices of the unique lists.
                Defaults to False.

        Returns:
            List of lists of integers, containing the unique lists of output indices in the sampling context.
            If ``return_indices`` is set to True, then an additional list of lists of integers is return, containing the indices for the unique lists.
        """
        unique_lists = []
        indices = []

        for i, output_ids in enumerate(self.output_ids):
            if not output_ids in unique_lists:
                unique_lists.append(output_ids)
                indices.append([i])
            else:
                idx = unique_lists.index(output_ids)
                indices[idx].append(i)

        if return_indices:
            return unique_lists, indices
        else:
            return unique_lists


def default_sampling_context(n: int) -> SamplingContext:
    """Returns an initialized ``SamplingContext`` object.

    Args:
        n:
            Integer specifying the number of instance indices to intialize.

    Returns:
        Sampling context initialized with instance indices from 0 to ``n`` and corresponding output indices as ``[]``.
    """
    return SamplingContext(list(range(n)), [[] for _ in range(n)])


def init_default_sampling_context(
    sampling_ctx: Union[SamplingContext, None], n: int
) -> SamplingContext:
    """Initializes sampling context, if it is not already initialized.

    Args
        sampling_ctx:
            ``SamplingContext`` object or None.
        n:
            Integer specifying the number of instance indices to intialize.

    Returns:
        Original sampling context if not None or a new initialized sampling context.
    """
    return (
        sampling_ctx
        if sampling_ctx is not None
        else default_sampling_context(n=n)
    )
