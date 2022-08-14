"""
Created on May 23, 2022

@authors: Philipp Deibert
"""
from typing import List, Optional, Union, Tuple


class SamplingContext:
    """Keeps track of instance ids to sample and which output ids to sample from (relevant for modules with multiple outputs).
    
    Args:
        instance_ids: list of ints representing the instances to sample.
        output_ids: list of lists of ints representing the output ids for the corresponding instances to sample from (relevant for multi-output module).
                    As a shorthand, '[]' implies to sample from all outputs for a given instance id.
    """
    def __init__(self, instance_ids: List[int], output_ids: Optional[List[List[int]]]=None) -> None:

        if output_ids is None:
            # assume sampling over all outputs (i.e. [])
            output_ids = [[] for _ in instance_ids]

        if (len(output_ids) != len(instance_ids)):
            raise ValueError(f"Number of specified instance ids {len(instance_ids)} does not match specified number of output ids {len(output_ids)}.")

        self.instance_ids = instance_ids
        self.output_ids = output_ids

    def group_output_ids(self, n_total) -> Tuple[Union[int, None],List[int]]:
        """TODO"""

        output_id_dict = {}
        
        for instance_id, instance_output_ids in zip(self.instance_ids, self.output_ids):
            if instance_output_ids == []:
                instance_output_ids = list(range(n_total))
            for output_id in instance_output_ids:
                if output_id in output_id_dict:
                    output_id_dict[output_id].append(instance_id)
                else:
                    output_id_dict[output_id] = [instance_id]

        return tuple(output_id_dict.items())


def default_sampling_context(n: int) -> SamplingContext:
    return SamplingContext(list(range(n)), [[] for _ in range(n)])


def init_default_sampling_context(sampling_ctx: Union[SamplingContext, None], n: int) -> SamplingContext:
    return sampling_ctx if sampling_ctx is not None else default_sampling_context(n=n)