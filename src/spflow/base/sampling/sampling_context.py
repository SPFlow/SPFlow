"""
Created on May 23, 2022

@authors: Philipp Deibert
"""

import numpy as np
from multipledispatch import dispatch  # type: ignore
from typing import List, Optional
from spflow.base.structure.module import Module

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

    def group_output_ids(self):

        unique_ids = []
        indices = []
        
        for index, output_id in zip(self.instance_ids, self.output_ids):
            try:
                i = unique_ids.index(output_id)
                indices[i].append(index)
            except ValueError:
                unique_ids.append(output_id)
                indices.append([index])

        return zip(unique_ids, indices)