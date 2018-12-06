from spn.algorithms.MPE import add_node_mpe

from spn.structure.leaves.histogram.Moment import add_histogram_moment_support
from spn.structure.leaves.histogram.Inference import add_histogram_inference_support
from spn.structure.leaves.histogram.MPE import add_histogram_mpe_support
from spn.structure.leaves.histogram.Gradients import add_histogram_gradient_support

from spn.structure.leaves.parametric.Moment import add_parametric_moment_support
from spn.structure.leaves.parametric.Inference import add_parametric_inference_support
from spn.structure.leaves.parametric.MPE import add_parametric_mpe_support
from spn.structure.leaves.parametric.Sampling import add_parametric_sampling_support

from spn.structure.leaves.piecewise.Moment import add_piecewise_moment_support
from spn.structure.leaves.piecewise.Inference import add_piecewise_inference_support
from spn.structure.leaves.piecewise.MPE import add_piecewise_mpe_support
from spn.structure.leaves.piecewise.Gradients import add_piecewise_linear_gradient_support

from spn.structure.leaves.cltree.Expectation import add_cltree_expectation_support
from spn.structure.leaves.cltree.Inference import add_cltree_inference_support
from spn.structure.leaves.cltree.MPE import add_cltree_mpe_support
from spn.structure.leaves.cltree.Sampling import add_cltree_sampling_support

add_parametric_sampling_support()
add_parametric_inference_support()
add_parametric_moment_support()
add_parametric_mpe_support()

add_piecewise_inference_support()
add_parametric_mpe_support()
add_piecewise_moment_support()
add_piecewise_mpe_support()

add_histogram_inference_support()
add_histogram_moment_support()
add_histogram_mpe_support()

add_cltree_sampling_support()
add_cltree_inference_support()
add_cltree_expectation_support()
add_cltree_mpe_support()
