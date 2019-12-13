import json
import logging
import math
import numpy as np
import os
import re

from itertools import repeat
from spn.algorithms.Inference import log_likelihood
from spn.io.Text import spn_to_str_equation, to_JSON, spn_to_str_ref_graph
from spn.structure.Base import Context, Product, Sum, assign_ids, rebuild_scopes_bottom_up
from spn.structure.leaves.histogram.Histograms import create_histogram_leaf
from spn.structure.StatisticalTypes import MetaType

logger = logging.getLogger(__name__)

# These values will be used by 'create_spflow_spn_histogram' and 'generate_benchmark' to generate matching data ranges
GENBENCH_LO = 0
GENBENCH_HI = 72


def create_spflow_spn_histogram(num_features, num_samples_training=1000):
    hist1 = []
    hist2 = []

    # Prepare training data for histograms
    data1 = np.random.randint(low=GENBENCH_LO, high=GENBENCH_HI, size=(num_samples_training, num_features))
    data2 = np.random.randint(low=GENBENCH_LO, high=GENBENCH_HI, size=(num_samples_training, num_features))

    meta_types = list(repeat(MetaType.DISCRETE, num_features))
    ds_context1 = Context(meta_types)
    ds_context2 = Context(meta_types)
    ds_context1.add_domains(data1)
    ds_context2.add_domains(data2)

    for i in range(num_features):
        h1 = create_histogram_leaf(data1[:, [i]], ds_context1, scope=[i], alpha=False)
        h2 = create_histogram_leaf(data2[:, [i]], ds_context2, scope=[i], alpha=False)
        hist1.append(h1)
        hist2.append(h2)

    prods1 = []
    prods2 = []
    for i in range(0, num_features, 2):
        # Allow odd number of features and try produce different indices
        ii = (i + 1) if (i + 1) < num_features else (i - 1) if (i - 1) >= 0 else i
        p1 = Product([hist1[i], hist1[ii]])
        p2 = Product([hist2[i], hist2[ii]])
        prods1.append(p1)
        prods2.append(p2)

    sums = []
    for i in range(math.ceil(num_features / 2)):
        s = Sum(weights=[0.5, 0.5], children=[prods1[i], prods2[i]])
        sums.append(s)

    spflow_spn = Product(sums)
    assign_ids(spflow_spn)
    rebuild_scopes_bottom_up(spflow_spn)

    return spflow_spn


def generate_benchmark(num_features, abs_store_path=os.getcwd(),
                       rng_seed=17, data_size=100, data_prefix="",
                       spn_gen=create_spflow_spn_histogram, eval_func=log_likelihood):
    np.random.seed(rng_seed)
    err = False

    # Create random SPN with given size / number of features
    if type(num_features) is int and num_features > 0:
        spn = spn_gen(num_features)
    else:
        logger.warning("Unsupported type/value for 'num_features': {}. Expected: int > 0. Was: '{}'"
                       .format(type(num_features), num_features))
        spn = None
        err = True

    # Sanity checks
    if type(data_size) is not int or data_size <= 0:
        logger.warning("Unsupported type for 'data_size': {}. Expected: int > 0. Was: '{}'"
                       .format(type(data_size), data_size))
        err = True

    if type(data_prefix) is not str:
        logger.warning("Unsupported type for 'data_prefix': {}. Expected: str. Was: '{}'"
                       .format(type(data_prefix), data_prefix))
        err = True

    if err:
        logger.warning("Error during benchmark generation: No benchmark data written.")
        return spn

    # Create input and output-data
    ds_input = np.random.randint(low=GENBENCH_LO, high=GENBENCH_HI, size=(data_size, num_features))
    # Evaluating the generated input-dataset will yield the output-dataset
    out = eval_func(spn, ds_input)
    ds_output = np.array(out)

    # Format input / output
    # Set printoptions for input and output -- there MUST NOT be a linewidth constraint (prevents linefeeds)
    np.set_printoptions(linewidth=np.inf, sign=' ', formatter={'float': lambda x: format(x, '6.18e')})
    ds_input_str = np.array2string(ds_input, separator=";")
    ds_output_str = np.array2string(ds_output, separator=";")
    # Remove '[' ']]' and '];' to be consistent with the previous format
    regex_remove = r'([ \[]|(\]([;\]])?))'
    ds_input_str = re.sub(regex_remove, '', ds_input_str)
    ds_output_str = re.sub(regex_remove, '', ds_output_str)

    # Sanity-check (and possible creation) of the given absolute path
    # Make sure there are no redundant separators and eventually convert (back-)slashes
    path = os.path.normcase("{}/".format(os.path.normpath(abs_store_path)))
    if not os.path.isdir(path):
        os.makedirs(path)

    # Prepare filenames
    fn_input = "{}{}inputdata.txt".format(path, data_prefix)
    fn_output = "{}{}outputdata.txt".format(path, data_prefix)
    fn_equation = "{}equation.txt".format(path)
    fn_json = "{}spn.json".format(path)
    fn_spn = "{}structure.spn".format(path)

    # Store all information
    with open(fn_input, "w") as file:
        file.write(ds_input_str)

    with open(fn_output, "w") as file:
        file.write(ds_output_str)

    # Set printoptions for SPN output: with default linewidth (easier to read)
    np.set_printoptions(sign=' ', formatter={'float': lambda x: format(x, '6.18e')})

    with open(fn_equation, "w") as file:
        file.write(spn_to_str_equation(spn))

    with open(fn_json, "w") as file:
        parsed = json.loads(to_JSON(spn))
        file.write(json.dumps(parsed, indent=4, sort_keys=True))

    with open(fn_spn, "w") as file:
        np.set_printoptions(sign=' ',
                            formatter={'float': lambda x: format(x, '1.10'), 'int': lambda x: '{}.'.format(x)})
        # Get the SPN-scope and append it to the str_ref_graph
        # Note that each variable ID must be prefixed with a letter (here: 'V')
        scope = ';'.join(['V{}'.format(x) for x in spn.scope])
        str_ref_graph = "{}\n# {}".format(spn_to_str_ref_graph(spn), scope)
        file.write(str_ref_graph)

    logger.info("Benchmark created at '{}'.".format(abs_store_path))
    return spn
