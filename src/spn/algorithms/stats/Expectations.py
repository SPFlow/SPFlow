from spn.algorithms.Inference import log_likelihood
from spn.io.Text import spn_to_str_equation
from spn.structure.Base import Leaf
from spn.structure.leaves.parametric.Parametric import Gaussian
from spn.structure.leaves.parametric.Text import add_parametric_text_support


def Expectation(spn, feature_scope, evidence):
    def leaf_expectation(node, data, dtype=np.float64, node_log_likelihood=None):

        if node.scope[0] == feature_scope:
            #return the expectation of the leaf
            pass


        return 1.0

    log_expectation = log_likelihood(spn, evidence, node_log_likelihood={Leaf: leaf_expectation})

    return np.exp(log_expectation)


if __name__ == '__main__':
    add_parametric_text_support()

    spn = 0.3 * (Gaussian(1.0, 1.0, scope=[0]) * Gaussian(5.0, 1.0, scope=[1])) + \
          0.7 * (Gaussian(10.0, 1.0, scope=[0]) * Gaussian(15.0, 1.0, scope=[1]))

    print(spn_to_str_equation(spn))
