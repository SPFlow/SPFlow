'''
Created on July 24, 2018

@author: Alejandro Molina
'''


def test():
    from numpy.random.mtrand import RandomState

    from spn.algorithms.Inference import likelihood, log_likelihood, add_node_likelihood
    from spn.algorithms.LearningWrappers import learn_parametric, learn_classifier
    from spn.algorithms.MPE import mpe
    from spn.algorithms.Marginalization import marginalize
    from spn.algorithms.Sampling import sample_instances
    from spn.algorithms.Statistics import get_structure_stats
    from spn.algorithms.Validity import is_valid
    from spn.gpu.TensorFlow import spn_to_tf_graph, eval_tf, optimize_tf
    from spn.io.Graphics import plot_spn
    from spn.io.Text import spn_to_str_equation
    from spn.structure.Base import Context, Leaf
    from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
    import numpy as np


def create_SPN():
    from spn.structure.leaves.parametric.Parametric import Categorical

    spn = 0.4 * (Categorical(p=[0.2, 0.8], scope=0) * \
                 (0.3 * (Categorical(p=[0.3, 0.7], scope=1) * Categorical(p=[0.4, 0.6], scope=2)) + \
                  0.7 * (Categorical(p=[0.5, 0.5], scope=1) * Categorical(p=[0.6, 0.4], scope=2)))) \
          + 0.6 * (Categorical(p=[0.2, 0.8], scope=0) * \
                   Categorical(p=[0.3, 0.7], scope=1) * \
                   Categorical(p=[0.4, 0.6], scope=2))

    return spn


def to_str():
    spn = create_SPN()
    spn_marg = marginalize()

    from spn.io.Text import spn_to_str_equation

    print(spn_to_str_equation(spn))
    print(spn_to_str_equation(spn_marg))


def plot():
    spn = create_SPN()
    spn_marg = marginalize()

    from spn.io.Graphics import plot_spn

    plot_spn(spn, 'basicspn.png')
    plot_spn(spn_marg, 'marginalspn.png')


def marginalize():
    spn = create_SPN()

    from spn.algorithms.Marginalization import marginalize

    spn_marg = marginalize(spn, [1, 2])

    return spn_marg


def inference():
    import numpy as np

    spn = create_SPN()
    spn_marg = marginalize()

    test_data = np.array([1.0, 0.0, 1.0]).reshape(-1, 3)

    from spn.algorithms.Inference import log_likelihood

    ll = log_likelihood(spn, test_data)
    print("python ll", ll, np.exp(ll))

    llm = log_likelihood(spn_marg, test_data)
    print("python ll spn_marg", llm, np.exp(llm))

    test_data2 = np.array([np.nan, 0.0, 1.0]).reshape(-1, 3)
    llom = log_likelihood(spn, test_data2)
    print("python ll spn with nan", llom, np.exp(llom))


def tensorflow():
    import numpy as np

    spn = create_SPN()

    test_data = np.array([1.0, 0.0, 1.0]).reshape(-1, 3)

    from spn.gpu.TensorFlow import eval_tf
    lltf = eval_tf(spn, test_data)
    print("tensorflow ll", lltf, np.exp(lltf))

    from spn.algorithms.Inference import log_likelihood
    from spn.gpu.TensorFlow import optimize_tf
    optimized_spn = optimize_tf(spn, test_data)
    lloptimized = log_likelihood(optimized_spn, test_data)
    print("tensorflow optimized ll", lloptimized, np.exp(lloptimized))


def valid():
    spn = create_SPN()
    spn_marg = marginalize()
    from spn.algorithms.Validity import is_valid
    print(is_valid(spn))
    print(is_valid(spn_marg))


def stats():
    spn = create_SPN()
    from spn.algorithms.Statistics import get_structure_stats
    print(get_structure_stats(spn))


def sample():
    spn = create_SPN()
    import numpy as np

    from numpy.random.mtrand import RandomState
    from spn.algorithms.Sampling import sample_instances
    print(sample_instances(spn, np.array([np.nan, 0, 0] * 5).reshape(-1, 3), RandomState(123)))
    print(sample_instances(spn, np.array([np.nan, np.nan, np.nan] * 5).reshape(-1, 3), RandomState(123)))


def classification():
    import numpy as np
    np.random.seed(123)
    train_data = np.c_[np.r_[np.random.normal(5, 1, (500, 2)), np.random.normal(10, 1, (500, 2))],
                       np.r_[np.zeros((500, 1)), np.ones((500, 1))]]

    centers = [[5, 5], [10, 10]]

    import matplotlib.pyplot as plt
    colors = ['#bda36b', '#7aaab4']
    plt.figure()
    #plt.hold(True)
    for k, col in zip(range(2), colors):
        my_members = train_data[:, 2] == k
        plt.plot(train_data[my_members, 0], train_data[my_members, 1], 'w', markerfacecolor=col, marker='.')
        plt.plot(centers[k][0], centers[k][1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
    plt.title('Training Data')
    plt.grid(True)
    plt.savefig("classification_training_data.png", bbox_inches='tight', pad_inches=0)

    from spn.algorithms.LearningWrappers import learn_parametric, learn_classifier
    from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
    from spn.structure.Base import Context
    spn_classification = learn_classifier(train_data,
                           Context(parametric_type=[Gaussian, Gaussian, Categorical]).add_domains(train_data),
                           learn_parametric, 2)

    test_classification = np.array([3.0, 4.0, np.nan, 12.0, 18.0, np.nan]).reshape(-1, 3)

    from spn.algorithms.MPE import mpe
    print(mpe(spn_classification, test_classification))

if __name__ == '__main__':
    create_SPN()
    to_str()
    plot()
    inference()
    # tensorflow()
    valid()
    stats()
    sample()
    classification()

    0 / 0

    print(mpe(spn, np.array([np.nan, 0, 0]).reshape(-1, 3)))

    spn = learn_classifier(train_data,
                           Context(parametric_type=[Gaussian, Gaussian, Categorical]).add_domains(train_data),
                           learn_parametric, 2)

    print(mpe(spn, np.array([3.0, 4.0, np.nan, 12.0, 18.0, np.nan]).reshape(-1, 3)))


    class Pareto(Leaf):
        def __init__(self, a, scope=None):
            Leaf.__init__(self, scope=scope)
            self.a = a


    def pareto_likelihood(node, data, dtype=np.float64):
        probs = np.ones((data.shape[0], 1), dtype=dtype)
        from scipy.stats import pareto
        probs[:] = pareto.pdf(data[:, node.scope], node.a)
        return probs


    add_node_likelihood(Pareto, pareto_likelihood)

    spn = 0.3 * Pareto(2.0, scope=0) + 0.7 * Pareto(3.0, scope=0)

    print("python", log_likelihood(spn, np.array([1.0, 1.5]).reshape(-1, 1)))
