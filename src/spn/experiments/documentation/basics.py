'''
Created on July 24, 2018

@author: Alejandro Molina
'''
from numpy.random.mtrand import RandomState

from spn.algorithms.Inference import likelihood
from spn.algorithms.LearningWrappers import learn_parametric, learn_classifier
from spn.algorithms.MPE import mpe
from spn.algorithms.Marginalization import marginalize
from spn.algorithms.Sampling import sample_instances
from spn.algorithms.Statistics import get_structure_stats
from spn.algorithms.Validity import is_valid
from spn.io.Graphics import plot_spn
from spn.io.Text import spn_to_str_equation
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
import numpy as np

if __name__ == '__main__':
    spn = 0.4 * (Categorical(p=[0.2, 0.8], scope=0) * \
                 (0.3 * (Categorical(p=[0.3, 0.7], scope=1) * Categorical(p=[0.4, 0.6], scope=2)) + \
                  0.7 * (Categorical(p=[0.5, 0.5], scope=1) * Categorical(p=[0.6, 0.4], scope=2)))) \
          + 0.6 * (Categorical(p=[0.2, 0.8], scope=0) * \
                   Categorical(p=[0.3, 0.7], scope=1) * \
                   Categorical(p=[0.4, 0.6], scope=2))

    print(spn_to_str_equation(spn))
    plot_spn(spn, '/tmp/basicspn.pdf')

    print(spn_to_str_equation(marginalize(spn, [0])))

    print(likelihood(spn, np.array([1, 0, 1]).reshape(-1, 3)))

    print(likelihood(spn, np.array([1, 0, np.nan]).reshape(-1, 3)))

    print(is_valid(spn))

    print(get_structure_stats(spn))

    print(sample_instances(spn, np.array([np.nan, 0, 0]).reshape(-1, 3), RandomState(123)))

    print(mpe(spn, np.array([np.nan, 0, 0]).reshape(-1, 3)))

    train_data = np.c_[np.r_[np.random.normal(5, 1, (500, 2)), np.random.normal(15, 1, (500, 2))],
                       np.r_[np.zeros((500, 1)), np.ones((500, 1))]]

    spn = learn_classifier(train_data,
                           Context(parametric_type=[Gaussian, Gaussian, Categorical]).add_domains(train_data),
                           learn_parametric, 2)

    print(mpe(spn, np.array([3.0, 4.0, np.nan, 12.0, 18.0, np.nan]).reshape(-1, 3)))
