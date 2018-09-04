'''
Created on April 15, 2018

@author: Alejandro Molina
'''
import numpy as np
import warnings
from scipy.stats import gamma, lognorm
from sklearn.linear_model import ElasticNet

from spn.structure.leaves.conditional.Conditional import Conditional_Gaussian, Conditional_Poisson, \
    Conditional_Bernoulli
import statsmodels.api as sm

from os.path import dirname

path = dirname(__file__) + "/"


def update_glm_parameters_mle(node, data, scope):  # assume data is tuple (output np array, conditional np array)

    assert len(scope) == 1, 'more than one output variable in scope?'
    data = data[~np.isnan(data)].reshape(data.shape)

    dataOut = data[:, :len(scope)]
    dataIn = data[:, len(scope):]

    assert dataOut.shape[1] == 1, 'more than one output variable in scope?'

    if dataOut.shape[0] == 0:
        return

    dataIn = np.c_[dataIn, np.ones((dataIn.shape[0]))]

    if isinstance(node, Conditional_Gaussian):
        reg = ElasticNet(random_state=0, alpha=0.01, max_iter=2000, fit_intercept=False)
        reg.fit(dataIn, dataOut)
        if reg.n_iter_ < reg.max_iter:
            node.weights = reg.coef_.tolist()
            return



        family = sm.families.Gaussian()
    elif isinstance(node, Conditional_Poisson):
        family = sm.families.Poisson()
    elif isinstance(node, Conditional_Bernoulli):
        family = sm.families.Binomial()
    else:
        raise Exception("Unknown conditional " + str(type(node)))

    glmfit = sm.GLM(dataOut, dataIn, family=family).fit_regularized(alpha=0.0001, maxiter=5)
    node.weights = glmfit.params.tolist()
    return
    try:
        import tensorflow as tf
        import tensorflow_probability as tfp;
        tfd = tfp.distributions

        dataOut = dataOut.reshape(-1)
        w, linear_response, is_converged, num_iter = tfp.glm.fit(
            model_matrix=tf.constant(dataIn),
            response=tf.constant(dataOut),
            model=tfp.glm.Poisson(),
            l2_regularizer=0.0001)
        log_likelihood = tfp.glm.Poisson().log_prob(tf.constant(dataOut), linear_response)

        with tf.Session() as sess:
            [w_, linear_response_, is_converged_, num_iter_, Y_, log_likelihood_] = sess.run(
                [w, linear_response, is_converged, num_iter, tf.constant(dataOut), log_likelihood])

        node.weights = w_
        print("node.weights", node.weights)
        # glmfit = sm.GLM(dataOut, dataIn, family=family).fit_regularized(alpha=0.001)
        # node.weights = glmfit.params
        # # if glmfit.converged is False:
        # #     warnings.warn("Maximum number of iterations reached")
    except Exception:
        glmfit = sm.GLM(dataOut, dataIn, family=family).fit_regularized(alpha=0.0001)
        node.weights = glmfit.params
        print("node.weights with glmfit", node.weights)
        np.savez(path + "tmp_glm_mle_data", dataIn=dataIn, dataOut=dataOut)
