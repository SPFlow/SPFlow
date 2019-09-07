import spn
#from spn.structure.leaves.mvgauss.MVG import *
from spn.io.Text import *
import sys
from spn.structure.leaves.parametric.Parametric import *
from spn.structure.leaves.parametric.MLE import *
from spn.algorithms.MPE import mpe
from spn.structure.prometheus.disc import *
from scipy.stats import multivariate_normal as mn
#from spn.structure.prometheus.disc import *

node = MultivariateGaussian(np.inf, np.inf)
data = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 1, 3, 3, 6, 2]).reshape(-1, 2)
update_parametric_parameters_mle(node, data)

print(node.mean, node.sigma)

print(node.scope)

dummydata = np.asarray([[1, 2, 4, 8], [2.1, 4.1, 8.1, 16.1], [
                       4.1, 8.1, 16.1, 32.1], [8.8, 16.5, 32.3, 64.2]])
dummyscope = list([0, 1, 2, 3])

spn = MultivariateGaussian(np.inf, np.inf)

update_parametric_parameters_mle(spn, dummydata)

print(spn.mean)
print(spn.sigma)

spn.scope = dummyscope

#print(mn.pdf(spn.mean, spn.mean, spn.cov))

print(spn.scope)

dummydata = np.asarray([[np.nan, 2.0, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, 64.3]])

print(np.shape(dummydata))
print(np.shape(np.asarray(spn.mean)))
print(np.shape(np.asarray(spn.sigma)))

print(mpe(spn, dummydata))

print(spn_to_str_equation(spn))

recreate = (str_to_spn(spn_to_str_equation(spn)))

print(spn_to_str_equation(recreate))

print(recreate.mean)
print(recreate.sigma)

arr = np.load('./test.npy')
teststruct = prometheus(arr, 1, itermult=0, leafsize=4, maxsize=6)

testspn = str_to_spn(teststruct)

recreate = spn_to_str_equation(testspn)

file = open('./ca.txt', 'w')
file.write(teststruct)
file.close()
