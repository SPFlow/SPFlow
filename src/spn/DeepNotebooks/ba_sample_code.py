# import all needed data and modules
from tfspn.SPN import SPN
import tensorflow

print(tensorflow.__version__)
print(SPN)

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics.classification import accuracy_score
import time

from mlutils.benchmarks import Chrono
import tensorflow as tf
from tfspn.SPN import Splitting, SPN
from tfspn.tfspn import JointCost, DiscriminativeCost

print('\nDONE')

name = "twospirals"
data = np.loadtxt("experiments/classification/standard/" + name + ".csv", delimiter=",")

x = data[:, :2]
y = data[:, 2]

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=1337)

traindata = np.c_[train_x, train_y]
testdata = np.c_[test_x, test_y]

# print(traindata)
# print(testdata)

spn = SPN.LearnStructure(traindata, 
                         ['continuous', 'continuous', 'categorical'], 
                         min_instances_slice=100,
                         families=["gaussian", "gaussian", "bernoulli"],
                         row_split_method=Splitting.KmeansRows(),
                         col_split_method=Splitting.IndependenceTest())

#spn = SPN.LearnStructure(traindata,
#        #featureNames=feature_names,
#        #domains=domains,
#        featureTypes=['continuous', 'continuous', 'categorical'],
#        row_split_method=Splitting.RandomPartitionConditioningRows(),
#        col_split_method=Splitting.RDCTestOHEpy(threshold=0.75))
#print(traindata.shape)

marg = spn.marginalize([0])

query = np.array([[0.3, 0.3, 0], [0.3, 0.3, 1]])
p_abc = spn.root.eval(query)

marginalized = spn.marginalize([0,1])
query = np.array([[0.3, 0.3]])
p_ab = marginalized.eval(query)

#print(np.exp(p_abc-p_ab))

evaluation = testdata[:,:2]
evaluation_gold_lable = testdata[:,2:]
nan_array = np.zeros((testdata.shape[0], 1))
nan_array[:] = np.nan
query = np.concatenate([evaluation, nan_array], 1)
#print(query)
result = spn.root.mpe_eval(query)
evaluation_result = result[1][:,2:]
#print(np.sum(evaluation_gold_lable == evaluation_result))
#print(testdata.shape)

query = np.array([[0.3, 0.3, np.nan], [0.3, 0.3, 1]])
spn.root.mpe_eval(query)

"""
PLotting of the function


import matplotlib.pyplot as plt

fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')

x_0 = traindata[traindata[:,2]==0][:,0]
x_1 = traindata[traindata[:,2]==1][:,0]
y_0 = traindata[traindata[:,2]==0][:,1]
y_1 = traindata[traindata[:,2]==1][:,1]
colors = ['red']
plt.scatter(x_0, y_0, c=colors, alpha=0.5)
colors = ['blue']
plt.scatter(x_1, y_1, c=colors, alpha=0.5)
plt.show()

def recursive_plotting(node, tabs=0):
    if not node.leaf:
        print("\t"*tabs + str(node.__class__) + "\n")
        for child in node.children:
            recursive_plotting(child, tabs=tabs+1)
    else:
        print("\t" * tabs + str(node))
        
recursive_plotting(spn.root)

class_conditional = spn.marginalize([2]).eval(np.array([[np.nan, np.nan, 0]]))
#print(np.exp(class_conditional))

def get_moment(spn, moment=1, sample_size=10000, rand_gen=None):
    import numpy as np
    import scipy
    if rand_gen is None:
        rand_gen = np.random.RandomState(0)
    method = None
    if moment == 1:
        method = np.mean
    elif moment == 2:
        method = np.var
    elif moment == 3:
        method = scipy.stats.skew
    else:
        raise NotImplementedError("Higher moments then 3 are not supported")
    
    query = np.zeros((sample_size,3))
    query[:] = np.nan
    result = spn.root.sample(query, rand_gen=rand_gen)
    return method(result[1], axis=0)

#print(get_moment(spn))
#print(get_moment(spn, moment=2))
#print(get_moment(spn, moment=3))

marg = spn.marginalize([2])
query = np.zeros((1000,3))
query[:] = np.nan
#print(query)
#nan_array = np.ones((1000,1))
#query = np.concatenate([query, nan_array], axis = 1)
result = marg.sample(query, rand_gen=np.random.RandomState(0))
#print(result)
#print(np.mean(result[1], axis=0))
"""
