[![pypi](https://img.shields.io/pypi/v/spflow.svg)](https://pypi.org/project/spflow/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Build Status](https://travis-ci.com/SPFlow/SPFlow.svg?branch=master)](https://travis-ci.com/SPFlow/SPFlow)


# SPFlow: An Easy and Extensible Library for Sum-Product Networks

SPFlow, an open-source Python library providing a simple interface to inference,
learning  and  manipulation  routines  for  deep  and  tractable  probabilistic  models called Sum-Product Networks (SPNs).
The library allows one to quickly create SPNs both from data and through a domain specific language (DSL).
It efficiently implements several probabilistic inference routines like computing marginals, conditionals and (approximate) most probable explanations (MPEs)
along with sampling as well as utilities for serializing,plotting and structure statistics on an SPN.

Furthermore, SPFlow is extremely extensible and customizable, allowing users to  promptly  create  new  inference  and  learning  routines
by  injecting  custom  code  into  a light-weight functional-oriented API framework.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Installing

To install the latest released version of SPFlow using pip

```sh
pip3 install spflow
```

An AUR package is available for Arch Linux. The PKGBUILD should automatically apply a patch for SPFlow to work with Tensorflow 2.

```sh
yay -S python-spflow
```

## Examples

We start by creating an SPN. Using a Domain-Specific Language (DSL), we can quickly create an SPN of categorical
leave nodes like this:


```python
from spn.structure.leaves.parametric.Parametric import Categorical

spn = 0.4 * (Categorical(p=[0.2, 0.8], scope=0) *
             (0.3 * (Categorical(p=[0.3, 0.7], scope=1) *
                     Categorical(p=[0.4, 0.6], scope=2))
            + 0.7 * (Categorical(p=[0.5, 0.5], scope=1) *
                     Categorical(p=[0.6, 0.4], scope=2)))) \
    + 0.6 * (Categorical(p=[0.2, 0.8], scope=0) *
             Categorical(p=[0.3, 0.7], scope=1) *
             Categorical(p=[0.4, 0.6], scope=2))
```

We can create the same SPN using the object hierarchy:

```python
from spn.structure.leaves.parametric.Parametric import Categorical

from spn.structure.Base import Sum, Product

from spn.structure.Base import assign_ids, rebuild_scopes_bottom_up


p0 = Product(children=[Categorical(p=[0.3, 0.7], scope=1), Categorical(p=[0.4, 0.6], scope=2)])
p1 = Product(children=[Categorical(p=[0.5, 0.5], scope=1), Categorical(p=[0.6, 0.4], scope=2)])
s1 = Sum(weights=[0.3, 0.7], children=[p0, p1])
p2 = Product(children=[Categorical(p=[0.2, 0.8], scope=0), s1])
p3 = Product(children=[Categorical(p=[0.2, 0.8], scope=0), Categorical(p=[0.3, 0.7], scope=1)])
p4 = Product(children=[p3, Categorical(p=[0.4, 0.6], scope=2)])
spn = Sum(weights=[0.4, 0.6], children=[p2, p4])

assign_ids(spn)
rebuild_scopes_bottom_up(spn)
```

The p parameter indicates the probabilities, and the scope indicates the variable we are modeling.


We can now visualize the SPN using:

```python
from spn.io.Graphics import plot_spn

plot_spn(spn, 'basicspn.png')
```

![basicspn.png](https://github.com/SPFlow/SPFlow/blob/master/Documentation/basicspn.png)

Marginalizing an SPN means summing out all the other non-relevant variables.
So, if we want to marginalize the above SPN and sum out all other variables leaving only variables 1 and 2, we can do:

```python
from spn.algorithms.Marginalization import marginalize

spn_marg = marginalize(spn, [1,2])
```
Here, we marginalize all the variables not in [1,2], and create a *NEW* structure that knows nothing about the previous one
nor about the variable 0.

We can use this new spn to do all the operations we are interested in. That means, we can also plot it!
```python
plot_spn(spn_marg, 'marginalspn.png')
```
![basicspn.png](https://github.com/SPFlow/SPFlow/blob/master/Documentation/marginalspn.png)

We can also dump the SPN as text:
```python
from spn.io.Text import spn_to_str_equation
txt = spn_to_str_equation(spn_marg)
print(txt)
```
And the output is:
```python
(0.6*((Categorical(V1|p=[0.3, 0.7]) * Categorical(V2|p=[0.4, 0.6]))) + 0.12000000000000002*((Categorical(V1|p=[0.3, 0.7]) * Categorical(V2|p=[0.4, 0.6]))) + 0.27999999999999997*((Categorical(V1|p=[0.5, 0.5]) * Categorical(V2|p=[0.6, 0.4]))))
```

However, the most interesting aspect of SPNs is the tractable inference. Here is an example on how to evaluate the SPNs from above.
Since we have 3 variables, we want to create a 2D numpy array of 3 columns and 1 row.
```python
import numpy as np
test_data = np.array([1.0, 0.0, 1.0]).reshape(-1, 3)
```

We then compute the log-likelihood:
```python
from spn.algorithms.Inference import log_likelihood

ll = log_likelihood(spn, test_data)
print(ll, np.exp(ll))
```

And the output is:
```python
[[-1.90730501]] [[0.14848]]
```

We can also compute the log-likelihood of the marginal SPN:
```python
llm = log_likelihood(spn_marg, test_data)
print(llm, np.exp(llm))
```
Note that we used the same test_data input, as the SPN is still expecting a numpy array with data at columns 1 and 2, ignoring column 0.
The output is:
```python
[[-1.68416146]] [[0.1856]]
```

Another alternative, is marginal inference on the original SPN. This is done by setting as np.nan the feature we want to marginalize on the fly.
It does not change the structure.

```python
test_data2 = np.array([np.nan, 0.0, 1.0]).reshape(-1, 3)
llom =  log_likelihood(spn, test_data2)
print(llom, np.exp(llom))
```

The output is exactly the same as the evaluation of the marginal spn:
```python
[[-1.68416146]] [[0.1856]]
```

We can use tensorflow to do the evaluation in a GPU:
```python
from spn.gpu.TensorFlow import eval_tf
lltf = eval_tf(spn, test_data)
print(lltf, np.exp(lltf))
```
The output is as expected, equal to the one in python:
```python
[[-1.90730501]] [[0.14848]]
```

We can also use tensorflow to do the parameter optimization in a GPU:
```python
from spn.gpu.TensorFlow import optimize_tf
optimized_spn = optimize_tf(spn, test_data)
lloptimized = log_likelihood(optimized_spn, test_data)
print(lloptimized, np.exp(lloptimized))
```
The output is of course, higher likelihoods:
```python
[[-1.38152628]] [[0.25119487]]
```

We can generate new samples that follow the joint distribution captured by the SPN!
```python
from numpy.random.mtrand import RandomState
from spn.algorithms.Sampling import sample_instances
print(sample_instances(spn, np.array([np.nan, np.nan, np.nan] * 5).reshape(-1, 3), RandomState(123)))
```
Here we created 5 new instances that follow the distribution
```python
[[0. 1. 0.]
 [1. 0. 0.]
 [1. 1. 0.]
 [1. 1. 1.]
 [1. 1. 0.]]
```
the np.nan values indicate the columns we want to sample.

We can also do conditional sampling, that is, if we have evidence for some of the variables we can pass that information
to the SPN and sample for the rest of the variables:
```python
from numpy.random.mtrand import RandomState
from spn.algorithms.Sampling import sample_instances
print(sample_instances(spn, np.array([np.nan, 0, 0] * 5).reshape(-1, 3), RandomState(123)))
```
Here we created 5 new instances whose evidence is V1=0 and V2=0
```python
[[0. 0. 0.]
 [1. 0. 0.]
 [0. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]]
```

We can do classification, by learning an SPN from data and then comparing the probabilities for the given classes:
Imagine we have the following dataset:

![basicspn.png](https://github.com/SPFlow/SPFlow/blob/master/Documentation/classification_training_data.png)

generated by two gaussians with means (5,5) and (10,10), and we label the cluster at (5,5) to be class 0 and the cluster at (10,10) to be class 1.

```python
np.random.seed(123)
train_data = np.c_[np.r_[np.random.normal(5, 1, (500, 2)), np.random.normal(10, 1, (500, 2))],
                   np.r_[np.zeros((500, 1)), np.ones((500, 1))]]
```

We can learn an SPN from data:

```python
from spn.algorithms.LearningWrappers import learn_parametric, learn_classifier
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
from spn.structure.Base import Context
spn_classification = learn_classifier(train_data,
                       Context(parametric_types=[Gaussian, Gaussian, Categorical]).add_domains(train_data),
                       learn_parametric, 2)
```
Here, we model our problem as containing 3 features, two Gaussians for the coordinates and one Categorical for the label.
We specify that the label is in column 2, and create the corresponding SPN.

Now, imagine we want to classify two instances, one located at (3,4) and another one at (12,8).
To do that, we first create an array with two rows and 3 columns. We set the last column to np.nan to indicate that we don't know the labels.
And we set the rest of the values in the 2D array accordingly.

```python
test_classification = np.array([3.0, 4.0, np.nan, 12.0, 18.0, np.nan]).reshape(-1, 3)
```
the first row is the first instance, the second row is the second instance.
```python
[[ 3.  4. nan]
 [12. 18. nan]]
```

We can do classification via approximate most probable explanation (MPE).
Here, we expect the first instance to be labeled as 0 and the second one as 1.
```python
from spn.algorithms.MPE import mpe
print(mpe(spn_classification, test_classification))
```
as we can see, both instances are classified correctly, as the correct label is set in the last column
```python
[[ 3.  4.  0.]
 [12. 18.  1.]]
```

We can learn an MSPN and a parametric SPN from data:

```python
import numpy as np
np.random.seed(123)

a = np.random.randint(2, size=1000).reshape(-1, 1)
b = np.random.randint(3, size=1000).reshape(-1, 1)
c = np.r_[np.random.normal(10, 5, (300, 1)), np.random.normal(20, 10, (700, 1))]
d = 5 * a + 3 * b + c
train_data = np.c_[a, b, c, d]

```
Here, we have a dataset containing four features, two Discrete and two Real valued.

We can learn an MSPN with:
```python
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType

ds_context = Context(meta_types=[MetaType.DISCRETE, MetaType.DISCRETE, MetaType.REAL, MetaType.REAL])
ds_context.add_domains(train_data)

from spn.algorithms.LearningWrappers import learn_mspn

mspn = learn_mspn(train_data, ds_context, min_instances_slice=20)
```

We can learn a parametric SPN with:
```python
from spn.structure.Base import Context
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian

ds_context = Context(parametric_types=[Categorical, Categorical, Gaussian, Gaussian]).add_domains(train_data)

from spn.algorithms.LearningWrappers import learn_parametric

spn = learn_parametric(train_data, ds_context, min_instances_slice=20)
```

### Multivariate leaf

We can learn a SPN with multivariate leaf. For instance SPN with Chow Liu tree (CLTs) as multivariate leaf can be learned with:
```python
import numpy as np
np.random.seed(123)

train_data = np.random.binomial(1, [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.1], size=(100,10))

from spn.structure.leaves.cltree.CLTree import create_cltree_leaf
from spn.structure.Base import Context
from spn.structure.leaves.parametric.Parametric import Bernoulli
from spn.algorithms.LearningWrappers import learn_parametric
from spn.algorithms.Inference import log_likelihood

ds_context = Context(parametric_types=[Bernoulli,Bernoulli,Bernoulli,Bernoulli,
                                       Bernoulli,Bernoulli,Bernoulli,Bernoulli,
                                       Bernoulli,Bernoulli]).add_domains(train_data)

spn = learn_parametric(train_data, 
                       ds_context, 
                       min_instances_slice=20, 
                       min_features_slice=1, 
                       multivariate_leaf=True, 
                       leaves=create_cltree_leaf)

ll = log_likelihood(spn, train_data)
print(np.mean(ll))
```

### Cutset Networks (CNets)

With SPFlow we can learn both the structure and the parameters of CNets, a particular kind of SPNs with CLTs as leaf providing exact MPE inference, with:
```python
import numpy as np
np.random.seed(123)


from spn.structure.leaves.cltree.CLTree import create_cltree_leaf
from spn.structure.Base import Context
from spn.structure.leaves.parametric.Parametric import Bernoulli
from spn.algorithms.LearningWrappers import learn_parametric, learn_cnet
from spn.algorithms.Inference import log_likelihood

train_data = np.random.binomial(1, [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.1], size=(100,10))

ds_context = Context(parametric_types=[Bernoulli,Bernoulli,Bernoulli,Bernoulli,
                                       Bernoulli,Bernoulli,Bernoulli,Bernoulli,
                                       Bernoulli,Bernoulli]).add_domains(train_data)

# learning a CNet with a naive mle conditioning
cnet_naive_mle = learn_cnet(train_data, 
                            ds_context, 
                            cond="naive_mle", 
                            min_instances_slice=20, 
                            min_features_slice=1)

# learning a CNet with random conditioning
cnet_random = learn_cnet(train_data, 
                         ds_context, 
                         cond="random", 
                         min_instances_slice=20, 
                         min_features_slice=1)

ll = log_likelihood(cnet_naive_mle, train_data)
print("Naive mle conditioning", np.mean(ll))

ll = log_likelihood(cnet_random, train_data)
print("Random conditioning", np.mean(ll))

# computing exact MPE
from spn.algorithms.MPE import mpe
train_data_mpe = train_data.astype(float)
train_data_mpe[:,0] = np.nan
print(mpe(cnet_random, train_data_mpe)) 
```

### Expectations and Moments
SPNs allow you to compute first and higher order moments of the represented probability function by directly evaluating the tree structure. There are three main functions implemented for that.

The _Expectations_ function allows you to directly compute first oder moments given an SPN and (optionally) a list of features for which you need the expectation and an array of evidence.

```python
from spn.algorithms.stats.Expectations import Expectation
from spn.structure.leaves.piecewise.PiecewiseLinear import PiecewiseLinear

piecewise_spn = ((0.5 * PiecewiseLinear([0, 1, 2], [0, 1, 0], [], scope=[0]) +
                  0.5 * PiecewiseLinear([-2, -1, 0], [0, 1, 0], [], scope=[0])) *
                 (0.5 * PiecewiseLinear([0, 1, 2], [0, 1, 0], [], scope=[1]) +
                  0.5 * PiecewiseLinear([-1, 0, 1], [0, 1, 0], [], scope=[1])))
Expectation(piecewise_spn) # = [[0, 0.5]]
```

If you pass a feature scope, only the expectation for those features will be returned:
```python
from spn.algorithms.stats.Expectations import Expectation

piecewise_spn = ((0.5 * PiecewiseLinear([0, 1, 2], [0, 1, 0], [], scope=[0]) +
                  0.5 * PiecewiseLinear([-2, -1, 0], [0, 1, 0], [], scope=[0])) *
                 (0.5 * PiecewiseLinear([0, 1, 2], [0, 1, 0], [], scope=[1]) +
                  0.5 * PiecewiseLinear([-1, 0, 1], [0, 1, 0], [], scope=[1])))
Expectation(piecewise_spn, feature_scope=[0]) # = [[0]]
Expectation(piecewise_spn, feature_scope=[1]) # = [[0.5]]
```

Finally, you can also pass evidence to the network which computes the conditional expectation:
```python
from spn.algorithms.stats.Expectations import Expectation

piecewise_spn = ((0.5 * PiecewiseLinear([0, 1, 2], [0, 1, 0], [], scope=[0]) +
                  0.5 * PiecewiseLinear([-2, -1, 0], [0, 1, 0], [], scope=[0])) *
                 (0.5 * PiecewiseLinear([0, 1, 2], [0, 1, 0], [], scope=[1]) +
                  0.5 * PiecewiseLinear([-1, 0, 1], [0, 1, 0], [], scope=[1])))
Expectation(piecewise_spn, feature_scope=[0], evidence=np.array([[np.nan, 0]])) # = [[0]]
Expectation(piecewise_spn, feature_scope=[1], evidence=np.array([[0, np.nan]])) # = [[0.5]]

```
### Utilities

Finally, we have some basic utilities for working with SPNs:

We can make sure that the SPN that we are using is valid, that is, it is consistent and complete.
```python
from spn.algorithms.Validity import is_valid
print(is_valid(spn))
```
The output indicates that the SPN is valid and there are no debugging error messages:
```python
(True, None)
```

To compute basic statistics on the structure of the SPN:
```python
from spn.algorithms.Statistics import get_structure_stats
print(get_structure_stats(spn))
```

### Layerwise SPN in PyTorch
A layerwise implementation of leaf, sum and product nodes in PyTorch is available in the `spn.algorithms.layerwise` module. For more information, check out the [Layerwise SPN README](./src/spn/algorithms/layerwise/README.rst).

### Extending the library

Using the SPN is as we have seen, relatively easy. However, we might need to extend it if we want to work with new distributions.

Imagine, we wanted to create a new Leaf type that models the Pareto distribution.
We start by creating a new class:
```python
from spn.structure.leaves.parametric.Parametric import Leaf
class Pareto(Leaf):
    def __init__(self, a, scope=None):
        Leaf.__init__(self, scope=scope)
        self.a = a
```

Now, if we want to do inference with this new node type, we just implement the corresponding likelihood function:
```python
def pareto_likelihood(node, data=None, dtype=np.float64):
    probs = np.ones((data.shape[0], 1), dtype=dtype)
    from scipy.stats import pareto
    probs[:] = pareto.pdf(data[:, node.scope], node.a)
    return probs
```

This function receives the node, the data on which to compute the probability and the numpy dtype for the result.

Now, we just need to register this function so that it can be used seamlessly by the rest of the infrastructure:

```python
from spn.algorithms.Inference import add_node_likelihood
add_node_likelihood(Pareto, pareto_likelihood)
```

Now, we can create SPNs that use the new distribution and also evaluate them.

```python
spn = 0.3 * Pareto(2.0, scope=0) + 0.7 * Pareto(3.0, scope=0)
log_likelihood(spn, np.array([1.5]).reshape(-1, 1))
```

this produces the output:
```python
[[-0.52324814]]
```

All other aspects of the SPN library can be extended in a similar same way.

## Papers SPFlow can reproduce

* Nicola Di Mauro, Antonio Vergari, Teresa M.A. Basile, Floriana Esposito. "Fast and Accurate Density Estimation with Extremely Randomized Cutset Networks". In: ECML/PKDD, 2017.
* Nicola Di Mauro, Antonio Vergari, and Teresa M.A. Basile. "Learning Bayesian Random Cutset Forests". In ISMIS 2015, LNAI 9384, pp. 1-11, Springer, 2015.
* Nicola Di Mauro, Antonio Vergari, and Floriana Esposito. "Learning Accurate Cutset Networks by Exploiting Decomposability". In AI*IA. 2015, LNAI 9336, 1-12, Springer, 2015.
* Antonio Vergari, Nicola Di Mauro, and Floriana Esposito. "Simplifying, Regularizing and Strengthening Sum-Product Network Structure Learning". In ECML/PKDD, LNCS, 343-358, Springer. 2015.

## Papers implemented in SPFlow

* Molina, Alejandro, Sriraam Natarajan, and Kristian Kersting. "Poisson Sum-Product Networks: A Deep Architecture for Tractable Multivariate Poisson Distributions." In AAAI, pp. 2357-2363. 2017.

* Molina, Alejandro, Antonio Vergari, Nicola Di Mauro, Sriraam Natarajan, Floriana Esposito, and Kristian Kersting. "Mixed sum-product networks: A deep architecture for hybrid domains." In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI). 2018.

## Citation
If you find SPFlow useful please cite us in your work:
```
@misc{Molina2019SPFlow,
  Author = {Alejandro Molina and Antonio Vergari and Karl Stelzner and Robert Peharz and Pranav Subramani and Nicola Di Mauro and Pascal Poupart and Kristian Kersting},
  Title = {SPFlow: An Easy and Extensible Library for Deep Probabilistic Learning using Sum-Product Networks},
  Year = {2019},
  Eprint = {arXiv:1901.03704},
}
```

## Authors

* **Alejandro Molina** - *TU Darmstadt*
* **Antonio Vergari** - *Max-Planck-Institute*
* **Karl Stelzner** - *TU Darmstadt*
* **Robert Peharz** - *University of Cambridge*
* **Nicola Di Mauro** - *University of Bari Aldo Moro*
* **Kristian Kersting** - *TU Darmstadt*

See also the list of [contributors](https://github.com/alejandromolinaml/SPFlow/contributors) who participated in this project.

## Contributors

* **Moritz Kulessa** - *TU Darmstadt*
* **Claas Voelcker** - *TU Darmstadt*
* **Simon Roesler** - *Karlsruhe Institute of Technology*
* **Steven Lang** - *TU Darmstadt*
* **Alexander L. Hayes** - *Indiana University, Bloomington*

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE.md](LICENSE.md) file for details



## Acknowledgments
<img src="https://github.com/SPFlow/SPFlow/blob/master/Documentation/acknowledgements/bmbf.png" height="100"/><img src="https://github.com/SPFlow/SPFlow/blob/master/Documentation/acknowledgements/dfg.jpg"  height="100"/><img src="https://github.com/SPFlow/SPFlow/blob/master/Documentation/acknowledgements/euc.png"  height="100"/>
* Parts of SPFlow as well as its motivating research have been supported by the Germany Science Foundation (DFG) - AIPHES, GRK 1994, and CAML, KE 1686/3-1 as part of SPP 1999- and the Federal Ministry of Education and Research (BMBF) - InDaS, 01IS17063B.

* This project received funding from the European Union's Horizon 2020 research and innovation programme under the Marie Sklodowska-Curie Grant Agreement No. 797223 (HYBSPN).
