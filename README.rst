=======================================================
A PyTorch Layer-wise Sum-Product Network Implementation
=======================================================

This module provides an efficient PyTorch layer-wise implementation of Sum-Product Networks.

Quick Start
-----------
The following is a short example on how to build the example layer-wise SPN:

.. image:: ./res/spn.png
    :alt: Example Sum-Product Network
    :align: center

The data matrices after each layer step look like this:

=========  =========  =========  =========
Scope {1}  Scope {2}  Scope {3}  Scope {4} 
=========  =========  =========  =========
x1         x2         x3         x4        
=========  =========  =========  =========

**Gaussian Layer (multiplicity=2):**

=========  =========  =========  =========
Scope {1}  Scope {2}  Scope {3}  Scope {4} 
=========  =========  =========  =========
x11        x21        x31        x41       
x12        x22        x32        x42       
=========  =========  =========  =========

**Product Layer (cardinality=2):**

===========  ===========
Scope {1,2}  Scope {3,4} 
===========  ===========
x11 * x21    x31 * x41   
x12 * x22    x32 * x42   
===========  ===========

**Sum Layer:**

=================================  =================================
Scope {1,2}                        Scope {3,4}                       
=================================  =================================
x11 * x21 * w11 + x12 * x22 * w12  x31 * x41 * w21 + x32 * x42 * w22 
=================================  =================================


**Product Layer (cardinality=2):**

=========================================================================  ==
**Scope {1,2,3,4}**                                                                                                                        
=========================================================================  == 
(x11 * x21 * w11 + x12 * x22 * w12) * (x31 * x41 * w21 + x32 * x42 * w22)   
=========================================================================  ==


This architecture can be implemented with the following code

.. code:: python


  from spn.algorithms.layerwise.distributions import Normal
  from spn.algorithms.layerwise.layers import Sum, Product
  import torch
  from torch import nn

  # Set seed
  torch.manual_seed(0)

  # 3 Samples, 4 features
  x = torch.rand(3, 4)

  # Create SPN layers
  gauss = Normal(multiplicity=2, in_features=4)
  prod1 = Product(in_features=4, cardinality=2)
  sum1 = Sum(in_features=2, in_channels=2, out_channels=1)
  prod2 = Product(in_features=2, cardinality=2)

  # Stack SPN layers
  spn = nn.Sequential(gauss, prod1, sum1, prod2)

  logprob = spn(x)
  print(logprob)
  # tensor([[-31.7358],                                                                                                            
        # [-22.1905],                                                                                                            
        # [-51.8741]], grad_fn=<SumBackward2>)

API
===

- Input distributions can be found in `distributions.py <./src/spn/algorithms/layerwise/distributions.py>`_.
- Sum and Product layers can be found in `layers.py <./src/spn/algorithms/layerwise/layers.py>`_.
- A RatSpn implementation based on the layer-wise Sum and Product nodes can be found in `rat_spn.py <./src/spn/experiments/RandomSPNs_layerwise/rat_spn.py>`_.


Minimal Working Example
_______________________

The following is a minimal working exam to use the layer-wise SPN
implementation: https://gist.github.com/steven-lang/ceb899a64630cb1473e84986b0bfb3b5



Layer-wise Principle
====================

The idea of the layer-wise implementation is that each layer can be represented by a single tensor operation that acts on a certain axis of the tensor.

Lets say we start with a data matrix of shape :code:`N x D` where :code:`N` is the batch size and :code:`D` is the number of features in the dataset:

Current dimensions: :code:`N x D`

.. code:: python

  #         D
  #     ________  
  #    |        |
  # N  |        |
  #    |        |
  #    |________|

The Leaf layer will start with transforming this data matrix into a data cube, where the third axis is the number of leaf nodes per input feature (= channels, :code:`C`). This means, for each input variable we now have multiple representations by different distributions.

Current dimensions: :code:`N x D x C`

.. code:: python

  #            D
  #       __________
  #      /         /|
  # C   /         / |
  #    /_________/  |
  #    |        |   |
  # N  |        |  /
  #    |        | /
  #    |________|/
  
Following the Leaf layer, we can now either apply a Product or a Sum layer. 

The Product layer represents an operation along the feature axis. E.g. a Product layer with :code:`cardinality=2`, which means each internal product node consists of exactly two children, would transform the shape from :code:`N x D x C` to :code:`N x D/2 x C`:

.. code:: python

  #            D                                      D/2
  #       __________                                _____
  #      /         /|                              /    /|
  # C   /         / |                         C   /    / |
  #    /_________/  |   -- Product with  ->      /____/  |
  #    |        |   |   -- cardinality=2 ->      |   |   |
  # N  |        |  /                          N  |   |  /
  #    |        | /                              |   | /
  #    |________|/                               |___|/

Equally, a Sum layer transforms the tensor along the third axis, affecting the number of channels. A Sum layer with :code:`out_channels=K` will have :code:`K` repeated Sum nodes for each scope in the previous layer. The shape will then be transformed as :code:`N x D x C` to :code:`N x D x K` like this:


.. code:: python

  #            D                                                      
  #       __________                                          D
  #      /         /|                                    _________    
  # C   /         / |                              K   /         /|   
  #    /_________/  |    -- Sum with       ->         /_________/ |   
  #    |        |   |    -- out_channels=2 ->         |        |  |   
  # N  |        |  /                               N  |        |  |
  #    |        | /                                   |        | /
  #    |________|/                                    |________|/      

It is important to remember the meaning of each axis:

- *Axis 1*: Batch axis, not relevant to any operation.
- *Axis 2*: Features / Input Variables / Scopes. Values along this axis all come from different input variables and have therefore different scopes. Hence, we apply the Product layer over the second axis.
- *Axis 3*: Channel / Representations. Values along this axis are all in the same scope. Therefore, we apply the Sum layer over the third axis.


Benchmark
_________

.. image:: ./res/benchmark.png
    :alt: Benchmark
    :align: center

The example architecture above has been used to benchmark the runtime with varying number of input features (batch size = 1024) and varying batch size (number of input features = 1024).

The comparison is against a node-wise implementation of SPNs in `SPFlow <https://github.com/SPFlow/SPFlow>`_ on the CPU and a node-wise implementation of SPNs in SPFlow on the GPU using Tensorflow.

Issues
======
