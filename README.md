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

### Prerequisites

The following modules must be installed:

```sh
$ pip3 install cppyy
$ pip3 install joblib
$ pip3 install matplotlib
$ pip3 install natsort
$ pip3 install networkx
$ pip3 install numba
$ pip3 install numpy
$ pip3 install pydot
$ pip3 install scipy
$ pip3 install sklearn
$ pip3 install statsmodels
$ pip3 install tqdm
$ pip3 install py-cpuinfo
$ pip3 install lark-parser
$ pip3 install tensorflow
```

### Installing

To install SPFlow, all you have to do is clone the repository.

```sh
mkdir SPFlow
git clone https://github.com/alejandromolinaml/SPFlow
```


## Authors

* **Alejandro Molina** - *TU Darmstadt*
* **Antonio Vergari** - *Max-Planck-Institute*
* **Karl Stelzner** - *TU Darmstadt*
* **Robert Peharz** - *University of Cambridge*
* **Kristian Kersting** - *TU Darmstadt*

See also the list of [contributors](https://github.com/alejandromolinaml/SPFlow/contributors) who participated in this project.

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Moritz Kulessa for the valuable code contributions
