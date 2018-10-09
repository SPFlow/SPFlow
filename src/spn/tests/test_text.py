'''
Created on March 22, 2018

@author: Alejandro Molina
'''

from spn.algorithms.Inference import likelihood
from spn.io.Text import str_to_spn, to_JSON, spn_to_str_equation
from spn.structure.StatisticalTypes import MetaType
from spn.structure.leaves.parametric.Inference import add_parametric_inference_support

from spn.structure.leaves.parametric.Parametric import *
from spn.structure.leaves.parametric.Text import add_parametric_text_support

if __name__ == '__main__':
    add_parametric_text_support()
    add_parametric_inference_support()


    cat = Categorical(p=[0.1, 0.2, 0.7])
    cat.scope.append(0)
    print(spn_to_str_equation(cat))
    catspn = str_to_spn(spn_to_str_equation(cat))
    print(spn_to_str_equation(catspn))

    original = Gaussian(mean=0, stdev=10)
    original.scope.append(0)
    s = spn_to_str_equation(original)
    print(s)
    recovered = str_to_spn(s)

    print(str_to_spn("Gaussian(V0|mean=1;stdev=10)"))

    gamma = Gamma(alpha=1, beta=2)
    gamma.scope.append(0)
    print(spn_to_str_equation(gamma))

    lnorm = LogNormal(mean=1, stdev=2)
    lnorm.scope.append(0)
    s2 = spn_to_str_equation(lnorm)
    print(s2)
    str_to_spn(s2)

    tm = TypeMixture(MetaType.REAL)
    tm.children = [gamma, lnorm]
    tm.weights = [0.3, 0.7]
    s3 = spn_to_str_equation(tm)
    print(s3)
    s4 = str_to_spn(s3)
    print(to_JSON(s4))

    root = Sum()
    root.children = [gamma, lnorm]
    root.weights = [0.2, 0.8]
    print(spn_to_str_equation(root))

    print(to_JSON(original))
    print(to_JSON(recovered))

    assert to_JSON(original) == to_JSON(recovered)

    data = np.asarray([1, 2]).reshape((-1, 1))

    llo = likelihood(original, data)
    print(llo)

    llr = likelihood(recovered, data)
    print(llr)

    assert np.all(np.isclose(llo, llr))
