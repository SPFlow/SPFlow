import numpy as np
import torch
import torch.nn.functional as F

from spn.experiments.RandomSPNs_layerwise.distributions import RatNormal
from spn.experiments.RandomSPNs_layerwise.cspn import CSPN, CspnConfig

from train_cspn_mnist_gen import print_cspn_params
import timeit

img_size = (1, 28, 28)  # 3 channels
cond_size = 10

config = CspnConfig()
config.F_cond = (cond_size,)
config.C = 1
config.nr_feat_layers = 2
config.fc_sum_param_layers = 2
config.fc_dist_param_layers = 2
config.F = int(np.prod(img_size))
config.R = 10
config.D = 7
config.I = 10
config.S = 10
config.dropout = 0.0
config.leaf_base_class = RatNormal
config.leaf_base_kwargs = {'min_sigma': 0.1, 'max_sigma': 1.0, 'min_mean': 0.0, 'max_mean': 1.0}
model = CSPN(config)
print_cspn_params(model)

device = torch.device("cuda")
model = model.to(device)

batch_size = 64


def fun1():
    leaf_gmm_ent_lb = model.iterative_gmm_entropy_lb(
        condition=F.one_hot(torch.randint(0, 10, (batch_size,)), 10).float().to(device),
        reduction='mean'
    )


def fun2():
    leaf_gmm_ent_lb = model.gmm_entropy_lb(
        condition=F.one_hot(torch.randint(0, 10, (batch_size,)), 10).float().to(device),
        reduction='mean'
    )
    leaf_gmm_ent_lb.backward()


def fun3():
    ent = model.entropy_taylor_approx(
        condition=F.one_hot(torch.randint(0, 10, (batch_size,)), 10).float().to(device),
        components=3
    )


def fun4():
    ent, _ = model.leaf_entropy_taylor_approx(
        condition=F.one_hot(torch.randint(0, 10, (batch_size,)), 10).float().to(device),
    )
    ent.backward()


exp = -1
nr = 100
if exp == -1:
    print(timeit.timeit(fun2, number=nr))
    print(timeit.timeit(fun4, number=nr))
elif exp == 0:
    print(timeit.timeit(fun1, number=nr))
    print(timeit.timeit(fun2, number=nr))
    print(timeit.timeit(fun3, number=nr))
    print(timeit.timeit(fun4, number=nr))
elif exp == 1:
    print(timeit.timeit(fun1, number=nr))
elif exp == 2:
    fun2()
elif exp == 3:
    fun3()
else:
    fun4()
