import numpy as np
import torch
import torch.nn.functional as F

from distributions import RatNormal
from cspn import CSPN, CspnConfig, print_cspn_params

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
config.R = 3
config.D = 3
config.I = 5
config.S = 5
config.dropout = 0.0
config.leaf_base_class = RatNormal
config.leaf_base_kwargs = {'tanh_bounds': (0.0, 1.0)}
# config.leaf_base_kwargs = {'min_sigma': 0.1, 'max_sigma': 1.0, 'min_mean': 0.0, 'max_mean': 1.0}
model = CSPN(config)
print_cspn_params(model)

device = torch.device("cuda")
model = model.to(device)

batch_size = 16


def fun1():
    ent = model.vi_entropy_approx(
        condition=F.one_hot(torch.randint(0, 10, (batch_size,)), 10).float().to(device),
        sample_size=5
    ).mean()
    ent.backward()


exp = 0
nr = 1000
if exp == 0:
    print(timeit.timeit(fun1, number=nr))
elif exp == 1:
    for i in range(nr):
        fun1()
