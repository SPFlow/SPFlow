from timeit import default_timer

import torch
import torchvision
import torchvision.transforms as transforms

from joblib import Memory
from tqdm import tqdm

from spn.algorithms.Inference import log_likelihood
from spn.algorithms.LearningWrappers import learn_classifier, learn_parametric
from spn.experiments.layers.layers import to_layers, elapsed_timer, to_compressed_layers
from spn.experiments.layers.pytorch import get_torch_spn
from spn.structure.Base import Context, get_nodes_by_type
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian

import numpy as np

memory = Memory('/tmp/cache', verbose=0, compress=9)


@memory.cache
def get_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_train_data = torchvision.datasets.MNIST(root='/tmp/MNIST', train=True, download=True, transform=transform)
    mnist_test_data = torchvision.datasets.MNIST(root='/tmp/MNIST', train=False, download=True, transform=transform)

    def r(ds):
        rows = []
        labels = []
        for d, l in ds:
            rows.append(d.view(-1))
            labels.append(l)
        pixels = torch.stack(rows, dim=0)
        labels = torch.tensor(labels)
        return torch.cat([labels.unsqueeze(1).float(), pixels], dim=1)

    return r(mnist_train_data), r(mnist_test_data)


from spn.algorithms.splitting.Base import split_data_by_clusters


def get_split_cols_random_partition(rand_gen, fail=0.6):
    def split_cols_random_partitions(local_data, ds_context, scope):
        if rand_gen.random_sample() < fail:
            return [(local_data, scope, 1.0)]

        clusters = np.zeros_like(scope)

        for i, new_scope in enumerate(np.array_split(np.argsort(scope), 2)):
            clusters[new_scope] = i

        return split_data_by_clusters(local_data, clusters, scope, rows=False)

    return split_cols_random_partitions


@memory.cache
def learn_spn(data, min_inst):
    spn_classification = learn_classifier(data,
                                          Context(parametric_types=[Categorical] + [Gaussian] * (28 * 28)).add_domains(
                                              data),
                                          learn_parametric, 0,
                                          cols=get_split_cols_random_partition(np.random.RandomState(17)),
                                          rows="kmeans",
                                          min_instances_slice=min_inst)
    return spn_classification


@memory.cache
def to_torch(layers):
    return get_torch_spn(layers)


# @memory.cache
def spn_torch_cached(spn):
    layers = to_compressed_layers(spn)
    return get_torch_spn(layers)


#    return to_torch(layers)

def get_mnist_spn(min_inst):
    with elapsed_timer() as e:
        trainds, testds = get_data()

        print('loading data in', e())

    with elapsed_timer() as e:
        spn = learn_spn(trainds.numpy(), min_inst)
        print('learning spn classification', e())

    rng = np.random.RandomState(17)
    for n in get_nodes_by_type(spn, Gaussian):
        n.mean = rng.normal()

    device = 'cpu'
    with elapsed_timer() as e:
        torch_spn = spn_torch_cached(spn).to(device)
        print('to pytorch', e())

    return trainds, testds, spn, torch_spn


def pred_prob_per_class(spn, data):
    results = []

    #imgx = torch.cat([torch.zeros((data.shape[0], 1)), data], dim=1)
    #imgx[:, 0] = float('nan')
    #llx = spn(imgx)
    for i in range(10):
        img = torch.cat([torch.zeros((data.shape[0], 1)), data], dim=1)#torch.tensor(data)
        img[:, 0] = i
        ll = spn(img)
        results.append(ll)
    #res = torch.exp(torch.cat(results, dim=1) - llx)

    alpha = 1.0
    res = (torch.nn.functional.softmax(torch.cat(results, dim=1), dim=1) + alpha) / (1 + 10 * alpha)

    return res


if __name__ == '__main__':

    trainds, testds, spn, torch_spn = get_mnist_spn(200)
    traindsnp, testdsnp = trainds.numpy(), testds.numpy()
    device = 'cpu'

    avgll = 0
    pytorchtime = 0
    spftime = 0
    with elapsed_timer() as e:
        # with torch.no_grad():
        for img in tqdm(torch.split(testds, 10)):
            #pred_prob_per_class(spn, img)

            start = default_timer()
            tll = torch_spn(img).detach().numpy()
            pytorchtime += default_timer() - start

            start = default_timer()
            sll = log_likelihood(spn, img.numpy())
            spftime += default_timer() - start

            if not np.all(np.isclose(tll, sll)):
                print(tll - sll)
                # 0/0
            avgll += tll.sum().item()

        print('elapsed', e())
    print('pytorch time', pytorchtime)
    print('spflow time', spftime)

    print(avgll / testds.shape[0])
