"""
Created on September 05, 2018

@author: Alejandro Molina
"""
from sklearn import manifold, random_projection, decomposition

from numpy.random.mtrand import RandomState
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from MulticoreTSNE import MulticoreTSNE as TSNE


from spn.algorithms.splitting.RDC import get_split_rows_RDC_py, rdc_transformer
from spn.algorithms.splitting.Random import get_split_rows_random_partition
from spn.data.datasets import get_mnist
import numpy as np

from spn.experiments.conditional.img_tools import standardize, stitch_imgs, save_img
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType


def one_hot(y):
    if len(y.shape) != 1:
        return y
    values = np.array(sorted(list(set(y))))
    return np.array([values == v for v in y], dtype=int)


if __name__ == "__main__":
    images_tr, labels_tr, images_te, labels_te = get_mnist()
    print("mnist loaded")
    images = np.reshape(images_tr, (-1, 28, 28))
    images = standardize(images)

    blocks = 45
    n_instances_per_class = blocks * blocks  # 2025

    print(np.unique(labels_tr, return_counts=True))
    num3 = images[labels_tr == 3][:n_instances_per_class].reshape(n_instances_per_class, -1)
    num5 = images[labels_tr == 5][:n_instances_per_class].reshape(n_instances_per_class, -1)

    num3_img = stitch_imgs(
        imgs=1,
        img_size=(28 * blocks, 28 * blocks),
        num_blocks=(blocks, blocks),
        blocks={i: num3[i, :].reshape(28, 28) for i in range(n_instances_per_class)},
    )

    save_img(num3_img[0], "/Users/alejomc/PycharmProjects/SimpleSPN/src/spn/experiments/conditional/num3.png")

    num5_img = stitch_imgs(
        imgs=1,
        img_size=(28 * blocks, 28 * blocks),
        num_blocks=(blocks, blocks),
        blocks={i: num5[i, :].reshape(28, 28) for i in range(n_instances_per_class)},
    )

    save_img(num5_img[0], "/Users/alejomc/PycharmProjects/SimpleSPN/src/spn/experiments/conditional/num5.png")

    both = np.concatenate((num3, num5), axis=0)

    labels = np.concatenate((np.zeros(n_instances_per_class), np.ones(n_instances_per_class)), axis=0).reshape(-1, 1)

    print(num5.shape, both.shape, labels.shape)

    shf = np.arange(labels.shape[0])
    RandomState(17).shuffle(shf)

    labels = labels[shf]
    both = both[shf, :]

    def print_results(name, labels, predictions):
        print("%s%s%s" % ("-" * 10, name, "-" * 10))

        print("acc", accuracy_score(labels, predictions))
        print("acc", 1 - accuracy_score(labels, predictions))
        print("confusion km\n", confusion_matrix(labels, predictions))

        print("\n" * 2)

    data = np.concatenate((both, labels), axis=1)
    # data = np.concatenate((both, one_hot(labels)), axis=1)
    labels = labels.reshape(-1)

    def compute_kmeans(data):
        return KMeans(n_clusters=2, n_init=20, max_iter=1000, random_state=17).fit_predict(data)

    def compute_kmeans_rdc(data):
        ds_context = Context(meta_types=[MetaType.REAL] * data.shape[1])
        ds_context.add_domains(data)
        scope = list(range(data.shape[1]))
        meta_types = ds_context.get_meta_types_by_scope(scope)
        domains = ds_context.get_domains_by_scope(scope)

        rdc_data = rdc_transformer(
            data, meta_types, domains, k=10, s=1 / 6, non_linearity=np.sin, return_matrix=True, rand_gen=RandomState(17)
        )

        return KMeans(n_clusters=2, random_state=RandomState(17)).fit_predict(rdc_data)

    print_results("kmeans", labels, compute_kmeans(data))
    # print_results("kmeans+rdc", labels, compute_kmeans_rdc(data))

    tsne_embedding_data = TSNE(n_components=3, verbose=10, n_jobs=4, random_state=17).fit_transform(data)
    print_results("tsne fast kmeans", labels, compute_kmeans(tsne_embedding_data))

    tree_embedding_data = RandomTreesEmbedding(n_estimators=200, random_state=0, max_depth=5).fit_transform(data)
    print_results("tree kmeans", labels, compute_kmeans(tree_embedding_data))

    0 / 0
    srp_emb_data = random_projection.SparseRandomProjection(n_components=20, random_state=42).fit_transform(data)
    print_results("SparseRandomProjection kmeans", labels, compute_kmeans(srp_emb_data))

    iso_emb_data = manifold.Isomap(30, n_components=2).fit_transform(data)
    print_results("iso kmeans", labels, compute_kmeans(iso_emb_data))

    # lle_emb_data = manifold.LocallyLinearEmbedding(10, n_components=2, method='ltsa').fit_transform(data)
    # print_results("lle kmeans", labels, compute_kmeans(lle_emb_data))

    svd = decomposition.TruncatedSVD(n_components=2).fit_transform(data)
    print_results("svd kmeans", labels, compute_kmeans(svd))

    tsne_emb_data = manifold.TSNE(n_components=3, init="pca", random_state=17).fit_transform(data)
    print_results("tsne kmeans", labels, compute_kmeans(tsne_emb_data))

    0 / 0

    # print_results("tree kmeans+rdc", labels, compute_kmeans_rdc(tree_embedding_data))

    mds_embedding_data = manifold.MDS(n_components=10, n_init=1, max_iter=100)
    print_results("tree kmeans", labels, compute_kmeans(mds_embedding_data))
    print_results("tree kmeans+rdc", labels, compute_kmeans_rdc(mds_embedding_data))

    spectral_embedding_data = manifold.SpectralEmbedding(
        n_components=10, random_state=17, eigen_solver="arpack"
    ).fit_transform(data)

    print_results("spectral10 kmeans", labels, compute_kmeans(spectral_embedding_data))
    print_results("spectral10 kmeans+rdc", labels, compute_kmeans_rdc(spectral_embedding_data))

    spectral_embedding_data = manifold.SpectralEmbedding(
        n_components=20, random_state=17, eigen_solver="arpack"
    ).fit_transform(data)

    print_results("spectral20 kmeans", labels, compute_kmeans(spectral_embedding_data))
    print_results("spectral20 kmeans+rdc", labels, compute_kmeans_rdc(spectral_embedding_data))

    spectral_embedding_data = manifold.SpectralEmbedding(
        n_components=5, random_state=17, eigen_solver="arpack"
    ).fit_transform(data)

    print_results("spectral5 kmeans", labels, compute_kmeans(spectral_embedding_data))
    print_results("spectral5 kmeans+rdc", labels, compute_kmeans_rdc(spectral_embedding_data))
