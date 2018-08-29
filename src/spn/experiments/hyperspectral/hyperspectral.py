import numpy as np
import scipy.io as sio
import sklearn as sk
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

from spn.algorithms.Inference import log_likelihood
from spn.algorithms.LearningWrappers import learn_parametric
from spn.experiments.hyperspectral.DataManager import *
from spn.io.Text import spn_to_str_equation
from spn.structure.Base import Context, Sum, assign_ids
from spn.structure.StatisticalTypes import MetaType
from spn.structure.leaves.parametric.Inference import add_parametric_inference_support, Gaussian
from spn.structure.leaves.parametric.Parametric import Categorical
from spn.structure.leaves.parametric.Text import add_parametric_text_support
from spn.algorithms.Statistics import *


def train_spn(window_size=3, min_instances_slice=10000, features=None, number_of_classes=3):
    if features is None:
        features = [20, 120]

    add_parametric_inference_support()
    add_parametric_text_support()

    data = get_data_in_window(window_size=window_size, features=features, three_classes=number_of_classes==3)

    sss = sk.model_selection.StratifiedShuffleSplit(test_size=0.2, train_size=0.8, random_state=42)
    for train_index, test_index in sss.split(data[:, 0:window_size * window_size * len(features)],
                                             data[:, (window_size * window_size * len(features)) + (
                                                     int(window_size * window_size / 2))]):
        X_train, X_test = data[train_index], data[test_index]

    context_list = list()
    parametric_list = list()
    number_of_features = len(features)
    for _ in range(number_of_features * window_size * window_size):
        context_list.append(MetaType.REAL)
        parametric_list.append(Gaussian)

    for _ in range(window_size * window_size):
        context_list.append(MetaType.DISCRETE)
        parametric_list.append(Categorical)

    ds_context = Context(meta_types=context_list)
    ds_context.add_domains(data)
    ds_context.parametric_types = parametric_list

    spn = load_spn(window_size, features, min_instances_slice, number_of_classes)
    if spn is None:
        spn = Sum()
        for class_pixel in tqdm(range(-window_size * window_size, 0)):
            for label, count in zip(*np.unique(data[:, class_pixel], return_counts=True)):
                train_data = X_train[X_train[:, class_pixel] == label, :]
                branch = learn_parametric(train_data, ds_context, min_instances_slice=min_instances_slice)
                spn.children.append(branch)
                spn.weights.append(train_data.shape[0])

        spn.scope.extend(branch.scope)
        spn.weights = (np.array(spn.weights) / sum(spn.weights)).tolist()

        assign_ids(spn)
        save_spn(spn, window_size, features, min_instances_slice, number_of_classes)

    res = np.ndarray((X_test.shape[0], number_of_classes))

    for i in tqdm(range(number_of_classes)):
        tmp = X_test.copy()
        tmp[:, -int((window_size ** 2) / 2)] = i
        res[:, i] = log_likelihood(spn, tmp)[:, 0]

    predicted_classes = np.argmax(res, axis=1).reshape((X_test.shape[0], 1))

    correct_predicted = 0
    for x, y in zip(X_test[:, -5], predicted_classes):
        if x == y[0]:
            correct_predicted += 1
    accuracy = correct_predicted / X_test.shape[0]
    return spn, accuracy

    # print(spn)
    # print(spn_to_str_equation(spn))
    # print(log_likelihood(spn, data))


def predict_img(spn, values=None, window_size=3, number_of_classes=3):
    if values is None:
        values = [20, 140]

    plt.subplot(1, 2, 1)
    X, Y = read_img()
    plt.title("Truth")
    plt.imshow(Y)

    def predict(spn):
        data = get_data_in_window(features=values)
        res = np.ndarray((Y.shape[0] * Y.shape[1], number_of_classes))
        for i in range(number_of_classes):
            tmp = data.copy()
            tmp[:, -int((window_size ** 2) / 2)] = i
            res[:, i] = log_likelihood(spn, tmp)[:, 0]
        return np.argmax(res, axis=1).reshape((Y.shape[0], Y.shape[1]))

    plt.subplot(1, 2, 2)
    plt.title("Predicted")
    a = predict(spn)
    plt.imshow(a)

    plt.savefig("comparison.png")
    plt.show()


def plot_experiments():
    # # Plot different window size
    # acc = list()
    # x = list()
    # for window_size in range(3,7,2):
    #     x.append(window_size)
    #     tmp = list()
    #     for min_instances_slice in range (1,4):
    #         _, accuracy = train_spn(window_size, 10**(min_instances_slice+1))
    #         tmp.append(accuracy)
    #     acc.append(tmp)
    # plt.plot(x, acc, "x-")
    # plt.title("Experiment with Window Size")
    # plt.ylabel("Accuracy")
    # plt.xlabel("Window Size")
    # plt.savefig("exp_window_size.png")
    # plt.show()

    # Plot different window size
    # acc = list()
    # x = list()
    # for window_size in range(3, 13, 2):
    #     x.append(window_size)
    #     _, accuracy = train_spn(window_size, 10000)
    #     acc.append(accuracy)
    # plt.plot(x, acc, "x-")
    # plt.title("Experiment with Window Size")
    # plt.ylabel("Accuracy")
    # plt.xlabel("Window Size")
    # plt.savefig("exp_window_size.png")
    # plt.show()

    # # Plot with different features
    # acc = list()
    # x = list()
    # for i in range(10, 90, 10):
    #     x.append(len(list(range(i, 90, 10))))
    #     _, accuracy = train_spn(window_size=3, min_instances_slice=1000,features=list(range(i, 100, 10)))
    #     acc.append(accuracy)
    # plt.plot(x, acc, "x-")
    # plt.title("Experiment with Window Size")
    # plt.ylabel("Accuracy")
    # plt.xlabel("Window Size")
    # plt.savefig("exp_window_size.png")
    # plt.show()

    # # Plot different feature number
    # acc = list()
    # x = list()
    # for i in range(14):
    #     x.append(10 * (i + 1))
    #     _, accuracy = train_spn(300, list(range(20, 30 + (i * 10))))
    #     acc.append(accuracy)
    # plt.plot(x, acc, "x-")
    # plt.title("Experiment with Feature Size")
    # plt.ylabel("Accuracy")
    # plt.xlabel("Number of Features")
    # plt.savefig("exp_number_of_features.png")
    # plt.show()

    feature_list = list(range(0, 160, 1))
    spn, acc = train_spn(3, 2000, feature_list)
    print(get_structure_stats(spn))
    print("Accuracy on spn: {}".format(acc))
    # predict_img(spn, feature_list)


def find_good_features():
    res = list()
    for i in range(0, 160):
        _, acc = train_spn(window_size=3, min_instances_slice=10000, features=[i])
        res.append([i, acc])
    return res


if __name__ == '__main__':
    # _, acc = train_spn()
    # print("Accuracy: {}".format(acc))
    plot_experiments()
    # res = find_good_features()
    # pass
