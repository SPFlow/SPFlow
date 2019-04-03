"""
Created on March 28, 2019
@author: Hari Teja Tatavarti

"""
import numpy as np

# following functions can be used for manipulating dataset for use with learn_spmn

def align_data(data_frame, partial_order):
    """
    :param data_frame: pandas data frame with column names
    :param partial_order: partial order
    :return: pandas data frame aligned w.r.t partial order
    """
    partial_order = partial_order
    columns_titles = [var for var_set in partial_order for var in var_set]
    # print("col_titles", columns_titles)
    df = data_frame.reindex(columns=columns_titles)
    # print('df',df)
    return df, columns_titles


def cooper_tranformation(train_data, col_ind):
    """
    :param train_data: numpy array
    :param col_ind: index of utility node in train_data
    :return: utility variable converted to binary random variable
    """
    train_data = convert_utility_to_probability(train_data, col_ind)
    bin_data = np.repeat(train_data, [10], axis=0)
    # print('repeated_data', bin_data[0:20])
    for ins in bin_data:
        rand = np.random.uniform(0, 1)
        if (((rand < ins[col_ind]) or (ins[col_ind] == 1)) and (ins[col_ind] != 0)):
            ins[col_ind] = 1
        elif ((rand >= ins[col_ind]) or (ins[col_ind] == 0)):
            ins[col_ind] = 0
    # print('bin_data', bin_data[0:20])
    return bin_data


def convert_utility_to_probability(train_dataa, col_ind):
    """
    computes prob = (val-min)/(max-m in); cooper transformation
    :param train_data:
    :param col_ind:
    :return: train_data with values of column changed into probabilities
    """
    train_data = train_dataa.copy()
    cost_vals = np.unique(train_data[:, col_ind])
    # print('cost_vals', cost_vals)
    range = np.ptp(cost_vals)  # (max-min)
    # print('range', range)
    min = np.amin(cost_vals)
    x = np.subtract(cost_vals, min)  # (val-min)
    prob = np.true_divide(x, range)

    for i in np.arange(cost_vals.size):
        train_data[train_data == cost_vals[i]] = prob[i]

    return train_data
