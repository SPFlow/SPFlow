"""
Created on Jun 21, 2018

@author: Moritz
"""

import time
import numpy as np
import matplotlib.pyplot as plt


from spn.experiments.AQP.Ranges import NumericRange

EPS = 0.00000001


def integral(m, b, x):
    return 0.5 * m * (x ** 2) + b * x


def inverse_integral(m, b, y):
    p = b / (0.5 * m)
    q = y / (0.5 * m)

    # Our scenario is easy we exactly know what value we want
    if m >= 0:
        # Left value is important
        return -(p / 2) + np.sqrt((p / 2) ** 2 + q)
    else:
        # Right value is important
        return -(p / 2) - np.sqrt((p / 2) ** 2 + q)


def cumulative(cumulative_stats, x):

    possible_bin_ids = np.where((cumulative_stats[:, 2] <= x) & (x <= cumulative_stats[:, 3]))
    bin_id = possible_bin_ids[0][0]
    stats = cumulative_stats[bin_id]

    return integral(stats[0], stats[1], x) - stats[4] + np.sum(cumulative_stats[:bin_id, 6])


def inverse_cumulative(cumulative_stats, y):

    bin_probs = cumulative_stats[:, 6]
    cumulative_probs = np.cumsum(bin_probs)

    # +EPS to avoid incorrect behavior caused by floating point errors
    cumulative_probs[-1] += EPS

    bin_id = np.where((y <= cumulative_probs))[0][0]
    stats = cumulative_stats[bin_id]

    lower_cumulative_prob = 0 if bin_id == 0 else cumulative_probs[bin_id - 1]
    y_perc = (y - lower_cumulative_prob) / bin_probs[bin_id]
    y_val = (stats[5] - stats[4]) * y_perc + stats[4]

    return inverse_integral(stats[0], stats[1], y_val)


def compute_cumulative_stats(x_range, y_range):

    # Compute representation of linear functions and cumulative probability
    cumulative_stats = []
    for i in range(len(x_range) - 1):
        y0 = y_range[i]
        y1 = y_range[i + 1]
        x0 = x_range[i]
        x1 = x_range[i + 1]

        m = (y0 - y1) / (x0 - x1)
        b = (x0 * y1 - x1 * y0) / (x0 - x1)

        lowest_y = integral(m, b, x0)
        highest_y = integral(m, b, x1)
        prob = highest_y - lowest_y

        cumulative_stats.append([m, b, x0, x1, lowest_y, highest_y, prob])

    return np.array(cumulative_stats, dtype=np.float64)


def random_sample(x_range, y_range, n_samples, rand_gen):

    # Modify x_range and y_range for specific ranges!!!

    cumulative_stats = compute_cumulative_stats(x_range, y_range)

    rand_probs = rand_gen.rand(n_samples)

    vals = [inverse_cumulative(cumulative_stats, prob) for prob in rand_probs]

    return vals

    # bin_probs = cumulative_stats[:,6]
    # rand_gen.choice(p=bin_probs)


def sample(x_range, y_range, ranges, n_samples, rand_gen):

    if ranges is None or ranges[0] is None:
        # Generate bins for random sampling because no range is specified
        bins_x = list(zip(x_range[:-1], x_range[1:]))
        bins_y = list(zip(y_range[:-1], y_range[1:]))
    else:
        # Generate bins for the specified range
        rang = ranges[0]
        assert isinstance(rang, NumericRange)

        bins_x = []
        bins_y = []

        # Iterate over the specified ranges
        intervals = rang.get_ranges()
        for interval in intervals:

            lower = interval[0]
            higher = interval[0] if len(interval) == 1 else interval[1]

            lower_prob = np.interp(lower, xp=x_range, fp=y_range)
            higher_prob = np.interp(higher, xp=x_range, fp=y_range)
            indicies = np.where((lower < x_range) & (x_range < higher))

            x_interval = [lower] + list(x_range[indicies]) + [higher]
            y_interval = [lower_prob] + list(y_range[indicies]) + [higher_prob]

            bins_x += list(zip(x_interval[:-1], x_interval[1:]))
            bins_y += list(zip(y_interval[:-1], y_interval[1:]))

    cumulative_stats = []
    for i in range(len(bins_x)):
        y0 = bins_y[i][0]
        y1 = bins_y[i][1]
        x0 = bins_x[i][0]
        x1 = bins_x[i][1]

        m = (y0 - y1) / (x0 - x1)
        b = (x0 * y1 - x1 * y0) / (x0 - x1)

        lowest_y = integral(m, b, x0)
        highest_y = integral(m, b, x1)
        prob = highest_y - lowest_y

        cumulative_stats.append([m, b, x0, x1, lowest_y, highest_y, prob])

    cumulative_stats = np.array(cumulative_stats, dtype=np.float64)

    cumulative_stats[:, 6] = cumulative_stats[:, 6] / np.sum(cumulative_stats[:, 6])

    rand_probs = rand_gen.rand(n_samples)

    vals = [inverse_cumulative(cumulative_stats, prob) for prob in rand_probs]

    return cumulative_stats, vals


def _rejection_sampling(x_range, y_range, n_samples, rand_gen):

    bins_x = list(zip(x_range[:-1], x_range[1:]))
    bins_y = list(zip(y_range[:-1], y_range[1:]))

    masses = []
    for i in range(len(bins_x)):
        if bins_x[i][0] == bins_x[i][1]:
            # Case that the range only contains one value .. Is that correct?
            assert bins_y[i][0] == bins_y[i][1]
            masses.append(bins_y[i][0])
        else:
            masses.append(np.trapz(bins_y[i], bins_x[i]))

    samples = []
    while len(samples) < n_samples:

        rand_bin = rand_gen.choice(len(masses), p=masses)
        #
        # generate random point uniformly in the box
        r_x = rand_gen.uniform(bins_x[rand_bin][0], bins_x[rand_bin][1])
        r_y = rand_gen.uniform(0, bins_y[rand_bin][1])
        #
        # is it in the trapezoid?
        trap_y = np.interp(r_x, xp=bins_x[rand_bin], fp=bins_y[rand_bin])
        if r_y < trap_y:
            samples.append(r_x)

    return np.array(samples)


def test_sample():
    # Create distribution
    y_range = [0.0, 10, 100, 30, 10, 200, 0.0]
    x_range = [0.0, 2, 4.0, 6, 8, 10, 12]
    x_range, y_range = np.array(x_range), np.array(y_range)
    x_range *= 1000
    auc = np.trapz(y_range, x_range)
    y_range = y_range / auc

    rand_gen = np.random.RandomState(10)

    t0 = time.time()
    samples = random_sample(x_range, y_range, 100000, rand_gen)
    exc_time = time.time() - t0

    print("cum_sampling: " + str(exc_time))

    t0 = time.time()
    samples = _rejection_sampling(x_range, y_range, 100000, rand_gen)
    exc_time = time.time() - t0

    print("rej_sampling: " + str(exc_time))

    # Plot distribution
    plt.title("Actual distribution")
    plt.plot(x_range, y_range)
    plt.show()

    plt.hist(samples, bins=50)
    plt.show()


def test_sample2():
    # Create distribution
    y_range = [0.0, 10, 100, 30, 10, 200, 0.0]
    x_range = [0.0, 2, 4.0, 6, 8, 10, 12]
    x_range, y_range = np.array(x_range), np.array(y_range)
    auc = np.trapz(y_range, x_range)
    y_range = y_range / auc

    rand_gen = np.random.RandomState(10)

    ranges = [NumericRange([[0.0, 4.0], [9.0, 12.0]])]

    t0 = time.time()
    cumulative_stats, samples = sample(x_range, y_range, ranges, 100000, rand_gen)
    exc_time = time.time() - t0

    print("cum_sampling: " + str(exc_time))

    # Plot distribution
    plt.title("Actual distribution")
    plt.plot(x_range, y_range)
    plt.show()

    plt.hist(samples, bins=50)
    plt.show()

    # Plot inverse cumulative distribution
    x_domain = np.linspace(0, 1, 100)
    y_domain = np.zeros(len(x_domain))
    for i, x_val in enumerate(x_domain):
        y_domain[i] = inverse_cumulative(cumulative_stats, x_val)

    plt.title("Inverse cumulative distribution")
    plt.plot(x_domain, y_domain)
    plt.show()


if __name__ == "__main__":

    test_sample2()
    exit()

    # Create distribution
    y_range = [0.0, 10, 100, 30, 10, 200, 0.0]
    x_range = [0.0, 2, 4.0, 6, 8, 10, 12]
    x_range, y_range = np.array(x_range), np.array(y_range)
    auc = np.trapz(y_range, x_range)
    y_range = y_range / auc

    cumulative_stats = compute_cumulative_stats(x_range, y_range)

    # Plot distribution
    plt.title("Actual distribution")
    plt.plot(x_range, y_range)
    plt.show()

    # Plot cumulative distribution
    x_domain = np.linspace(np.min(x_range), np.max(x_range), 100)
    y_domain = np.zeros(len(x_domain))
    for i, x_val in enumerate(x_domain):
        y_domain[i] = cumulative(cumulative_stats, x_val)

    plt.title("Cumulative distribution")
    plt.plot(x_domain, y_domain)
    plt.show()

    # Plot inverse cumulative distribution
    x_domain = np.linspace(0, 1, 100)
    y_domain = np.zeros(len(x_domain))
    for i, x_val in enumerate(x_domain):
        y_domain[i] = inverse_cumulative(cumulative_stats, x_val)

    plt.title("Inverse cumulative distribution")
    plt.plot(x_domain, y_domain)
    plt.show()
