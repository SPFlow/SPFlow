'''
Created on March 20, 2018

@author: Alejandro Molina
'''


def compute_histogram_type_wise(data, feature_type, domain, alpha=0.0):
    repr_points = None

    if feature_type == 'continuous':
        maxx = numpy.max(domain)
        minx = numpy.min(domain)

        prior_density = 1 / (maxx - minx)

        if numpy.var(data) > 1e-10:
            breaks, densities, mids = getHistogramVals(data)
        else:
            breaks = numpy.array([minx, maxx])
            densities = numpy.array([prior_density])
            mids = numpy.array([minx + (maxx - minx) / 2])

        repr_points = mids

    elif feature_type in {'discrete', 'categorical'}:
        prior_density = 1 / len(domain)

        #
        # augmenting one fake bin left
        breaks = numpy.array([d for d in domain] + [domain[-1] + 1])
        # print('categorical binning', breaks)
        # if numpy.var(data_slice.getData()) == 0.0:
        densities, breaks = compute_histogram(data, bins=breaks, density=True)

        repr_points = domain
        #
        # laplacian smoothing?
        if alpha:
            n_samples = data.shape[0]
            n_bins = len(breaks) - 1
            counts = densities * n_samples
            densities = (counts + alpha) / (n_samples + n_bins * alpha)

    assert len(densities) == len(repr_points)
    assert len(densities) == len(breaks) - 1

    return densities, breaks, prior_density, repr_points
