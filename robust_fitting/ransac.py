#! /usr/bin/env python2

import numpy as np

"""
General RANSAC implementation.

"""

def niter_RANSAC(p, epsilon, s, Nmax=100000):
    """
    How many samples to compute with RANSAC.

    Parameters
    ----------
    p : float
        Probability that at least 1 of the samples is free of outliers
    epsilon : float
        Proportion of outliers
    s : int
        Sample size (model complexity)
    Nmax : int, optional
        Upper bound on number of iterations. Default = 100000.
    """
    if Nmax < 1:
        return 100000
    if epsilon <= 0:
        return 1

    logarg = -np.exp(s * np.log(1.0 - epsilon))
    logval = np.log(1.0 + logarg)
    N = np.log(1.0 - p) / logval
    if logval < 0 and N < Nmax:
        return int(np.ceil(N))
    return Nmax


def estimate(X, m, estimate_model, estimate_inliers, return_cost=False, p_outlier=0.5, recompute_model=True, min_iters=1000, max_iters=10000, seed=None, do_print=False):
    """
    General estimation function, using RANSAC.

    Parameters
    ----------
    X : (N, D) array_like
        Array of datapoints. Each row corresponds to a datapoint, each
        column to a dimension.
    m : int
        Number of datapoints to be contained in a single sample.
    estimate_model : function
        Function to fit the model to a selection of datapoints. Should take
        a single array_like object as parameter, so it can be called as

            model = estimate_model(X[selection,:])

        The function estimate_model can also return None, in case no valid model
        can be fitted to the sample.

    estimate_inliers : function
        Function to select inliers. Should take an array_like object as
        parameter, as well as the model, and return a binary array of inliers.

            inliers = estimate_inliers(X, model)

        If return_cost is set, estimate_inliers returns an additional cost that
        is minimized (instead of minimizing the number of inliers). In this
        case,
            inliers, cost = estimate_inliers(X,model).

    return_cost : boolean, optional
        Whether estimate_inliers returns an additional cost to minimize.
        See above.
        Default: False
    p_outlier : float, optional
        The prior probability for outliers. Used to calculate the best number
        of iterations.
        Default: 0.5
    recompute_model : bool, optional
        Recompute the model using all inliers. In order to do so, the function
        estimate_model has to be able to handle more than m datapoints.
        Default: True
    min_iters : int, optional
        Minimum number of iterations. Used if the min number according to
        the given p_outlier is very small.
        Default: 1000
    max_iters : int, optional
        Maximum number of iterations.
        Default: 10000
    seed : int, optional
        Initialize random number generator to a given state.
        Default: No seed.

    Returns
    -------
    model : object
        The computed model with most inliers.
    inliers : (N,) array_like
        Inlier map.

    """

    if seed is not None:
        np.random.seed(int(max(0,seed)))

    p = 0.99
    N = X.shape[0]

    niters = niter_RANSAC(p, p_outlier, m)

    # Enforce a minimum number of iterations.
    niters = min(max(niters,min_iters),max_iters)

    best_inlier_map = np.zeros(N).astype('bool')
    best_model = np.zeros(m)
    best_cost = 1e12

    if do_print:
        print('[RANSAC] Running RANSAC for a maximum of {} iterations.'.format(niters))

    unsuccessful_attempts=0

    for i in range(niters):
        sample_indices = np.random.choice(N, m)
        sample = X[sample_indices,:]

        try:
            model = estimate_model(sample)
        except:
            unsuccessful_attempts += 1
            continue

        if model is None:
            unsuccessful_attempts += 1
            continue

        if return_cost:
            # Do cost-based minimization
            inlier_map,cost = estimate_inliers(X, model)

            # Check if we found a better inlier map
            if cost < best_cost and inlier_map.sum() > best_inlier_map.sum():
                best_inlier_map[:] = inlier_map
                best_model = model
                best_cost = cost
        else:
            inlier_map = estimate_inliers(X, model)

            # Check if we found a better inlier map
            if inlier_map.sum() > best_inlier_map.sum():
                best_inlier_map[:] = inlier_map
                best_model = model

    if do_print:
        print('[RANSAC] Done looping. {} of {} attempts were successful.'.format(niters-unsuccessful_attempts, niters))

    if all(best_inlier_map==0):
        return None,None

    if recompute_model:
        model_final = estimate_model(X[best_inlier_map,:])
    else:
        model_final = best_model

    return model_final, best_inlier_map



