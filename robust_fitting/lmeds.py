#! /usr/bin/env python2

import numpy as np
import sys # For exception handling

"""
General LMedS implementation.

"""

def niter_LMEDS(p, epsilon, s, Nmax=100000):
    """
    How many samples to compute with LMEDS.

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

    N = int(np.ceil(np.log(1-p) / np.log(1-(1-epsilon)**s)))
    if N > Nmax:
        return Nmax
    else:
        return N



def solve(A, b, m=-1, seed=None, min_iters=1, robust=False, recompute_model=True, do_print=False):
    """
    Solve least-squares problem using LMedS.

    Solves
        x = argmin_q median ||Aq - b||

    Parameters
    ----------
    A : array_like
    b : array_like
        Parameters of problem
    m : int, optional
        Model complexity. Default: A.shape[1]
    seed : int
        Optional seed for the random number generator
    min_iters : int
        Optional minimum amount of iterations.
    robust : bool, optional
        If set to True, use the absolute error instead of the squared error. Default = False.
    recompute_model : bool, optional
        Whether to recompute the model as a weighted least-squares problem. Default = True

    Returns
    -------
    model : array_like
        The best computed model
    inliers : (N,) array_like
        Inlier map.

    """

    if m == -1:
        s = A.shape[1]
    else:
        s = m
    p = 0.99
    epsilon = 0.5

    if seed is not None:
        np.random.seed(int(max(0,seed)))

    niters = niter_LMEDS(p, epsilon, s)
    niters = max(niters,min_iters)
    if do_print:
        print('Running for {} iterations.'.format(niters))

    N = A.shape[0]

    best_model = np.zeros(s)
    best_cost = 1e12
    best_residuals = np.zeros(N)

    for i in range(niters):
        sample_indices = np.random.choice(N, s, replace=False)
        A_sample = A[sample_indices,:]
        b_sample = b[sample_indices]

        try:
            model = np.linalg.lstsq(A_sample,b_sample)[0]
        except Exception as inst:
            print('Exception occured..')
            print(inst)
            continue

        if robust:
            cost_sq = np.abs(A.dot(model) - b)
        else:
            cost_sq = (A.dot(model) - b)**2

        med_cost_sq = np.median(cost_sq)

        if med_cost_sq < best_cost:
            if do_print:
                print('Updating.')
                print('\t New model: {}'.format(model))
                print('\t Median cost: {}'.format(med_cost_sq))
                print('\t Best model computed from data points:')
                print(np.c_[A_sample, b_sample])
            best_model = model.ravel()
            best_cost = med_cost_sq
            best_residuals = cost_sq.ravel()

    if best_cost == 1e12:
        return None, None

    if recompute_model:
        # Compute final least squares estimate, according to 
        # http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/node25.html
        #sigma_hat = 1.4826 * (1.0 + 5.0 / (2*N - s + 1)) * np.sqrt(best_cost)
        sigma_hat = 1.4826 * (1.0 + 5.0 / (N - s)) * np.sqrt(best_cost)
        print('Robust standard deviation: {}'.format(sigma_hat))
        #inliers = best_residuals < (2.5 * sigma_hat)**2
        inliers = best_residuals <= best_cost
        print('\t Number of inliers: {0} ({1:2.2f} %)'.format(inliers.sum(),inliers.sum() * 1.0/len(inliers) * 100.0))
        model = np.linalg.lstsq(A[inliers,:],b[inliers])[0]
    else:
        inliers = np.ones_like(best_residuals)
        model = best_model

    return model, inliers


def estimate(X, m, estimate_model, estimate_residuals, recompute_model=True, recompute_init=False, seed=None, min_iters=1, do_print=False):
    """
    General estimation function, using LMEDS.

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

    estimate_residuals : function
        Function to compute residuals. Should take an array_like object as
        parameter, as well as the model, and return an array of squared residuals.

            residuals = estimate_residuals(X, model)

    recompute_model : bool, optional
        Recompute the model using all inliers, computed by using the robust
        estimation for the standard deviation. In order to do so, the function
        estimate_model has to be able to handle more than m datapoints.
        Default: True
    recompute_init : bool, optional
        If set and recompute_model is True, the best estimate is supplied to
        estimate_model() via the ``init'' kwarg. That is, in the model
        recomputation step, 

            model_final = estimate_model(X[inliers,:], init=best_model)

        For this to work, estimate_model has to take an optional init kwarg.
        Default: False.
    seed : int, optional
        Initialize random number generator to a given state.
        Default: No seed.
    min_iters : int, optional
        Minimum number of iterations.
        Default: 1

    Returns
    -------
    model : object
        The computed model with most inliers.
    inliers : (N,) array_like
        Inlier map. If the results are not re-computed, this is all ones.

    """

    if seed is not None:
        np.random.seed(int(max(0,seed)))

    p = 0.99
    N = X.shape[0]

    niters = niter_LMEDS(p, 0.5, m)

    # Enforce a minimum number of iterations.
    niters = max(niters,min_iters)

    best_model = []
    best_cost = 1e12
    best_residuals = np.zeros(N)

    if do_print:
        print('Running LMEDS for a maximum of {} iterations.'.format(niters))

    for i in range(niters):
        sample_indices = np.random.choice(N, m)
        sample = X[sample_indices,:]

        try:
            model = estimate_model(sample)
        except Exception as inst:
            print('Exception occured..')
            print(inst)
            continue

        if model is None:
            continue

        cost_sq = estimate_residuals(X, model)
        med_cost_sq = np.median(cost_sq)

        if med_cost_sq < best_cost:
            if do_print:
                print('Iteration {}. Updating.'.format(i))
                print('\t New model: {}'.format(model))
                print('\t Median cost: {}'.format(med_cost_sq))
                print('\t Best model computed from data points:')
                print(sample)
            best_model = model
            best_cost = med_cost_sq
            best_residuals = cost_sq

    if recompute_model:
        # Compute final least squares estimate, according to 
        # http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/node25.html
        #sigma_hat = 1.4826 * (1.0 + 5.0 / (2*N - m + 1)) * np.sqrt(best_cost)
        #inliers = best_residuals < (2.5 * sigma_hat)**2

        # Just use best 50% of data points
        inliers = best_residuals <= best_cost
        if recompute_init:
            model = estimate_model(X[inliers,:],init=best_model)
        else:
            model = estimate_model(X[inliers,:])
    else:
        inliers = np.ones_like(best_residuals) > 0
        model = best_model

    return model, inliers






        


