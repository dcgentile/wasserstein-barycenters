#!/usr/bin/env python3
'''
utility functions for geodesic convexity testing

'''

import numpy as np
from gaussbarys import barycenter, werenski_matrix
from gaussian_utilities import opt_lam_grad_norm
from scipy.linalg import sqrtm

def generate_refs(p: int, dim: int):
    '''
    generate p positive definite matrices of size d x d
    '''
    refs = []
    for _ in range(p):                      # generate some covariance matrices
        x = np.random.rand(dim, dim)
        refs.append((0.5 * (x + x.T)) + dim * np.identity(dim))
    return refs


def simplex_point(n: int):
    '''
    helper function to generate a random point in the n-simplex
    by default rand generates draws uniformly from (0,1)
    '''
    vec = np.random.rand(n, 1)
    return vec / np.linalg.norm(vec, 1)


def generate_sample(p: int, dim: int, method=None):
    '''
    currently unused
    n is the number of reference measures to generate
    dim is the size of the matrices
    returns the l2 norm of the difference between the
    point on the geodesic curve between two computed barycenters
    and the barycenter obtained with the approximate barycentric
    coordinates of that point on the geodesic
    '''
    refs = generate_refs(p, dim)
    simp_points = (simplex_point(p), simplex_point(p))
    t = simplex_point(2)
    n0, _ = barycenter(refs, simp_points[0])
    n1, _ = barycenter(refs, simp_points[1])
    nt, _ = barycenter([n0, n1], t)
    # n0 and n1 are barycenters of the reference measures
    # nt is the point on the geodesic from n0 to n1 at time t
    processed_refs = [sqrtm(sqrtm(nt)@ref@sqrtm(nt)) for ref in refs]
    if method == 'grad':
        approx_lambda = opt_lam_grad_norm(processed_refs, nt)
    else:
        a = werenski_matrix(refs, nt)
        evals, evecs = np.linalg.eig(a)
        i = np.argmin(evals)
        approx_lambda = np.abs(evecs[:, i] / np.linalg.norm(evecs[:, i], 1))
    b, _ = barycenter(refs, approx_lambda)
    return np.linalg.norm(nt - b)


def werenski_test(p: int, dim: int):
    '''
    a priori, if we compute an explicit barycenter and
    solve for the barycentric coordinates
    with the werenski criterion, we should
    necessarily see a 0 eigenvalue at the end of this
    and the corresponding eigenvector should recover the
    barycentric coordinates
    '''
    measures = []
    for _ in range(p):
        x = np.random.rand(dim, dim)
        measures.append((0.5 * (x + x.T)) + dim * np.identity(dim))
    l0 = simplex_point(p)
    n0, _ = barycenter(measures, l0)
    processed_refs = [sqrtm(sqrtm(n0)@measure@sqrtm(n0)) for measure in measures]
    approx_lambda = opt_lam_grad_norm(processed_refs, n0)
    n1, _ = barycenter(measures, approx_lambda)
    sample = np.linalg.norm(n1 - n0)
    print(f"Frobenius norm discrepancy between geodesic point and its geodesic approximation: {sample}")


def eigen_method(p: int, dim: int):
    '''
    instead of solving the convex optimization problem,
    we instead search for eigenvectors of the W matrix
    with eigenvalue 0
    '''
    pd = lambda x: (0.5 * (x + x.T)) + dim * np.identity(dim)
    measures = [pd(np.random.rand(dim,dim)) for _ in range(p)]
    l0 = simplex_point(p)
    n0, _ = barycenter(measures, l0)
    v = opt_lam_eigen(measures, n0)
    l2diff = np.linalg.norm(l0 - v)
    return l0, v, l2diff


def opt_lam_eigen(refs, target):
    '''
    given a family of reference measures and a target measure,
    presumed to be a barycenter of the references
    retrieve the barycentric coordinates of target
    wrt refs
    '''
    a = werenski_matrix(refs, target)
    evals, evecs = np.linalg.eig(a)
    i = np.argmin(evals)
    v = evecs[:,i]
    return v / np.linalg.norm(v, 1)
