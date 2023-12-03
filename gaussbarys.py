#!/usr/bin/env python3
'''
functions for approximating wasserstein barycenters for families of centered
gaussian measures
'''
import numpy as np
from numpy import trace
from numpy.linalg import inv
from scipy.linalg import sqrtm


def barycenter(refs: np.ndarray, coords: np.ndarray, steps: int = 16):
    '''
    assumes that:
    refs is a list of n x n PSD matrices
    coords is a point in the |refs|-simplex
    method is iterative, so steps can be as large we please
    with obvious time/accuracy trade-off. 16 was chosen somewhat
    arbitrarily after playing around and looking for a
    point after which the difference between approximants
    was < 10e-5
    '''
    iden = np.identity(refs[0].shape[0])
    old_approximant = barycentric_stepper(iden, refs, coords)
    norms = []
    for _ in range(steps):
        new_approximant = barycentric_stepper(old_approximant, refs, coords)
        norms.append(np.linalg.norm(new_approximant - old_approximant))
        old_approximant = new_approximant
    return new_approximant, norms


def barycentric_stepper(sn: np.ndarray, refs: np.ndarray, weights: np.ndarray):
    '''
    assumes that:
    sn is an n x n PSD matrix
    refs is a list of n x n PSD matrices
    weights is a point in the |refs|-simplex
    computes an approximation of the barycenter of refs wrt weights,
    given initial guess sn
    details of this computation are given in Alvarez-Esteban 2016, theorem 4.2
    the requirements for this theorem are that at least one of the
    matrices is positive definite
    '''
    dim = sn.shape[0]
    sroot = sqrtm(sn)
    invroot = inv(sroot)
    partial_sum = np.zeros((dim, dim))
    for reference, weight in zip(refs, weights):
        partial_sum += (weight * sqrtm(sroot@reference@sroot))
    return invroot@partial_sum@partial_sum@invroot


def transport_coefficients(si: np.ndarray, s0: np.ndarray):
    '''
    assumes that si, s0 are PSD n x n matrices
    compute the optimal transport map between two
    centered gaussian measures with covariance matrices
    si (for the sink) and s0 (for the source)
    '''
    refroot = sqrtm(s0)
    invroot = inv(refroot)
    prod = refroot@si@refroot
    return invroot@sqrtm(prod)@invroot


def werenski_coefficients(si: np.ndarray, sj: np.ndarray, s0: np.ndarray):
    '''
    assumes that si, sj, s0 are n x n PSD matrices
    computes the coefficients of the weresnski matrix as per
    Corollary 1 in Werenski et al 2022
    '''
    ci = transport_coefficients(si, s0)
    cj = transport_coefficients(sj, s0)
    iden = np.identity(si.shape[0])
    return trace((ci - iden)@(cj - iden)@s0)


def werenski_matrix(refs: np.ndarray, targ: np.ndarray):
    '''
    assumes that:
    refs is a list of n x n PSD matrices
    targ is an n x n PSD matrix
    computes the werenski matrix for targ w.r.t to refs.
    which is defined by A_{ij} = Tr((C_i - I)(C_j - I)S0)
    cf. werenski_coefficients
    '''
    dim = len(refs)
    out = np.zeros((dim, dim))
    for i, _ in enumerate(refs):
        for j, _ in enumerate(refs):
            out[i, j] = werenski_coefficients(refs[i], refs[j], targ)
    return out


def simplex_point(n: int):
    '''
    helper function to generate a random point in the n-simplex
    by default rand generates draws uniformly from (0,1)
    '''
    vec = np.random.rand(n, 1)
    return vec / np.linalg.norm(vec, 1)
