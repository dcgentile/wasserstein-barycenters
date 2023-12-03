#!/usr/bin/env python3
'''
let m_i be a family of gaussians, let n_0, n_1 be wasserstein barycenters
of the meaures m_i, let n_t be any point on the Wasserstein geodesic
between n_0 and n_1. this script generates an arbitrary family m_i
and tests the discrepancy between n_t and the barycenter obtained
by approximating the barycentric coordinates of n_t wrt to the family m_i
if Wasserstein barycenters were geodesically closed, we would (should) see
0 discrepancy between the two.
'''
import argparse
import numpy as np
from gaussbarys import simplex_point, barycenter, werenski_matrix


parser = argparse.ArgumentParser()
parser.add_argument("ref_num", type=int)
parser.add_argument("dim_num", type=int)

args = parser.parse_args()


def main(refs: int, dim: int):
    '''
    main routine
    n is the number of references,
    d is the dimension of the matrices
    '''
    werenski_test(refs, dim)
    sample = generate_sample(refs, dim)
    print(f"Frobenius norm discrepancy between geodesic point\
    and barycentric approximation: {sample}")


def generate_sample(n: int, dim: int):
    '''
    currently unused
    n is the number of reference measures to generate
    dim is the size of the matrices
    returns the l2 norm of the difference between the
    point on the geodesic curve between two computed barycenters
    and the barycenter obtained with the approximate barycentric
    coordinates of that point on the geodesic
    '''
    refs = []
    for _ in range(n):                      # generate some covariance matrices
        x = np.random.rand(dim, dim)
        refs.append((0.5 * (x + x.T)) + dim * np.identity(dim))
    simp_points = (simplex_point(n), simplex_point(n))
    t = simplex_point(2)
    n0, _ = barycenter(refs, simp_points[0])
    n1, _ = barycenter(refs, simp_points[1])
    nt, _ = barycenter([n0, n1], t)
    # n0 and n1 are barycenters of the reference measures
    # nt is the point on the geodesic from n0 to n1 at time t

    a = werenski_matrix(refs, nt)
    # form the werenski matrix A for nt wrt to the reference measures
    evals, evecs = np.linalg.eig(a)     # compute the eigenvalues/eigenvectors
    # here's where we get funky...
    # get the index of the smallest eigenvalue, which should
    # correspond to the best approximation of the barycentric coordinates
    # let v be the (normalized) corresponding eigenvector
    i = np.argmin(evals)
    v = np.abs(evecs[i]) / np.linalg.norm(evecs[i], 1)
    b, _ = barycenter(refs, v.T)
    return np.linalg.norm(nt - b)


def werenski_test(n: int, dim: int):
    '''
    a priori, if we compute an explicit barycenter and
    solve for the barycentric coordinates
    with the werenski criterion, we should
    necessarily see a 0 eigenvalue at the end of this
    and the corresponding eigenvector should recover the
    barycentric coordinates
    '''
    refs = []
    for _ in range(n):
        x = np.random.rand(dim, dim)
        refs.append((0.5 * (x + x.T)) + dim * np.identity(dim))
    l0 = simplex_point(n)
    print(f"The true coordinates are:\n {l0}\n")
    n0, _ = barycenter(refs, l0)
    a = werenski_matrix(refs, n0)
    evals, evecs = np.linalg.eig(a)
    i = np.argmin(evals)
    # note that a positive definite matrix has strictly nonnegative eigenvalues
    v = np.abs(evecs[i]) / np.linalg.norm(evecs[i], 1)
    print(f"The recovered barycentric coordinates are:\n {v}\n")
    b, _ = barycenter(refs, v.T)
    diff = np.linalg.norm(n0 - b)
    print(f"The Frobenius norm difference between recovererd barycenter\
    and its barycentric approximation is {diff}")


if __name__ == "__main__":
    main(args.ref_num, args.dim_num)
