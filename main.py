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
import scipy as sp
import matplotlib.pyplot as plt

from tqdm import tqdm
from os import path
from gaussian_utilities import opt_lam_grad_norm
from gaussbarys import simplex_point, barycenter, werenski_matrix

sqrtm = sp.linalg.sqrtm
inv = sp.linalg.inv


parser = argparse.ArgumentParser()
parser.add_argument("ref_num", type=int)
parser.add_argument("dim_num", type=int)
parser.add_argument("iteration_num", type=int)
parser.add_argument("prange", type=int)
parser.add_argument("dimrange", type=int)
parser.add_argument("job")

args = parser.parse_args()

IMG_PATH='./img/'

def main(p: int,
         dim: int,
         iterations=100,
         prange=10,
         dimrange=10,
         job=None):
    '''
    main routine
    n is the number of references,
    d is the dimension of the matrices
    '''
    if job=='p':
        pcomplexity_graphs(dim,iterations, prange,)
    elif job == 'd':
        dcomplexity_graphs(p, iterations, dimrange)
    else:
        print_to_cli(p, dim, iterations)


def dcomplexity_graphs(p: int, iterations: int, drange: int):
    output_path = path.join(IMG_PATH, f'd-complexity.png')
    pass

def pcomplexity_graphs(
                dim: int,
                iterations: int,
                prange: int,
                ):
    output_path = path.join(IMG_PATH, f'p-complexity.png')
    fig, ax = plt.subplots()

    xp = np.arange(1, prange)

    yp = [np.mean([generate_sample(p, dim) for _ in tqdm(range(iterations))]) for p in range(1, prange)]

    ax.scatter(xp, yp)
    ax.set(xlim=(0,prange), xticks=np.arange(1,prange))
    plt.savefig(output_path)




def print_to_cli(p: int, dim:int, iterations: int):
    werenski_test(p, dim)
    samples = [generate_sample(p, dim) for _ in tqdm(range(iterations))]
    sample_mean = np.mean(samples)
    sample_std = np.std(samples)
    lead_string = f"Statistics for Frobenius norm discrepancy {p} matrices with dimensions {dim} x {dim}:\n"
    sample_string = f"Sample size: {iterations}\n"
    mean_string = f"Mean: {sample_mean}\n"
    std_string = f"Standard Deviation: {sample_std}\n"
    print(lead_string)
    print(sample_string)
    print(mean_string)
    print(std_string)


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
    refs = []
    for _ in range(p):                      # generate some covariance matrices
        x = np.random.rand(dim, dim)
        refs.append((0.5 * (x + x.T)) + dim * np.identity(dim))
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
    measures = []
    for _ in range(p):
        x = np.random.rand(dim, dim)
        measures.append((0.5 * (x + x.T)) + dim * np.identity(dim))
    l0 = simplex_point(p)
    n0, _ = barycenter(measures, l0)
    a = werenski_matrix(measures, n0)
    evals, evecs = np.linalg.eig(a)
    for vec in evecs.T:
        print(np.squeeze(vec / np.linalg.norm(vec, 1)))
    i = np.argmin(evals)
    v = evecs[:, i] / np.linalg.norm(evecs[:, i], 1)
    l2diff = np.linalg.norm(l0 - v)
    return l0, v, evals[i], l2diff


if __name__ == "__main__":
    main(args.ref_num,
         args.dim_num,
         args.iteration_num,
         args.prange,
         args.dimrange,
         args.job)
