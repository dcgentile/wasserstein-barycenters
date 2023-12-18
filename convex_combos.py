#!/usr/bin/env python3
'''
a numerical experiment to test the hypothesis:
"the convex combination of coordinates gives
the coordinates of the convex combination"
'''

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from gaussbarys import barycenter
from utils import generate_refs, opt_lam_eigen, simplex_point


def convexity_test(p: int, d: int, step_count = 10):
    '''
    generate a family of measures, generate two arbitrary barycenters (l0, n0), (l1, n1) and walk
    the geodesic nt, comparing the recovered barycentric coordinates against lt
    '''
    l0, l1 = simplex_point(p), simplex_point(p)
    refs = generate_refs(p, d)
    n0, n1= barycenter(refs, l0), barycenter(refs, l1)
    hat_diffs = []
    tilde_diffs = []
    hat_tilde_diffs = []
    for i in tqdm(range(step_count)):
        t = i/step_count
        b = barycenter([n0, n1], [t, 1-t]) # geodesic in WB manifold
        coords = opt_lam_eigen(refs , b)
        btilde = barycenter(refs, coords)
        bhat = barycenter(refs, (t * l0) + ((1-t) * l1)) # geodesic in simplex
        tilde_diffs.append(np.linalg.norm(b - btilde))
        hat_diffs.append(np.linalg.norm(b - bhat))
        hat_tilde_diffs.append(np.linalg.norm(bhat - btilde))
    return np.arange(step_count), hat_diffs, tilde_diffs, hat_tilde_diffs

REF_COUNT = 5
DIM = 25
STEP_COUNT = 100

diffs = convexity_test(REF_COUNT, DIM, STEP_COUNT)

fig, ax = plt.subplots()

plt.title("Differences between true barycenters and \n Werenski/naive simplex apporoximants")
l1 = ax.scatter(diffs[0], diffs[1], c='r')
l2 = ax.scatter(diffs[0], diffs[2], c='b')
l3 = ax.scatter(diffs[0], diffs[3], c='g')

ax.legend((l1, l2, l3), ('Naive Simplex vs True Barycenter', 'Werenski Vs True Barycenter', 'Naive Simplex Vs Werenski'))
ax.set_xlabel(r'Time Step on $(1-t)\lambda_0 + t \lambda_1$')
ax.set_ylabel(r'Difference in $F$-norm')

plt.savefig(f'./img/diffs-{REF_COUNT}-{DIM}-{STEP_COUNT}.png')
