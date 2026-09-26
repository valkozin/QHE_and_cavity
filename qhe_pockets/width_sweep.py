"""Overlap study: at fixed B (nu = 1 plateau, pair present) vary the pocket
width, i.e. the spatial separation of the counter-propagating pair, and
record the backward transmission t_b of each bottom segment and the
resistances. Several disorder realizations.

    python width_sweep.py  -> data/width_sweep.npz
"""
import os
import time
import warnings
from multiprocessing import get_context

import numpy as np

from edge_modes import bottom_pair
from model import Params, Potential, make_hallbar, transport

warnings.filterwarnings('ignore')

PHI = 0.028
EF = 0.4
VDIP = 0.35
WIDTHS = [3, 4, 5, 6, 8, 10, 12, 14, 16]
SEEDS = [1, 2, 3]
U = 1.0


def prm_for(w, seed):
    return Params(pot=Potential(Vdip=VDIP, d_in=6, d_out=6 + w, edges='bottom'),
                  U=U, seed=seed)


def job(args):
    w, seed = args
    fs, gauge = make_hallbar(prm_for(w, seed))
    r = transport(fs, gauge, PHI, EF)
    return w, seed, r['V'], r['T']


def main():
    os.makedirs('data', exist_ok=True)
    t0 = time.time()
    jobs = [(w, s) for w in WIDTHS for s in SEEDS]
    with get_context('fork').Pool(4) as pool:
        res = pool.map(job, jobs, chunksize=1)
    V = np.zeros((len(WIDTHS), len(SEEDS), 6))
    T = np.zeros((len(WIDTHS), len(SEEDS), 6, 6))
    for w, s, v, t in res:
        V[WIDTHS.index(w), SEEDS.index(s)] = v
        T[WIDTHS.index(w), SEEDS.index(s)] = t
    lB = 1 / np.sqrt(2 * np.pi * PHI)
    sep, ov = [], []
    for w in WIDTHS:
        bp = bottom_pair(prm_for(w, 1), PHI, EF)
        sep.append(bp[0] / lB if bp else np.nan)
        ov.append(bp[1] if bp else np.nan)
    np.savez('data/width_sweep.npz', widths=WIDTHS, seeds=SEEDS, V=V, T=T,
             sep_lB=sep, overlap=ov, phi=PHI, EF=EF, U=U, Vdip=VDIP)
    print(f'width sweep: {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
