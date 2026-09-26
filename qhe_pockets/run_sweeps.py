"""Magnetic-field sweeps of R_xy and R_xx for several edge potentials.

    python run_sweeps.py            # all configurations -> data/<name>.npz
    python run_sweeps.py ref wide   # selected ones
"""
import os
import sys
import time
import warnings
from dataclasses import replace
from multiprocessing import get_context

import numpy as np

from model import Params, Potential, make_hallbar, transport

warnings.filterwarnings('ignore')

EF = 0.4
PHIS = np.round(np.arange(0.008, 0.0601, 0.0004), 5)
U = 1.0

WIDE = dict(Vdip=0.3, d_in=6, d_out=20)     # pair separated by >> l_B
NARROW = dict(Vdip=0.35, d_in=6, d_out=10)  # pair overlapping (~ l_B apart)

CONFIGS = {
    'ref':             Params(pot=Potential(), U=U),
    'wide_bottom':     Params(pot=Potential(**WIDE, edges='bottom'), U=U),
    'narrow_bottom':   Params(pot=Potential(**NARROW, edges='bottom'), U=U),
    'wide_both':       Params(pot=Potential(**WIDE, edges='both'), U=U),
    'narrow_both':     Params(pot=Potential(**NARROW, edges='both'), U=U),
    'wide_segment':    Params(pot=Potential(**WIDE, edges='segment'), U=U),
    'ref_clean':       Params(pot=Potential(), U=0.0),
    'wide_bottom_clean':   Params(pot=Potential(**WIDE, edges='bottom'), U=0.0),
    'narrow_bottom_clean': Params(pot=Potential(**NARROW, edges='bottom'), U=0.0),
}

# extra disorder realizations for ensemble averages
for _name in ['ref', 'wide_bottom', 'narrow_bottom']:
    for _seed in (2, 3):
        CONFIGS[f'{_name}_s{_seed}'] = replace(CONFIGS[_name], seed=_seed)

_SYS = None


def _point(phi):
    fs, gauge = _SYS
    r = transport(fs, gauge, phi, EF)
    return r['V'], r['T'], r['N']


def sweep(prm, phis=PHIS, nproc=4):
    global _SYS
    _SYS = make_hallbar(prm)
    with get_context('fork').Pool(nproc) as pool:
        res = pool.map(_point, phis, chunksize=1)
    V = np.array([r[0] for r in res])
    T = np.array([r[1] for r in res])
    N = np.array([r[2] for r in res])
    return V, T, N


def main(names):
    os.makedirs('data', exist_ok=True)
    for name in names:
        out = f'data/{name}.npz'
        if os.path.exists(out):
            print('skip', out)
            continue
        t0 = time.time()
        V, T, N = sweep(CONFIGS[name])
        np.savez(out, phi=PHIS, EF=EF, V=V, T=T, N=N, params=repr(CONFIGS[name]))
        print(f'{name}: {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:] or list(CONFIGS))
