"""Image-charge edge potential of arXiv:2511.04744 in the six-terminal Hall
bar and in a plain two-terminal bar.

  * B sweeps at fixed E_F (as in run_sweeps.py)          -> data/img_*.npz
  * E_F sweeps at fixed B (the protocol of the paper)    -> data/ef_*.npz

    python image_sweeps.py            # everything that is not yet in data/
"""
import os
import time
import warnings
from dataclasses import replace
from multiprocessing import get_context

import numpy as np

from model import (Params, Potential, make_bar, make_hallbar, transport,
                   transport_2t)
from run_sweeps import EF, PHIS, U

warnings.filterwarnings('ignore')

# metal at dedge = 18 a from the edge (~ 8 l_B at nu = 1, as in the experiment
# estimate of the paper); Eimg sets Gamma_0 ~ 0.1 hbar*omega_c at nu = 1
IMG = dict(kind='image', Eimg=9.0, dedge=18.0, edges='bottom')
PHI_EF = 0.03
EFS = np.round(np.arange(0.0, 0.8001, 0.004), 4)
SEEDS = (1, 2, 3)

B_CONFIGS = {f'img_bottom{"" if s == 1 else f"_s{s}"}':
             Params(pot=Potential(**IMG), U=U, seed=s) for s in SEEDS}
B_CONFIGS['img_bottom_clean'] = Params(pot=Potential(**IMG), U=0.0)

EF_CONFIGS = {}
for s in SEEDS:
    EF_CONFIGS[f'ref_s{s}'] = Params(pot=Potential(), U=U, seed=s)
    EF_CONFIGS[f'img_s{s}'] = Params(pot=Potential(**IMG), U=U, seed=s)
EF_CONFIGS['img_clean'] = Params(pot=Potential(**IMG), U=0.0)
EF_CONFIGS['ref_clean'] = Params(pot=Potential(), U=0.0)


def _safe(fs, gauge, phi, E):
    try:
        r = transport(fs, gauge, phi, E)
        return r['V'], r['T'], r['N']
    except np.linalg.LinAlgError:     # some probe has no propagating mode
        return np.full(6, np.nan), np.full((6, 6), np.nan), np.zeros(6, int)


def job(task):
    kind, name, prm = task
    out = f'data/{kind}_{name}.npz'
    if os.path.exists(out):
        return f'skip {out}'
    t0 = time.time()
    if kind == 'img':
        fs, g = make_hallbar(prm)
        res = [_safe(fs, g, phi, EF) for phi in PHIS]
        np.savez(out, phi=PHIS, EF=EF, V=[r[0] for r in res],
                 T=[r[1] for r in res], N=[r[2] for r in res], params=repr(prm))
    elif kind == 'efhb':
        fs, g = make_hallbar(prm)
        res = [_safe(fs, g, PHI_EF, E) for E in EFS]
        np.savez(out, EF=EFS, phi=PHI_EF, V=[r[0] for r in res],
                 T=[r[1] for r in res], N=[r[2] for r in res], params=repr(prm))
    else:  # 'ef2t'
        fs, g = make_bar(prm)
        res = [transport_2t(fs, g, PHI_EF, E) for E in EFS]
        np.savez(out, EF=EFS, phi=PHI_EF, G=[r[0] for r in res],
                 N=[r[1] for r in res], params=repr(prm))
    return f'{out}: {time.time() - t0:.0f} s'


def main():
    os.makedirs('data', exist_ok=True)
    tasks = [('img', n.replace('img_', '', 1), p) for n, p in B_CONFIGS.items()]
    tasks += [(k, n, p) for k in ('efhb', 'ef2t') for n, p in EF_CONFIGS.items()]
    with get_context('fork').Pool(4) as pool:
        for msg in pool.imap_unordered(job, tasks):
            print(msg, flush=True)


if __name__ == '__main__':
    main()
