"""Edge modes of an infinite ribbon with the same edge profile: dispersion,
mode positions and the overlap of the counter-propagating pair."""
import numpy as np

import kwant

from model import make_ribbon


def bands(prm, phi, ks=np.linspace(-np.pi, np.pi, 1201), sel_bottom=True,
          sel_top=False):
    rib = make_ribbon(prm, sel_bottom, sel_top)
    b = kwant.physics.Bands(rib, params=dict(phi=phi))
    return ks, np.array([b(k) for k in ks])


def ll_vs_position(prm, phi, nk=721, nbands=8, sel_bottom=True, sel_top=False):
    """Eigen-energies of the ribbon versus the mean position <y> of the
    eigenstate: the 'bent Landau levels' E_n(y)."""
    rib = make_ribbon(prm, sel_bottom, sel_top)
    p = dict(phi=phi)
    H0 = rib.cell_hamiltonian(params=p)
    V = rib.inter_cell_hopping(params=p)
    y = np.array([s.pos[1] for s in rib.sites[:H0.shape[0]]])
    Ys, Es = [], []
    for k in np.linspace(-np.pi, np.pi, nk, endpoint=False):
        Hk = H0 + V * np.exp(-1j * k) + V.conj().T * np.exp(1j * k)
        e, v = np.linalg.eigh(Hk)
        w = np.abs(v[:, :nbands]) ** 2
        Ys.append(y @ w)
        Es.append(e[:nbands])
    return np.concatenate(Ys), np.concatenate(Es)


def modes(prm, phi, E, sel_bottom=True, sel_top=False):
    """List of dicts (v, y, w) for all propagating modes at energy E;
    w is the normalized probability density across the ribbon."""
    rib = make_ribbon(prm, sel_bottom, sel_top)
    pm, _ = rib.modes(E, params=dict(phi=phi))
    y = np.arange(pm.wave_functions.shape[0])
    out = []
    for i, v in enumerate(pm.velocities):
        w = np.abs(pm.wave_functions[:, i]) ** 2
        w /= w.sum()
        out.append(dict(v=float(v), y=float((w * y).sum()), w=w,
                        k=float(pm.momenta[i])))
    return out


def bottom_pair(prm, phi, E):
    """Counter-propagating mode on the bottom edge and its nearest
    co-propagating partner. Returns (sep, overlap, counter, partner) or None.

    overlap = sum_y |psi_a| |psi_b|  (1 for identical, 0 for disjoint)."""
    W = prm.geo.W
    ms = [m for m in modes(prm, phi, E) if m['y'] < W / 2]
    # on the bottom edge of the ribbon (translation (-1,0), our B sign) the
    # chiral QHE channels have v < 0; the counter-propagating one has v > 0
    counter = [m for m in ms if m['v'] > 0]
    chiral = [m for m in ms if m['v'] < 0]
    if not counter:
        return None
    c = counter[0]
    p = min(chiral, key=lambda m: abs(m['y'] - c['y']))
    ov = float(np.sum(np.sqrt(c['w'] * p['w'])))
    return abs(c['y'] - p['y']), ov, c, p
