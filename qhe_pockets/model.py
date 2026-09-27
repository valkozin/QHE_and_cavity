"""Tight-binding model of a six-terminal Hall bar with a smooth confining
potential that optionally has a shallow "pocket" (potential well) right in
front of the edge wall.

Units: lattice constant a = 1, hopping t = 1, band bottom at E = 0.
Magnetic field is given as flux ``phi`` (in flux quanta h/e) per plaquette:
    hbar*omega_c ~= 4*pi*phi*t,   l_B = 1/sqrt(2*pi*phi).

Terminal numbering (counter-clockwise, looking from +z):

           5 (top-left)      4 (top-right)
             |                  |
    0 ------------------------------------- 3
    (source)                              (drain)
             |                  |
           1 (bottom-left)   2 (bottom-right)
"""
from dataclasses import dataclass, field, replace

import numpy as np
from scipy import ndimage

import kwant

LAT = kwant.lattice.square(1, norbs=1)


@dataclass(frozen=True)
class Geometry:
    L: int = 300          # length of the bar (x)
    W: int = 80           # width of the bar (y)
    arm_w: int = 56       # width of the voltage-probe arms (>> pocket width,
                          # so that the edge pair survives inside the arms)
    arm_len: int = 30     # length of the arms
    arm_x: tuple = (60, 184)  # left x-coordinate of the two arm pairs
    pad: int = 60         # padding used to make lead cross-sections uniform


@dataclass(frozen=True)
class Potential:
    """Edge potential as a function of distance d to the sample boundary.

    V(d) = Vwall * ((d0 - d)/d0)^2 * theta(d0 - d)            (smooth wall)
           - Vdip * window(d; d_in, d_out, s) * selector        (pocket)
    or, for kind == 'image', the image-charge term below instead of the window.

    ``edges`` chooses where the pocket exists:
      'none'     - standard sample
      'both'     - pocket along the whole perimeter (incl. arms and leads)
      'bottom'   - pocket only along the bottom edge (incl. bottom arms,
                   lower halves of source/drain leads) -> the extra
                   counter-propagating pair is connected to the contacts
      'segment'  - pocket only along the bottom edge between the two bottom
                   probes, not touching any contact (closed loop)
    """
    Vwall: float = 1.5
    d0: float = 6.0
    Vdip: float = 0.0
    d_in: float = 6.0     # outer border of the pocket (towards the wall)
    d_out: float = 20.0   # inner border of the pocket (towards the bulk)
    s: float = 1.0        # smoothing of the pocket borders
    edges: str = 'none'
    seg: tuple = (130, 170)  # x-window for edges == 'segment'
    seg_s: float = 3.0
    # kind == 'image': instead of the rectangular pocket use the image-charge
    # potential of a metal plate parallel to the edge (arXiv:2511.04744),
    #   -Eimg * [1/(d + dedge) - 1/(dc + dedge)] * theta(dc - d),
    # dedge = edge-metal distance; the constant makes it vanish at d = dc
    # (middle of the bar), so the bulk and the opposite edge are untouched.
    kind: str = 'window'
    Eimg: float = 0.0
    dedge: float = 18.0
    dc: float = 40.0


@dataclass(frozen=True)
class Params:
    geo: Geometry = field(default_factory=Geometry)
    pot: Potential = field(default_factory=Potential)
    U: float = 0.0        # Anderson disorder: onsite uniform in [-U/2, U/2]
    seed: int = 1


def wall(d, p):
    d = np.asarray(d, float)
    return p.Vwall * np.clip((p.d0 - d) / p.d0, 0, None) ** 2


def window(d, p):
    d = np.asarray(d, float)
    return 0.5 * (np.tanh((d - p.d_in) / p.s) - np.tanh((d - p.d_out) / p.s))


def image(d, p):
    d = np.asarray(d, float)
    return p.Eimg * np.where(d < p.dc, 1 / (d + p.dedge) - 1 / (p.dc + p.dedge), 0.0)


def edge_profile(d, p, sel=1.0):
    if p.kind == 'image':
        return wall(d, p) - image(d, p) * sel
    return wall(d, p) - p.Vdip * window(d, p) * sel


def _masks(g):
    """Boolean masks on a padded grid. Returns (x0, y0, full, system)."""
    x0 = -g.pad - 2
    y0 = -g.arm_len - g.pad - 2
    nx = g.L + 2 * g.pad + 4
    ny = g.W + 2 * g.arm_len + 2 * g.pad + 4
    X, Y = np.meshgrid(np.arange(nx) + x0, np.arange(ny) + y0, indexing='ij')
    body_ext = (X >= -g.pad) & (X < g.L + g.pad) & (Y >= 0) & (Y < g.W)
    body = (X >= 0) & (X < g.L) & (Y >= 0) & (Y < g.W)
    arms_ext = np.zeros_like(body)
    arms = np.zeros_like(body)
    for xa in g.arm_x:
        inx = (X >= xa) & (X < xa + g.arm_w)
        arms_ext |= inx & (Y >= -g.arm_len - g.pad) & (Y < g.W + g.arm_len + g.pad)
        arms |= inx & (Y >= -g.arm_len) & (Y < g.W + g.arm_len)
    full = body_ext | arms_ext
    system = body | arms
    return x0, y0, full, system


def site_potential(prm):
    """Dict {(x, y): V} for all sites of the scattering region."""
    g, p = prm.geo, prm.pot
    x0, y0, full, system = _masks(g)
    dist = ndimage.distance_transform_edt(full)
    xs, ys = np.nonzero(system)
    X, Y = xs + x0, ys + y0
    d = dist[xs, ys]
    sel = selector(X, Y, p, g)
    V = edge_profile(d, p, sel)
    if prm.U:
        rng = np.random.default_rng(prm.seed)
        V = V + prm.U * (rng.random(V.shape) - 0.5)
    return {(int(a), int(b)): float(v) for a, b, v in zip(X, Y, V)}


def selector(X, Y, p, g):
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if p.edges == 'none':
        return np.zeros_like(X)
    if p.edges == 'both':
        return np.ones_like(X)
    bottom = (Y < g.W / 2).astype(float)
    if p.edges == 'bottom':
        return bottom
    if p.edges == 'segment':
        a, b = p.seg
        env = 0.5 * (np.tanh((X - a) / p.seg_s) - np.tanh((X - b) / p.seg_s))
        # keep the pocket off the probe arms: only in the body, y >= 0
        return bottom * env * (Y >= 0)
    raise ValueError(p.edges)


def lead_potential_x(y, prm):
    """Potential in source/drain leads (translation along x)."""
    g, p = prm.geo, prm.pot
    y = np.asarray(y)
    d = np.minimum(y + 1, g.W - y)
    sel = selector(np.zeros_like(y), y, p, g) if p.edges != 'segment' else 0.0
    return edge_profile(d, p, sel)


def lead_potential_y(x, xa, bottom, prm):
    """Potential in probe leads (translation along y)."""
    g, p = prm.geo, prm.pot
    x = np.asarray(x)
    d = np.minimum(x - xa + 1, xa + g.arm_w - x)
    if p.edges in ('none', 'segment'):
        sel = 0.0
    elif p.edges == 'both':
        sel = 1.0
    else:  # bottom
        sel = 1.0 if bottom else 0.0
    return edge_profile(d, p, sel)


def _hop(s1, s2, peierls):
    return -peierls(s1, s2)


def _lead_hop(i):
    """Hopping for lead i; Kwant needs a distinct parameter name per lead."""
    ns = {}
    exec(f"def h(s1, s2, peierls_lead{i}):\n    return -peierls_lead{i}(s1, s2)", ns)
    return ns['h']


def make_hallbar(prm):
    """Finalized Kwant system + magnetic gauge object."""
    g = prm.geo
    Vsite = site_potential(prm)
    syst = kwant.Builder()
    for (x, y), v in Vsite.items():
        syst[LAT(x, y)] = 4.0 + v
    syst[LAT.neighbors()] = _hop

    def lead_x(direction, i):
        ld = kwant.Builder(kwant.TranslationalSymmetry((direction, 0)))
        for y in range(g.W):
            ld[LAT(0, y)] = 4.0 + float(lead_potential_x(y, prm))
        ld[LAT.neighbors()] = _lead_hop(i)
        return ld

    def lead_y(xa, direction, i):
        ld = kwant.Builder(kwant.TranslationalSymmetry((0, direction)))
        for x in range(xa, xa + g.arm_w):
            ld[LAT(x, 0)] = 4.0 + float(lead_potential_y(x, xa, direction < 0, prm))
        ld[LAT.neighbors()] = _lead_hop(i)
        return ld

    leads = [lead_x(-1, 0),                    # 0 source (left)
             lead_y(g.arm_x[0], -1, 1),        # 1 bottom-left
             lead_y(g.arm_x[1], -1, 2),        # 2 bottom-right
             lead_x(+1, 3),                    # 3 drain (right)
             lead_y(g.arm_x[1], +1, 4),        # 4 top-right
             lead_y(g.arm_x[0], +1, 5)]        # 5 top-left
    for ld in leads:
        syst.attach_lead(ld)
    fsyst = syst.finalized()
    gauge = kwant.physics.magnetic_gauge(fsyst)
    return fsyst, gauge


def make_ribbon(prm, sel_bottom=True, sel_top=True):
    """Infinite ribbon (translation along x) with the edge profile, for bands
    and mode profiles. Uses the Landau gauge A = (-B y, 0)."""
    g, p = prm.geo, prm.pot
    ld = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))
    for y in range(g.W):
        d = min(y + 1, g.W - y)
        on = (sel_bottom if y < g.W / 2 else sel_top)
        ld[LAT(0, y)] = 4.0 + float(edge_profile(d, p, float(on)))

    def hopx(s1, s2, phi):
        y = s1.pos[1]
        return -np.exp(-2j * np.pi * phi * y * (s1.pos[0] - s2.pos[0]))
    ld[kwant.builder.HoppingKind((1, 0), LAT)] = hopx
    ld[kwant.builder.HoppingKind((0, 1), LAT)] = -1.0
    return ld.finalized()


def solve_voltages(G, source=0, drain=3, I=1.0):
    """Given the conductance matrix (I = G V, in e^2/h), inject I into
    `source`, extract at `drain` (V_drain = 0), return voltages (h/e^2)."""
    n = G.shape[0]
    keep = [i for i in range(n) if i != drain]
    Ivec = np.zeros(n)
    Ivec[source] = I
    Ivec[drain] = -I
    V = np.zeros(n)
    V[keep] = np.linalg.solve(G[np.ix_(keep, keep)], Ivec[keep])
    return V


def transport(fsyst, gauge, phi, E):
    """Compute all resistances for flux phi and Fermi energy E."""
    # Calibrated against the Landau-gauge ribbon (LL spacing 4*pi*phi):
    # kwant.physics.magnetic_gauge needs B = 2*phi for phi flux quanta/plaquette.
    B = 2 * phi
    phases = gauge(B, *[B] * len(fsyst.leads))
    params = {'peierls': phases[0]}
    for i, ph in enumerate(phases[1:]):
        params[f'peierls_lead{i}'] = ph
    sm = kwant.smatrix(fsyst, E, params=params)
    G = np.array(sm.conductance_matrix())
    T = np.array([[sm.transmission(i, j) if i != j else 0.0
                   for j in range(6)] for i in range(6)])
    V = solve_voltages(G)
    out = dict(
        Rxy_L=V[1] - V[5],       # Hall voltage across the left probe pair
        Rxy_R=V[2] - V[4],       # Hall voltage across the right probe pair
        Rxx_bot=V[1] - V[2],     # longitudinal along bottom edge
        Rxx_top=V[5] - V[4],     # longitudinal along top edge
        R2t=V[0] - V[3],         # two-terminal
        V=V, T=T, N=np.array([sm.num_propagating(i) for i in range(6)]),
    )
    return out


def make_bar(prm):
    """Plain two-terminal bar (no probe arms), same edge profile: the
    geometry in which arXiv:2511.04744 computes G(E_F)."""
    g = replace(prm.geo, arm_x=())
    prm = replace(prm, geo=g)
    syst = kwant.Builder()
    for (x, y), v in site_potential(prm).items():
        syst[LAT(x, y)] = 4.0 + v
    syst[LAT.neighbors()] = _hop
    for direction, i in [(-1, 0), (+1, 1)]:
        ld = kwant.Builder(kwant.TranslationalSymmetry((direction, 0)))
        for y in range(g.W):
            ld[LAT(0, y)] = 4.0 + float(lead_potential_x(y, prm))
        ld[LAT.neighbors()] = _lead_hop(i)
        syst.attach_lead(ld)
    fsyst = syst.finalized()
    return fsyst, kwant.physics.magnetic_gauge(fsyst)


def transport_2t(fsyst, gauge, phi, E):
    """Two-terminal conductance G = T(0 -> 1) in e^2/h."""
    B = 2 * phi
    phases = gauge(B, B, B)
    sm = kwant.smatrix(fsyst, E, params={'peierls': phases[0],
                                         'peierls_lead0': phases[1],
                                         'peierls_lead1': phases[2]})
    return sm.transmission(1, 0), sm.num_propagating(0)
