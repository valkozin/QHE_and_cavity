"""Landauer-Buettiker model of the six-terminal Hall bar with an edge that
carries, besides the nu chiral channels, a counter-propagating pair.

Whatever happens *inside* one edge segment between two neighbouring contacts
(elastic backscattering, inter-channel mixing, equilibration), current
conservation fixes

    T(forward) - T(backward) = nu,

so a segment is fully described by one number, the backward transmission
t_b in [0, 1] (probability for an electron emitted by the downstream contact
to reach the upstream one against the chirality).  t_b = 0 is the perfect
QHE edge, t_b = 1 a ballistic counter-propagating channel.

Contacts are numbered as in model.py; with our sign of B electrons circulate
0 -> 5 -> 4 -> 3 -> 2 -> 1 -> 0 (top edge left-to-right, bottom edge
right-to-left).
"""
import numpy as np

ORDER = [0, 5, 4, 3, 2, 1]          # chirality order around the perimeter
SEGMENTS = [(ORDER[i], ORDER[(i + 1) % 6]) for i in range(6)]
BOTTOM = [(3, 2), (2, 1), (1, 0)]
TOP = [(0, 5), (5, 4), (4, 3)]


def conductance_matrix(nu, tb):
    """tb: dict {(upstream, downstream): t_b}. Missing segments -> 0.
    Returns G with I = G V (units e^2/h)."""
    T = np.zeros((6, 6))
    for (a, b) in SEGMENTS:
        t = tb.get((a, b), 0.0)
        T[b, a] += nu + t      # a -> b along the chirality
        T[a, b] += t           # b -> a against it
    G = -T.copy()
    np.fill_diagonal(G, T.sum(axis=0))
    return G


def resistances(nu, tb):
    from model import solve_voltages
    V = solve_voltages(conductance_matrix(nu, tb))
    return dict(Rxy_L=V[5] - V[1], Rxy_R=V[4] - V[2],
                Rxx_bot=V[1] - V[2], Rxx_top=V[5] - V[4], R2t=V[0] - V[3], V=V)
