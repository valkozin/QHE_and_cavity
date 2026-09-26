"""Figures from data/*.npz -> figs/*.png"""
import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import lb
from edge_modes import bottom_pair, ll_vs_position, modes
from model import Params, Potential, edge_profile
from run_sweeps import CONFIGS, EF, NARROW, WIDE

warnings.filterwarnings('ignore')

INK = '#0b0b0b'
INK2 = '#52514e'
GRID = '#e4e3df'
REF = '#52514e'          # reference sample: neutral
C_WIDE = '#2a78d6'       # blue
C_NARROW = '#eb6834'     # orange
C_3 = '#1baf7a'          # aqua
C_4 = '#4a3aa7'          # violet

plt.rcParams.update({
    'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
    'axes.spines.right': False, 'legend.frameon': False, 'lines.linewidth': 1.8,
    'figure.dpi': 130, 'savefig.bbox': 'tight',
})

os.makedirs('figs', exist_ok=True)


def load(name, average=True):
    """Load a sweep; if extra disorder realizations <name>_s2, _s3 exist,
    average the voltages and transmissions over them."""
    files = [f'data/{name}.npz']
    if average:
        files += [f'data/{name}_s{k}.npz' for k in (2, 3)
                  if os.path.exists(f'data/{name}_s{k}.npz')]
    ds = [np.load(f) for f in files]
    d = dict(phi=ds[0]['phi'], N=ds[0]['N'],
             V=np.mean([x['V'] for x in ds], axis=0),
             T=np.mean([x['T'] for x in ds], axis=0))
    V = d['V']
    return dict(phi=d['phi'], T=d['T'], N=d['N'], nreal=len(ds),
                Rxy_L=V[:, 5] - V[:, 1], Rxy_R=V[:, 4] - V[:, 2],
                Rxx_bot=V[:, 1] - V[:, 2], Rxx_top=V[:, 5] - V[:, 4],
                R2t=V[:, 0] - V[:, 3],
                tb=np.stack([V[:, 0] * 0 + d['T'][:, 3, 2], d['T'][:, 2, 1],
                             d['T'][:, 1, 0]], axis=1))


def pair_mask(pot, phis):
    prm = Params(pot=pot)
    return np.array([bottom_pair(prm, p, EF) is not None for p in phis])


def shade(ax, phis, mask, color, alpha=0.10):
    """Shade phi-ranges where the counter-propagating pair exists."""
    dp = phis[1] - phis[0]
    start = None
    for i, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = phis[i]
        if not m and start is not None:
            ax.axvspan((start - dp / 2) * 1e3, (phis[i - 1] + dp / 2) * 1e3, color=color,
                       alpha=alpha, lw=0)
            start = None


def nu_labels(ax, y=1.02):
    for nu in [1, 2, 3]:
        # plateau centres for hbar*wc = 4 pi phi, E_F between LL nu-1 and nu
        lo, hi = EF / (4 * np.pi * (nu + 0.5)), EF / (4 * np.pi * (nu - 0.5))
        c = np.sqrt(lo * hi)
        if 0.008 < c < 0.06:
            ax.text(c * 1e3, y, f'ν={nu}', transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom', color=INK2, fontsize=9)


# --------------------------------------------------------------------------
def fig_edge_structure():
    """Potential, Landau-level bending and edge modes for the three profiles."""
    phi = 0.028
    lB = 1 / np.sqrt(2 * np.pi * phi)
    cases = [('стандартный край', Potential(), REF),
             ('широкий карман (пара разнесена)',
              Potential(**WIDE, edges='bottom'), C_WIDE),
             ('узкий карман (пара перекрывается)',
              Potential(**NARROW, edges='bottom'), C_NARROW)]
    fig, axs = plt.subplots(3, 3, figsize=(12, 9.5), sharex=True,
                            gridspec_kw=dict(height_ratios=[1, 1.4, 1.2]))
    ymax = 32
    for j, (title, pot, col) in enumerate(cases):
        prm = Params(pot=pot)
        d = np.arange(1, 41)
        ax = axs[0, j]
        ax.plot(d - 1, edge_profile(d, pot, 1.0), color=col)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_ylim(-0.45, 1.0)
        ax.set_title(title, color=INK, fontsize=11)
        if j == 0:
            ax.set_ylabel('V(y) / t')

        ax = axs[1, j]
        Y, E = ll_vs_position(prm, phi)
        ax.scatter(Y, E, s=1.5, color=INK2, lw=0)
        ax.axhline(EF, color=col, lw=1.2, ls='--')
        ax.text(ymax - 1, EF + 0.02, 'E_F', color=INK2, ha='right', fontsize=9)
        ms = [m for m in modes(prm, phi, EF) if m['y'] < ymax]
        for m in ms:
            ax.annotate('', xy=(m['y'], EF + (0.09 if m['v'] > 0 else -0.09)),
                        xytext=(m['y'], EF),
                        arrowprops=dict(arrowstyle='-|>', lw=1.6,
                                        color=C_NARROW if m['v'] > 0 else INK))
        ax.set_ylim(-0.1, 1.2)
        if j == 0:
            ax.set_ylabel('E / t  (уровни Ландау E_n(⟨y⟩))')

        ax = axs[2, j]
        for m in ms:
            y = np.arange(len(m['w']))
            ax.fill_between(y, m['w'], color=C_NARROW if m['v'] > 0 else INK,
                            alpha=0.25 if m['v'] < 0 else 0.45, lw=0)
            ax.plot(y, m['w'], color=C_NARROW if m['v'] > 0 else INK, lw=1.2)
        bp = bottom_pair(prm, phi, EF)
        if bp:
            ax.text(0.97, 0.9, f'разнесение пары {bp[0] / lB:.1f} l_B\n'
                    f'перекрытие ∫|ψ₁||ψ₂| = {bp[1]:.2f}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=9,
                    color=INK)
        ax.set_xlabel('расстояние от края y (узлы решётки)')
        if j == 0:
            ax.set_ylabel('|ψ|² мод на E_F')
    axs[0, 0].set_xlim(0, ymax)
    fig.suptitle(f'Край образца при ν=1 (φ={phi}, l_B={lB:.1f}a): '
                 'стрелки: ↓ киральные каналы (к источнику), ↑ встречный канал',
                 color=INK)
    fig.savefig('figs/fig1_edge_structure.png')
    plt.close(fig)


# --------------------------------------------------------------------------
def panel_set(axs, series, masks=()):
    """axs: 4 axes (Rxy_L, Rxy_R, Rxx_bot, Rxx_top)."""
    keys = [('Rxy_L', 'R_xy, левая пара зондов (1–5)'),
            ('Rxy_R', 'R_xy, правая пара зондов (2–4)'),
            ('Rxx_bot', 'R_xx вдоль нижнего края (1–2)'),
            ('Rxx_top', 'R_xx вдоль верхнего края (5–4)')]
    for ax, (k, title) in zip(axs, keys):
        for m, c in masks:
            shade(ax, *m, c)
        for d, lab, col, ls in series:
            ax.plot(d['phi'] * 1e3, d[k], color=col, ls=ls, label=lab)
        ax.set_title(title, color=INK, fontsize=10, pad=14)
        if k.startswith('Rxy'):
            for nu in [1, 2, 3, 4]:
                ax.axhline(1 / nu, color=GRID, lw=1.0, zorder=0)
            ax.set_ylim(0, 1.15)
            ax.set_ylabel('R_xy  (h/e²)')
        else:
            ax.set_ylim(-0.02, 0.45)
            ax.set_ylabel('R_xx  (h/e²)')
        nu_labels(ax)
        ax.set_xlabel('B  (поток на плакетку φ × 10³)')
        ax.set_xlim(7.5, 60.5)


def fig_sweeps():
    ref = load('ref')
    wide = load('wide_bottom')
    narrow = load('narrow_bottom')
    mw = (ref['phi'], pair_mask(CONFIGS['wide_bottom'].pot, ref['phi']))
    mn = (ref['phi'], pair_mask(CONFIGS['narrow_bottom'].pot, ref['phi']))
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    panel_set(axs.ravel(), [
        (ref, 'стандартный край', REF, '-'),
        (wide, 'широкий карман (пара не перекрывается)', C_WIDE, '-'),
        (narrow, 'узкий карман (пара перекрывается)', C_NARROW, '-')],
        masks=[(mw, C_WIDE)])
    axs[0, 0].legend(loc='lower right', fontsize=9)
    fig.suptitle(f'Карман только на нижнем крае, беспорядок U = 1 t (среднее по '
                 f'{ref["nreal"]} реализациям). Заливка: B, при которых у нижнего края '
                 'есть встречная пара (широкий карман)', color=INK)
    fig.tight_layout()
    fig.savefig('figs/fig2_sweeps_bottom_pocket.png')
    plt.close(fig)

    # clean vs disordered
    refc = load('ref_clean')
    widec = load('wide_bottom_clean')
    narrowc = load('narrow_bottom_clean')
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    panel_set(axs.ravel(), [
        (refc, 'стандартный край, без беспорядка', REF, '-'),
        (widec, 'широкий карман, без беспорядка', C_WIDE, '-'),
        (narrowc, 'узкий карман, без беспорядка', C_NARROW, '-'),
        (narrow, 'узкий карман, U = 1 t', C_3, '--')],
        masks=[(mn, C_NARROW)])
    axs[0, 0].legend(loc='lower right', fontsize=9)
    fig.suptitle('Без беспорядка встречный канал баллистический (t_b = 1) и '
                 'квантование нарушено; обратное рассеяние (узкий карман + '
                 'беспорядок) его восстанавливает', color=INK)
    fig.tight_layout()
    fig.savefig('figs/fig3_clean_vs_disorder.png')
    plt.close(fig)

    # both edges and local pocket
    both_w = load('wide_both')
    both_n = load('narrow_both')
    seg = load('wide_segment')
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    panel_set(axs.ravel(), [
        (ref, 'стандартный край', REF, '-'),
        (both_w, 'широкий карман на обоих краях', C_WIDE, '-'),
        (both_n, 'узкий карман на обоих краях', C_NARROW, '-'),
        (seg, 'широкий карман только между зондами (петля)', C_3, '--')],
        masks=[(mw, C_WIDE)])
    axs[0, 0].legend(loc='lower right', fontsize=9)
    fig.suptitle('Карманы на обоих краях и локальный карман, не доходящий '
                 'до контактов (U = 1 t)', color=INK)
    fig.tight_layout()
    fig.savefig('figs/fig4_both_edges_and_local.png')
    plt.close(fig)

    # t_b and LB check
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axs[0]
    for d, lab, col, ls in [(widec, 'широкий, без беспорядка', C_WIDE, '-'),
                            (wide, 'широкий, U = 1 t', C_WIDE, ':'),
                            (narrowc, 'узкий, без беспорядка', C_NARROW, '-'),
                            (narrow, 'узкий, U = 1 t', C_NARROW, ':')]:
        ax.plot(d['phi'] * 1e3, d['tb'][:, 1], color=col, ls=ls, label=lab)
    ax.set_ylabel('t_b  (2 → 1 против киральности)\n(>1: две встречные пары)')
    ax.set_xlabel('B  (φ × 10³)')
    ax.set_title('Обратное прохождение по нижнему краю между зондами 1 и 2',
                 color=INK, fontsize=10, pad=14)
    nu_labels(ax)
    ax.legend(fontsize=9)
    ax = axs[1]
    t = np.linspace(0, 1, 101)
    for nu, col in [(1, INK), (2, C_WIDE)]:
        r = [lb.resistances(nu, {s: x for s in lb.BOTTOM}) for x in t]
        ax.plot(t, [q['Rxy_L'] * nu for q in r], color=col,
                label=f'ν={nu}: R_xy(1–5)·ν')
        ax.plot(t, [q['Rxy_R'] * nu for q in r], color=col, ls='--',
                label=f'ν={nu}: R_xy(2–4)·ν')
        ax.plot(t, [q['Rxx_bot'] * nu for q in r], color=col, ls=':',
                label=f'ν={nu}: R_xx(низ)·ν')
    # kwant points (clean wide, nu=1 & 2 plateau with pair)
    for d, mk, lab in [(widec, 'o', 'Kwant, без беспорядка'),
                       (load('wide_bottom', average=False), '^', 'Kwant, U = 1 t')]:
        nu = d['N'][:, 4]           # top-edge probe: number of chiral channels
        tb = d['tb'].mean(axis=1)
        # plateau regime only: clean top edge, pair present, one pair
        sel = (np.abs(d['Rxx_top']) < 0.005) & (tb > 0.02) & (tb <= 1.02) & (nu <= 2)
        ax.scatter(tb[sel], (d['Rxy_L'] * nu)[sel], s=18, marker=mk,
                   color=C_NARROW, alpha=0.7, zorder=3, label=lab)
    ax.set_xlabel('t_b  (среднее по сегментам нижнего края)')
    ax.set_ylabel('R · ν  (h/e²)')
    ax.set_title('Ландауэр–Бюттикер (линии) и Kwant (точки, R_xy(1–5))',
                 color=INK, fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig('figs/fig5_tb_and_LB.png')
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_width():
    d = np.load('data/width_sweep.npz')
    w = d['widths']
    T = d['T']
    V = d['V']
    tb = np.stack([T[..., 3, 2], T[..., 2, 1], T[..., 1, 0]], axis=-1)  # (w, seed, 3)
    rxy_l = V[..., 5] - V[..., 1]
    rxy_r = V[..., 4] - V[..., 2]
    rxx = V[..., 1] - V[..., 2]
    x = d['sep_lB']
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    ax = axs[0]
    ax.plot(x, d['overlap'], 'o-', color=INK)
    ax.set_ylabel('перекрытие пары ∫|ψ₁||ψ₂|')
    ax.set_title('Перекрытие встречной пары', color=INK, fontsize=10)
    ax = axs[1]
    for s in range(tb.shape[1]):
        ax.plot(x, tb[:, s, 1], 'o', color=C_WIDE, alpha=0.5, ms=4)
    ax.plot(x, tb[:, :, 1].mean(axis=1), '-', color=C_WIDE,
            label='сегмент 1–2 (между зондами, 68 a)')
    for s in range(tb.shape[1]):
        ax.plot(x, tb[:, s, 0], 's', color=C_NARROW, alpha=0.5, ms=4)
    ax.plot(x, tb[:, :, 0].mean(axis=1), '-', color=C_NARROW,
            label='сегмент 2–3 (зонд 2 – сток, 60 a)')
    ax.set_ylabel('t_b')
    ax.set_title('Обратное прохождение по краю', color=INK, fontsize=10)
    ax.legend(fontsize=9)
    ax = axs[2]
    for arr, col, lab in [(rxy_l, C_WIDE, 'R_xy (1–5)'),
                          (rxy_r, C_NARROW, 'R_xy (2–4)'),
                          (rxx, C_3, 'R_xx (низ)')]:
        ax.plot(x, arr, 'o', color=col, alpha=0.4, ms=4)
        ax.plot(x, arr.mean(axis=1), '-', color=col, label=lab)
    ax.axhline(1, color=GRID)
    ax.set_ylabel('R  (h/e²)')
    ax.set_title('Сопротивления при ν = 1', color=INK, fontsize=10)
    ax.legend(fontsize=9)
    for ax in axs:
        ax.set_xlabel('разнесение пары / l_B')
    fig.suptitle(f'Зависимость от ширины кармана (φ={float(d["phi"])}, '
                 f'V_dip={float(d["Vdip"])} t, U={float(d["U"])} t, '
                 f'{len(d["seeds"])} реализации беспорядка)', color=INK)
    fig.tight_layout()
    fig.savefig('figs/fig6_width_sweep.png')
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_wavefunctions(phi=0.028):
    """Current density carried by electrons injected from bottom probe 2."""
    import kwant
    from model import make_hallbar
    cases = [('стандартный край, U = 1 t', CONFIGS['ref']),
             ('широкий карман, без беспорядка', CONFIGS['wide_bottom_clean']),
             ('широкий карман, U = 1 t', CONFIGS['wide_bottom']),
             ('узкий карман, U = 1 t', CONFIGS['narrow_bottom'])]
    fig, axs = plt.subplots(2, 2, figsize=(13, 7.5))
    data = []
    for title, prm in cases:
        fs, gauge = make_hallbar(prm)
        B = 2 * phi
        ph = gauge(B, *[B] * len(fs.leads))
        params = {'peierls': ph[0]}
        params.update({f'peierls_lead{i}': p for i, p in enumerate(ph[1:])})
        psi = kwant.wave_function(fs, EF, params=params)(2)
        J = kwant.operator.Current(fs).bind(params=params)
        cur = sum(J(p) for p in psi)
        field, box = kwant.plotter.interpolate_current(fs, cur, relwidth=0.02)
        data.append((fs, field, box))
    # common colour scale, set by the ballistic edge channels of the clean case;
    # stronger local circulating currents around disorder resonances saturate
    vmax = 0.6 * np.linalg.norm(data[1][1], axis=-1).max()
    for ax, (title, prm), (fs, field, box) in zip(axs.ravel(), cases, data):
        mag = np.linalg.norm(field, axis=-1, keepdims=True)
        field = field * np.minimum(1.0, vmax / np.maximum(mag, 1e-300))
        kwant.plotter.streamplot(field, box, ax=ax, cmap='Oranges', vmax=vmax,
                                 linecolor=INK,
                                 max_linewidth=1.4, min_linewidth=0.25)
        pos = np.array([s.pos for s in fs.sites]).astype(int)
        g = prm.geo
        x0, y0 = pos[:, 0].min(), pos[:, 1].min()
        mask = np.zeros((pos[:, 1].max() - y0 + 3, pos[:, 0].max() - x0 + 3))
        mask[pos[:, 1] - y0 + 1, pos[:, 0] - x0 + 1] = 1.0
        ax.contour(np.arange(mask.shape[1]) + x0 - 1, np.arange(mask.shape[0]) + y0 - 1,
                   mask, levels=[0.5], colors=[INK2], linewidths=0.8)
        ext = (x0 - .5, pos[:, 0].max() + .5, y0 - .5, pos[:, 1].max() + .5)
        ax.set_xlim(ext[0] - 8, ext[1] + 8)
        ax.set_ylim(ext[2] - 2, ext[3] + 2)
        ax.set_title(title, color=INK, fontsize=10, loc='left')
        ax.grid(False)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        lab = {0: (x0 - 5, g.W / 2), 3: (g.L + 4, g.W / 2),
               1: (g.arm_x[0] + g.arm_w / 2, -g.arm_len + 6),
               2: (g.arm_x[1] + g.arm_w / 2, -g.arm_len + 6),
               4: (g.arm_x[1] + g.arm_w / 2, g.W + g.arm_len - 6),
               5: (g.arm_x[0] + g.arm_w / 2, g.W + g.arm_len - 6)}
        for k, (x, y) in lab.items():
            ax.text(x, y, str(k), ha='center', va='center', fontsize=11,
                    fontweight='bold', color=C_WIDE if k == 2 else INK)
    fig.suptitle(f'Плотность тока электронов, выпущенных зондом 2 (ν = 1, φ = {phi}).\n'
                 'Киральный путь 2 → 1 по нижнему краю; встречный канал уносит часть '
                 'потока 2 → 3', color=INK)
    fig.tight_layout()
    fig.savefig('figs/fig7_injection_from_probe2.png')
    plt.close(fig)


if __name__ == '__main__':
    fig_edge_structure()
    if not os.path.exists("figs/fig7_injection_from_probe2.png"):
        fig_wavefunctions()
    if os.path.exists('data/narrow_bottom_clean.npz'):
        fig_sweeps()
    if os.path.exists('data/width_sweep.npz'):
        fig_width()
