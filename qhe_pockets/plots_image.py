"""Figures for the image-charge edge potential (arXiv:2511.04744):
data/img_*.npz, data/efhb_*.npz, data/ef2t_*.npz -> figs/fig8..fig10"""
import numpy as np

import matplotlib.pyplot as plt

from edge_modes import bottom_pair, ll_vs_position, modes
from image_sweeps import EFS, IMG, PHI_EF, SEEDS
from model import Params, Potential, edge_profile
from plots import (C_3, C_NARROW, C_WIDE, GRID, INK, INK2, REF, load,
                   panel_set, shade)
from run_sweeps import EF, NARROW, WIDE

POT = Potential(**IMG)
C_IMG = '#4a3aa7'


def pair_vs(prm, phis, Es):
    """(exists, separation / l_B, overlap) of the bottom pair along a path."""
    out = []
    for phi, E in zip(phis, Es):
        bp = bottom_pair(prm, phi, E)
        lB = 1 / np.sqrt(2 * np.pi * phi)
        out.append((True, bp[0] / lB, bp[1]) if bp else (False, np.nan, np.nan))
    ex, sep, ov = map(np.array, zip(*out))
    return ex.astype(bool), sep, ov


def gamma0(phi):
    """Backscattering energy Gamma_0 = l_B * dU_im/dd at the pocket minimum."""
    d = np.arange(1, 41)
    dmin = d[np.argmin(edge_profile(d, POT))]
    return POT.Eimg / (dmin + POT.dedge) ** 2 / np.sqrt(2 * np.pi * phi)


def fig_profile():
    phi = PHI_EF
    lB = 1 / np.sqrt(2 * np.pi * phi)
    prm = Params(pot=POT)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axs[0]
    d = np.arange(1, 41)
    for pot, lab, col in [(Potential(**WIDE), 'широкий карман', C_WIDE),
                          (Potential(**NARROW), 'узкий карман', C_NARROW),
                          (POT, 'изображение, d_edge = 18a ≈ 8 l_B', C_IMG)]:
        ax.plot(d - 1, edge_profile(d, pot, 1.0), color=col, label=lab)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_ylim(-0.45, 1.0)
    ax.set_xlabel('расстояние от края  (a)')
    ax.set_ylabel('V / t')
    ax.set_title('Профиль потенциала у края', color=INK, fontsize=10)
    ax.legend(fontsize=8)

    ax = axs[1]
    Y, E = ll_vs_position(prm, phi)
    ax.scatter(Y, E, s=1.5, color=INK2, lw=0)
    for e in (0.4, 0.37):
        ax.axhline(e, color=C_IMG, lw=1.0, ls='--')
        for m in [m for m in modes(prm, phi, e) if m['y'] < 40]:
            ax.annotate('', xy=(m['y'], e + (0.07 if m['v'] > 0 else -0.07)),
                        xytext=(m['y'], e),
                        arrowprops=dict(arrowstyle='-|>', lw=1.4,
                                        color=C_NARROW if m['v'] > 0 else INK))
    ax.set_xlim(0, 40)
    ax.set_ylim(-0.1, 1.0)
    ax.set_xlabel('⟨y⟩  (a)')
    ax.set_ylabel('E / t')
    ax.set_title(f'Уровни Ландау у края, φ = {phi} (l_B = {lB:.1f}a)',
                 color=INK, fontsize=10)

    ax = axs[2]
    Es = np.linspace(0.2, 0.62, 211)
    ex, sep, ov = pair_vs(prm, [phi] * len(Es), Es)
    ax.plot(Es, sep, color=C_IMG, label='разнесение пары, l_B')
    ax2 = ax.twinx()
    ax2.plot(Es, ov, color=C_NARROW, ls='--', label='перекрытие ∫|ψ₁||ψ₂|')
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('перекрытие', color=C_NARROW)
    ax2.grid(False)
    e0 = Es[ex][0]
    ax.axvspan(e0, e0 + gamma0(phi), color=C_NARROW, alpha=0.12, lw=0)
    ax.text(e0 + gamma0(phi), 0.95, ' Γ₀', transform=ax.get_xaxis_transform(),
            color=C_NARROW, fontsize=9, va='top')
    ax.set_xlabel('E_F / t')
    ax.set_ylabel('разнесение пары  (l_B)', color=C_IMG)
    ax.set_title('Пара уровня n = 1 у нижнего края', color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig('figs/fig8_image_profile.png')
    plt.close(fig)


def fig_B():
    ref = load('ref')
    img = load('img_bottom')
    imgc = load('img_bottom_clean')
    mask = (ref['phi'], pair_vs(Params(pot=POT), ref['phi'], [EF] * len(ref['phi']))[0])
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    panel_set(axs.ravel(), [
        (ref, 'стандартный край, U = 1 t', REF, '-'),
        (imgc, 'изображение, без беспорядка', C_IMG, '-'),
        (img, f'изображение, U = 1 t (среднее по {img["nreal"]})', C_3, '-')],
        masks=[(mask, C_IMG)])
    axs[0, 0].legend(loc='lower right', fontsize=9)
    fig.suptitle('Потенциал изображения у нижнего края (d_edge = 18a), E_F = 0.4 t. '
                 'Заливка: B, при которых есть встречная пара', color=INK)
    fig.tight_layout()
    fig.savefig('figs/fig9_image_B_sweep.png')
    plt.close(fig)


def load_ef(kind, name):
    names = [f'{name}_s{s}' for s in SEEDS] if name in ('ref', 'img') else [name]
    ds = [np.load(f'data/{kind}_{n}.npz') for n in names]
    if kind == 'ef2t':
        return dict(EF=ds[0]['EF'], G=np.mean([d['G'] for d in ds], axis=0),
                    N=ds[0]['N'], nreal=len(ds))
    V = np.mean([d['V'] for d in ds], axis=0)
    T = np.mean([d['T'] for d in ds], axis=0)
    return dict(EF=ds[0]['EF'], nreal=len(ds), Rxy_L=V[:, 5] - V[:, 1],
                Rxy_R=V[:, 4] - V[:, 2], Rxx_bot=V[:, 1] - V[:, 2],
                Rxx_top=V[:, 5] - V[:, 4],
                tb=(T[:, 3, 2] + T[:, 2, 1] + T[:, 1, 0]) / 3)


def fig_EF():
    prm = Params(pot=POT)
    ex, sep, ov = pair_vs(prm, [PHI_EF] * len(EFS), EFS)
    series = [('ref', 'стандартный край', REF, '-'),
              ('img_clean', 'изображение, без беспорядка', C_IMG, '-'),
              ('img', 'изображение, U = 1 t', C_3, '-')]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    def decorate(ax):
        shade(ax, EFS * 1e-3, ex, C_IMG)          # shade() multiplies by 1e3
        shade(ax, EFS * 1e-3, ex & (ov > 0.3), C_NARROW, alpha=0.18)
        ax.set_xlabel('E_F / t')
        ax.set_xlim(EFS[0], EFS[-1])

    ax = axs[0, 0]
    decorate(ax)
    d = load_ef('ef2t', 'ref_clean')
    ax.plot(d['EF'], d['G'], color=REF, ls=':', label='стандартный край, без беспорядка')
    for name, lab, col, ls in series:
        d = load_ef('ef2t', name)
        ax.plot(d['EF'], d['G'], color=col, ls=ls, label=lab)
    for n in (1, 2, 3):
        ax.axhline(n, color=GRID, lw=1.0, zorder=0)
    ax.set_ylabel('G  (e²/h)')
    ax.set_ylim(0, 3.2)
    ax.set_title('Двухтерминальный кондактанс G(E_F) (протокол статьи)',
                 color=INK, fontsize=10)
    ax.legend(fontsize=8, loc='upper left')

    ax = axs[0, 1]
    decorate(ax)
    for name, lab, col, ls in series:
        d = load_ef('efhb', name)
        ax.plot(d['EF'], d['tb'], color=col, ls=ls, label=lab)
    ax.set_ylabel('t_b  (среднее по трём сегментам)')
    ax.set_ylim(-0.05, 1.1)
    ax.set_title('Обратное прохождение по сегментам нижнего края мостика',
                 color=INK, fontsize=10)

    for ax, keys, ylab in [(axs[1, 0], [('Rxy_L', '-'), ('Rxy_R', '--')], 'R_xy  (h/e²)'),
                           (axs[1, 1], [('Rxx_bot', '-'), ('Rxx_top', '--')], 'R_xx  (h/e²)')]:
        decorate(ax)
        for name, lab, col, _ in series:
            d = load_ef('efhb', name)
            for k, ls in keys:
                ax.plot(d['EF'], d[k], color=col, ls=ls)
        ax.set_ylabel(ylab)
    for n in (1, 2, 3):
        axs[1, 0].axhline(1 / n, color=GRID, lw=1.0, zorder=0)
    axs[1, 0].set_ylim(0, 1.15)
    axs[1, 0].set_title('Холловский мостик: R_xy(1–5) сплошные, R_xy(2–4) пунктир',
                        color=INK, fontsize=10)
    axs[1, 1].set_ylim(-0.02, 0.45)
    axs[1, 1].set_title('R_xx: нижний край сплошные, верхний пунктир',
                        color=INK, fontsize=10)
    fig.suptitle(f'Развёртка по E_F при φ = {PHI_EF} (ν = 1 → 2). Светлая заливка: '
                 'есть встречная пара; тёмная: пара перекрывается (> 0.3)', color=INK)
    fig.tight_layout()
    fig.savefig('figs/fig10_image_EF_sweep.png')
    plt.close(fig)


if __name__ == '__main__':
    fig_profile()
    fig_B()
    fig_EF()
