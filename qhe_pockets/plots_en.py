"""English versions of all figures -> figs/en/*.png

Runs the figure functions of plots.py and plots_image.py unchanged and
translates every text element right before saving. A Cyrillic string
without a translation raises an error, so nothing Russian slips through.

    python plots_en.py            # write figs/en/
    COLLECT=1 python plots_en.py  # only list untranslated strings
"""
import os
import re

import matplotlib
from matplotlib.figure import Figure

import plots
import plots_image
from translations import TR

CYR = re.compile('[А-Яа-яЁё]')
COLLECT = bool(os.environ.get('COLLECT'))
MISSING = []
_savefig = Figure.savefig


def _translated_savefig(self, fname, *args, **kwargs):
    missing = []
    for t in self.findobj(matplotlib.text.Text):
        s = t.get_text()
        if CYR.search(s):
            if s in TR:
                t.set_text(TR[s])
            else:
                missing.append(s)
    if COLLECT:
        MISSING.extend(m for m in missing if m not in MISSING)
        return
    if missing:
        raise KeyError(f'untranslated in {fname}: {missing}')
    os.makedirs('figs/en', exist_ok=True)
    _savefig(self, os.path.join('figs/en', os.path.basename(fname)), *args, **kwargs)


Figure.savefig = _translated_savefig


def main():
    plots.fig_edge_structure()
    plots.fig_wavefunctions()
    plots.fig_sweeps()
    plots.fig_width()
    plots_image.fig_profile()
    plots_image.fig_B()
    plots_image.fig_EF()
    if COLLECT:
        for m in MISSING:
            print(repr(m))


if __name__ == '__main__':
    main()
