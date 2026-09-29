"""Russian -> English for the text in the figures (used by plots_en.py)."""
TR = {
    'стандартный край': 'standard edge',
    'широкий карман (пара разнесена)': 'wide pocket (pair separated)',
    'узкий карман (пара перекрывается)': 'narrow pocket (pair overlaps)',
    'E / t  (уровни Ландау E_n(⟨y⟩))': 'E / t  (Landau levels E_n(⟨y⟩))',
    'расстояние от края y (узлы решётки)': 'distance from the edge y (lattice sites)',
    '|ψ|² мод на E_F': '|ψ|² of the modes at E_F',
    'разнесение пары 5.8 l_B\nперекрытие ∫|ψ₁||ψ₂| = 0.00':
        'pair separation 5.8 l_B\noverlap ∫|ψ₁||ψ₂| = 0.00',
    'разнесение пары 1.6 l_B\nперекрытие ∫|ψ₁||ψ₂| = 0.57':
        'pair separation 1.6 l_B\noverlap ∫|ψ₁||ψ₂| = 0.57',
    'Край образца при ν=1 (φ=0.028, l_B=2.4a): стрелки: ↓ киральные каналы '
    '(к источнику), ↑ встречный канал':
        'Sample edge at ν=1 (φ=0.028, l_B=2.4a); arrows: ↓ chiral channels '
        '(towards the source), ↑ counter-propagating channel',
    'стандартный край, U = 1 t': 'standard edge, U = 1 t',
    'широкий карман, без беспорядка': 'wide pocket, clean',
    'широкий карман, U = 1 t': 'wide pocket, U = 1 t',
    'узкий карман, U = 1 t': 'narrow pocket, U = 1 t',
    'Плотность тока электронов, выпущенных зондом 2 (ν = 1, φ = 0.028).\n'
    'Киральный путь 2 → 1 по нижнему краю; встречный канал уносит часть потока 2 → 3':
        'Current density of electrons injected from probe 2 (ν = 1, φ = 0.028).\n'
        'Chiral path 2 → 1 along the bottom edge; the counter-propagating '
        'channel carries part of the flow 2 → 3',
    'B  (поток на плакетку φ × 10³)': 'B  (flux per plaquette φ × 10³)',
    'R_xy, левая пара зондов (1–5)': 'R_xy, left probe pair (1–5)',
    'широкий карман (пара не перекрывается)': 'wide pocket (pair does not overlap)',
    'R_xy, правая пара зондов (2–4)': 'R_xy, right probe pair (2–4)',
    'R_xx вдоль нижнего края (1–2)': 'R_xx along the bottom edge (1–2)',
    'R_xx вдоль верхнего края (5–4)': 'R_xx along the top edge (5–4)',
    'Карман только на нижнем крае, беспорядок U = 1 t (среднее по 3 реализациям). '
    'Заливка: B, при которых у нижнего края есть встречная пара (широкий карман)':
        'Pocket on the bottom edge only, disorder U = 1 t (average over 3 '
        'realizations). Shading: fields with a counter-propagating pair at the '
        'bottom edge (wide pocket)',
    'стандартный край, без беспорядка': 'standard edge, clean',
    'узкий карман, без беспорядка': 'narrow pocket, clean',
    'Без беспорядка встречный канал баллистический (t_b = 1) и квантование '
    'нарушено; обратное рассеяние (узкий карман + беспорядок) его восстанавливает':
        'Without disorder the counter-propagating channel is ballistic (t_b = 1) '
        'and quantization is broken; backscattering (narrow pocket + disorder) '
        'restores it',
    'широкий карман на обоих краях': 'wide pocket on both edges',
    'узкий карман на обоих краях': 'narrow pocket on both edges',
    'широкий карман только между зондами (петля)':
        'wide pocket only between the probes (closed loop)',
    'Карманы на обоих краях и локальный карман, не доходящий до контактов (U = 1 t)':
        'Pockets on both edges, and a local pocket that does not reach the '
        'contacts (U = 1 t)',
    't_b  (2 → 1 против киральности)\n(>1: две встречные пары)':
        't_b  (2 → 1 against the chirality)\n(>1: two counter-propagating pairs)',
    'Обратное прохождение по нижнему краю между зондами 1 и 2':
        'Backward transmission along the bottom edge between probes 1 and 2',
    'широкий, без беспорядка': 'wide, clean',
    'широкий, U = 1 t': 'wide, U = 1 t',
    'узкий, без беспорядка': 'narrow, clean',
    'узкий, U = 1 t': 'narrow, U = 1 t',
    't_b  (среднее по сегментам нижнего края)': 't_b  (mean over bottom-edge segments)',
    'Ландауэр–Бюттикер (линии) и Kwant (точки, R_xy(1–5))':
        'Landauer–Büttiker (lines) and Kwant (points, R_xy(1–5))',
    'ν=1: R_xx(низ)·ν': 'ν=1: R_xx(bottom)·ν',
    'ν=2: R_xx(низ)·ν': 'ν=2: R_xx(bottom)·ν',
    'Kwant, без беспорядка': 'Kwant, clean',
    'разнесение пары / l_B': 'pair separation / l_B',
    'перекрытие пары ∫|ψ₁||ψ₂|': 'pair overlap ∫|ψ₁||ψ₂|',
    'Перекрытие встречной пары': 'Overlap of the counter-propagating pair',
    'Обратное прохождение по краю': 'Backward transmission along the edge',
    'сегмент 1–2 (между зондами, 68 a)': 'segment 1–2 (between the probes, 68 a)',
    'сегмент 2–3 (зонд 2 – сток, 60 a)': 'segment 2–3 (probe 2 – drain, 60 a)',
    'Сопротивления при ν = 1': 'Resistances at ν = 1',
    'R_xx (низ)': 'R_xx (bottom)',
    'Зависимость от ширины кармана (φ=0.028, V_dip=0.35 t, U=1.0 t, '
    '3 реализации беспорядка)':
        'Dependence on the pocket width (φ=0.028, V_dip=0.35 t, U=1.0 t, '
        '3 disorder realizations)',
    'расстояние от края  (a)': 'distance from the edge  (a)',
    'Профиль потенциала у края': 'Edge potential profile',
    'широкий карман': 'wide pocket',
    'узкий карман': 'narrow pocket',
    'изображение, d_edge = 18a ≈ 8 l_B': 'image charge, d_edge = 18a ≈ 8 l_B',
    'Уровни Ландау у края, φ = 0.03 (l_B = 2.3a)':
        'Landau levels near the edge, φ = 0.03 (l_B = 2.3a)',
    'разнесение пары  (l_B)': 'pair separation  (l_B)',
    'Пара уровня n = 1 у нижнего края': 'n = 1 pair at the bottom edge',
    'перекрытие': 'overlap',
    'изображение, без беспорядка': 'image pocket, clean',
    'изображение, U = 1 t (среднее по 3)': 'image pocket, U = 1 t (average of 3)',
    'Потенциал изображения у нижнего края (d_edge = 18a), E_F = 0.4 t. '
    'Заливка: B, при которых есть встречная пара':
        'Image-charge potential at the bottom edge (d_edge = 18a), E_F = 0.4 t. '
        'Shading: fields with a counter-propagating pair',
    'Двухтерминальный кондактанс G(E_F) (протокол статьи)':
        'Two-terminal conductance G(E_F) (protocol of the paper)',
    'изображение, U = 1 t': 'image pocket, U = 1 t',
    't_b  (среднее по трём сегментам)': 't_b  (mean over the three segments)',
    'Обратное прохождение по сегментам нижнего края мостика':
        'Backward transmission along the bottom-edge segments of the Hall bar',
    'Холловский мостик: R_xy(1–5) сплошные, R_xy(2–4) пунктир':
        'Hall bar: R_xy(1–5) solid, R_xy(2–4) dashed',
    'R_xx: нижний край сплошные, верхний пунктир':
        'R_xx: bottom edge solid, top edge dashed',
    'Развёртка по E_F при φ = 0.03 (ν = 1 → 2). Светлая заливка: есть встречная '
    'пара; тёмная: пара перекрывается (> 0.3)':
        'E_F sweep at φ = 0.03 (ν = 1 → 2). Light shading: counter-propagating '
        'pair present; dark: the pair overlaps (> 0.3)',
}
