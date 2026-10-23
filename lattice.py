"""Geometria siatki: z wyprostowanych kropek (tablica xy) do komórek i tekstu. Bez OpenCV."""
from dataclasses import dataclass, field

import numpy as np

from config import Config
from letters import BLANK, decode_cells


@dataclass
class Cell:
    pattern: tuple
    bbox: tuple
    char: str = ""


@dataclass
class Result:
    text: str = ""
    lines: list = field(default_factory=list)
    d: float = 0.0
    cell_pitch_ratio: float = 0.0
    angle: float = 0.0


def split_by_gap(values, eps):
    """Grupy indeksów posortowanych wartości; cięcie tam, gdzie luka > eps."""
    values = np.asarray(values, float)
    if values.size == 0:
        return []
    order = np.argsort(values, kind="stable")
    breaks = np.where(np.diff(values[order]) > eps)[0] + 1
    return np.split(order, breaks)


def resolve_line_tops(centers_per_line, d_y):
    """Dla każdej linii tekstu wyznacza y górnego rzędu (punkty 1 i 4)."""
    full = [i for i, c in enumerate(centers_per_line)
            if len(c) == 3 and np.all(np.abs(np.diff(c) - d_y) < 0.25 * d_y)]
    full_tops = [centers_per_line[i][0] for i in full]

    adj = [centers_per_line[j][0] - centers_per_line[i][0]
           for i, j in zip(full, full[1:]) if j == i + 1]
    pitch = float(np.median(adj)) if adj else 4.0 * d_y

    tops = []
    for c in centers_per_line:
        c0 = c[0]
        if len(c) >= 3 or not full_tops:
            tops.append(c0)
            continue
        best = None
        for shift in range(3):
            t = c0 - shift * d_y
            if any(round((ci - t) / d_y) > 2 for ci in c):
                continue
            ref = min(full_tops, key=lambda r: abs(r - t))
            n = round((t - ref) / pitch)
            err = abs(t - (ref + n * pitch))
            if best is None or err < best[0]:
                best = (err, t)
        tops.append(best[1] if best else c0)
    return tops


def assign_columns(xs, x0, P, dx):
    """Numer komórki k i kolumna c (0/1) dla każdej kropki."""
    u = xs - x0
    k0 = np.round(u / P)
    k1 = np.round((u - dx) / P)
    e0 = np.abs(u - k0 * P)
    e1 = np.abs(u - dx - k1 * P)
    use1 = e1 < e0
    return np.where(use1, k1, k0).astype(int), use1.astype(int)


def _fit_from_anchor(xs, x0, P, dx, cap):
    """Doprecyzowuje fazę x0 (mediana reszt) i zwraca koszt dopasowania."""
    for _ in range(2):
        k, c = assign_columns(xs, x0, P, dx)
        shift = float(np.median(xs - (x0 + k * P + c * dx)))
        x0 += float(np.clip(shift, -0.35 * dx, 0.35 * dx))
    k, c = assign_columns(xs, x0, P, dx)
    res = np.abs(xs - (x0 + k * P + c * dx))
    return float((np.minimum(res, cap) ** 2).sum()), x0, k, c


def fit_word(xs, P, dx):
    """Siatka jednego wyrazu. Zwraca (x0, k, c); k liczone od 0."""
    cap = 0.35 * dx
    left = _fit_from_anchor(xs, float(xs.min()), P, dx, cap)
    right = _fit_from_anchor(xs, float(xs.min()) - dx, P, dx, cap)
    use_right = right[0] + 0.25 * cap ** 2 < left[0]
    _, x0, k, c = right if use_right else left
    kmin = int(k.min())
    return x0 + kmin * P, k - kmin, c


def _pitch_cost(multi, P, dx, cap):
    return sum(min(_fit_from_anchor(xs, float(xs.min()), P, dx, cap)[0],
                   _fit_from_anchor(xs, float(xs.min()) - dx, P, dx, cap)[0])
               for xs in multi)


def fit_cell_pitch(words_xs, d, dx, cfg: Config):
    """Rozstaw komórek P wspólny dla całego ROI (przeszukiwanie zgrubne, potem dokładne)."""
    cap = 0.35 * dx
    multi = [xs for xs in words_xs if np.ptp(xs) > 0.5 * dx]
    if not multi:
        return 2.4 * d
    lo, hi = cfg.cell_pitch_range[0] * d, cfg.cell_pitch_range[1] * d
    coarse = np.linspace(lo, hi, 17)
    best = float(coarse[int(np.argmin([_pitch_cost(multi, P, dx, cap) for P in coarse]))])
    step = coarse[1] - coarse[0]
    fine = np.linspace(max(lo, best - step), min(hi, best + step), 17)
    return float(fine[int(np.argmin([_pitch_cost(multi, P, dx, cap) for P in fine]))])


def decode_lattice(xy, d0, cfg: Config):
    """Dopasowanie siatki do (już wyprostowanych) kropek i dekodowanie."""
    groups = split_by_gap(xy[:, 1], cfg.line_gap_factor * d0)
    centers = []
    for g in groups:
        ys = xy[g, 1]
        centers.append([float(np.median(ys[r]))
                        for r in split_by_gap(ys, cfg.row_gap_factor * d0)])

    gaps = [np.diff(c) for c in centers if len(c) == 3]
    gaps = np.concatenate(gaps) if gaps else np.array([])
    gaps = gaps[(gaps > 0.6 * d0) & (gaps < 1.6 * d0)]
    d_y = float(np.median(gaps)) if len(gaps) else d0
    dx = d_y

    tops = resolve_line_tops(centers, d_y)

    rough = [[xy[g[w], 0] for w in split_by_gap(xy[g, 0], cfg.word_gap_factor * d_y)]
             for g in groups]
    P = fit_cell_pitch([xs for words in rough for xs in words], d_y, dx, cfg)

    word_gap = 1.5 * P - 0.5 * dx
    words_per_line = [[g[w] for w in split_by_gap(xy[g, 0], word_gap)] for g in groups]

    result = Result(d=d_y, cell_pitch_ratio=P / d_y)
    pad = 0.5 * d_y
    for words, top in zip(words_per_line, tops):
        cells = []
        for wi, w in enumerate(words):
            xs, ys = xy[w, 0], xy[w, 1]
            x0, k, c = fit_word(xs, P, dx)
            r = np.clip(np.round((ys - top) / d_y), 0, 2).astype(int)
            grids = {}
            for ki, ci, ri in zip(k, c, r):
                grids.setdefault(int(ki), [0] * 6)[int(ri + 3 * ci)] = 1
            if wi > 0:
                cells.append(Cell(BLANK, (0, 0, 0, 0)))
            for i in range(int(k.max()) + 1):
                xl = x0 + i * P
                cells.append(Cell(tuple(grids.get(i, [0] * 6)),
                                  (xl - pad, top - pad, xl + dx + pad, top + 2 * d_y + pad)))
        for cl, ch in zip(cells, decode_cells([c_.pattern for c_ in cells])):
            cl.char = ch
        result.lines.append(cells)

    result.text = "\n".join("".join(c.char for c in line).rstrip() for line in result.lines)
    return result
