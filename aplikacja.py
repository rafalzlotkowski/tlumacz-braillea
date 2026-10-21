#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dekoder polskiego Braille'a ze zdjęcia."""
import argparse
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.spatial import KDTree

PREVIEW_WIDTH = 800
MAX_WORK_SIDE = 2400
MAX_SKEW_DEG = 25
CELL_PITCH_RANGE = (2.2, 3.8)
WORD_GAP_FACTOR = 4.0
LINE_GAP_FACTOR = 1.5
ROW_GAP_FACTOR = 0.5
BG_DOWNSCALE = 0.25
UNKNOWN_CHAR = "*"
SPLIT_BLOBS = True
DEPTH_FRAC = 0.5


def cell(dots: str) -> tuple:
    """cell('145') -> (1,0,0,1,1,0). Kolejność: punkty 1..6."""
    return tuple(1 if str(i) in dots else 0 for i in range(1, 7))


_LETTERS = {
    "a": "1", "b": "12", "c": "14", "d": "145", "e": "15", "f": "124",
    "g": "1245", "h": "125", "i": "24", "j": "245", "k": "13", "l": "123",
    "m": "134", "n": "1345", "o": "135", "p": "1234", "q": "12345",
    "r": "1235", "s": "234", "t": "2345", "u": "136", "v": "1236",
    "w": "2456", "x": "1346", "y": "13456", "z": "1356",
    "ą": "16", "ć": "146", "ę": "156", "ł": "126", "ń": "1456",
    "ó": "346", "ś": "246", "ź": "2346", "ż": "12346",
}
_PUNCT = {
    ",": "2", ";": "23", ":": "25", ".": "256", "?": "26", "!": "235",
    "-": "36",
}

CHAR_TO_CELL = {ch: cell(v) for ch, v in {**_LETTERS, **_PUNCT}.items()}
CHAR_TO_CELL[" "] = cell("")
BRAILLE_DICT = {pat: ch for ch, pat in CHAR_TO_CELL.items()}
assert len(BRAILLE_DICT) == len(CHAR_TO_CELL), "Powtórzony wzorzec w tabeli!"

CAPITAL_SIGN = cell("6")
NUMBER_SIGN = cell("3456")
SIGN_LABELS = {CAPITAL_SIGN: "^", NUMBER_SIGN: "#"}

BRAILLE_DIGITS = {"a": "1", "b": "2", "c": "3", "d": "4", "e": "5",
                  "f": "6", "g": "7", "h": "8", "i": "9", "j": "0"}
DIGIT_TO_LETTER = {v: k for k, v in BRAILLE_DIGITS.items()}

_FOLD = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ")


def select_roi(img):
    """Wybór ROI na podglądzie, ale wycinek z ORYGINAŁU (pełna rozdzielczość)."""
    scale = PREVIEW_WIDTH / img.shape[1]
    preview = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    title = "Zaznacz tekst Braille'a i nacisnij ENTER"
    x, y, w, h = [int(v) for v in cv2.selectROI(
        title, preview, showCrosshair=True, fromCenter=False)]
    cv2.destroyWindow(title)
    if w == 0 or h == 0:
        return img
    return img[int(y / scale):int((y + h) / scale), int(x / scale):int((x + w) / scale)]


def preprocess(image):
    """Normalizacja oświetlenia (tło liczone na zmniejszonym obrazie) + próg względem szumu tła."""
    h, w = image.shape[:2]
    base = np.sqrt(h * w)
    blur_sz = max(int(base * 0.012), 3)
    blur_sz += 1 - blur_sz % 2
    morph_sz = max(2, int(base * 0.004))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    s = BG_DOWNSCALE if min(h, w) >= 200 else 1.0
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    small = cv2.GaussianBlur(small, (0, 0), base * 0.08 * s)
    background = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR) if s < 1 else small
    flat = cv2.divide(gray, background, scale=255)
    flat = cv2.GaussianBlur(flat, (blur_sz, blur_sz), 0)

    med = float(np.median(flat))
    mad = float(np.median(np.abs(flat - med))) * 1.4826
    noise_thr = med - max(8.0, 4.0 * mad)
    core = float(np.percentile(flat, 2))
    half_thr = med - DEPTH_FRAC * (med - core)
    mask = (flat < min(noise_thr, half_thr)).astype(np.uint8) * 255

    kernel = np.ones((morph_sz, morph_sz), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _split_blob(cnt, k):
    """Rozdziela zlaną plamę na k kropek (k-means na jej pikselach)."""
    x, y, w, h = cv2.boundingRect(cnt)
    blob = np.zeros((h, w), np.uint8)
    cv2.drawContours(blob, [cnt], -1, 255, -1, offset=(-x, -y))
    py, px = np.nonzero(blob)
    pts = np.column_stack([px, py]).astype(np.float32)
    if len(pts) < k:
        return []
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    _, _, centers = cv2.kmeans(pts, k, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    return [(cx + x, cy + y) for cx, cy in centers]


def detect_dots(mask):
    """Zwraca tablicę (N, 3): x, y (float), pole. Zlane kropki są rozdzielane."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    info = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perim = cv2.arcLength(cnt, True)
        if area < 4 or perim == 0:
            continue
        info.append((cnt, area, 4 * np.pi * area / perim ** 2))
    if not info:
        return np.empty((0, 3))

    round_areas = [a for _, a, c in info if c > 0.7]
    ref = float(np.median(round_areas if round_areas else [a for _, a, _ in info]))

    dots = []
    for cnt, area, circ in info:
        if area < 0.35 * ref:
            continue
        if not SPLIT_BLOBS:
            if area <= 4 * ref and circ > 0.2:
                m = cv2.moments(cnt)
                if m["m00"] > 0:
                    dots.append([m["m10"] / m["m00"], m["m01"] / m["m00"], area])
            continue
        if area <= 1.7 * ref:
            if circ < 0.4 or circ > 1.3:
                continue
            m = cv2.moments(cnt)
            if m["m00"] > 0:
                dots.append([m["m10"] / m["m00"], m["m01"] / m["m00"], area])
        else:
            k = int(round(area / ref))
            x, y, w, h = cv2.boundingRect(cnt)
            at_border = x <= 0 or y <= 0 or x + w >= mask.shape[1] or y + h >= mask.shape[0]
            if 2 <= k <= 8 and not at_border:
                dots += [[cx, cy, area / k] for cx, cy in _split_blob(cnt, k)]
    return np.array(dots) if dots else np.empty((0, 3))


def nn_distance(xy):
    """Typowa odległość do najbliższego sąsiada (mediana, bez górnych 20%)."""
    if len(xy) < 3:
        return 20.0
    dist, _ = KDTree(xy).query(xy, k=2)
    nn = dist[:, 1]
    trimmed = nn[nn <= np.percentile(nn, 80)]
    return float(np.median(trimmed if len(trimmed) else nn))


def _row_score(xy, angle_deg, bin_w):
    a = np.radians(angle_deg)
    y = -np.sin(a) * xy[:, 0] + np.cos(a) * xy[:, 1]
    bins = np.floor((y - y.min()) / bin_w).astype(int)
    return float(np.sum(np.bincount(bins).astype(float) ** 2))


def estimate_skew(xy, d, max_deg=MAX_SKEW_DEG):
    """Kąt (stopnie) dla cv2.getRotationMatrix2D, który prostuje rzędy kropek."""
    bin_w = d / 3.0
    coarse = np.arange(-max_deg, max_deg + 1e-9, 1.0)
    best = coarse[int(np.argmax([_row_score(xy, a, bin_w) for a in coarse]))]
    fine = np.arange(best - 1.0, best + 1.0 + 1e-9, 0.1)
    best = fine[int(np.argmax([_row_score(xy, a, bin_w) for a in fine]))]
    finer = np.arange(best - 0.1, best + 0.1 + 1e-9, 0.02)
    return float(finer[int(np.argmax([_row_score(xy, a, bin_w) for a in finer]))])


def rotate_keep_all(image, xy, angle, warp=True):
    """Obrót z powiększeniem płótna + transformacja punktów; warp=False pomija obrót obrazu."""
    h, w = image.shape[:2]
    c = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(c, angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2.0 - c[0]
    M[1, 2] += nh / 2.0 - c[1]
    rotated = (cv2.warpAffine(image, M, (nw, nh), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE) if warp else None)
    pts = cv2.transform(xy.reshape(-1, 1, 2).astype(np.float64), M).reshape(-1, 2)
    return rotated, pts


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


def fit_cell_pitch(words_xs, d, dx):
    """Rozstaw komórek P wspólny dla całego ROI (przeszukiwanie zgrubne, potem dokładne)."""
    cap = 0.35 * dx
    multi = [xs for xs in words_xs if np.ptp(xs) > 0.5 * dx]
    if not multi:
        return 2.4 * d
    lo, hi = CELL_PITCH_RANGE[0] * d, CELL_PITCH_RANGE[1] * d
    coarse = np.linspace(lo, hi, 17)
    best = float(coarse[int(np.argmin([_pitch_cost(multi, P, dx, cap) for P in coarse]))])
    step = coarse[1] - coarse[0]
    fine = np.linspace(max(lo, best - step), min(hi, best + step), 17)
    return float(fine[int(np.argmin([_pitch_cost(multi, P, dx, cap) for P in fine]))])


def decode_cells(patterns):
    """Automat stanów. Zwraca listę tekstów (po jednym na komórkę; '' dla znaków sterujących)."""
    out = []
    upper_next = upper_word = number = False
    for pat in patterns:
        if pat == CAPITAL_SIGN:
            if upper_next:
                upper_word = True
            upper_next = True
            out.append("")
            continue
        if pat == NUMBER_SIGN:
            number = True
            out.append("")
            continue
        ch = BRAILLE_DICT.get(pat, UNKNOWN_CHAR)
        if ch == " ":
            number = upper_next = upper_word = False
            out.append(" ")
            continue
        if number:
            if ch in BRAILLE_DIGITS:
                ch = BRAILLE_DIGITS[ch]
            elif ch not in ",.":
                number = False
        if upper_next or upper_word:
            ch = ch.upper()
            upper_next = False
        out.append(ch)
    return out


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


def decode_lattice(xy, d0):
    """Dopasowanie siatki do (już wyprostowanych) kropek i dekodowanie."""
    groups = split_by_gap(xy[:, 1], LINE_GAP_FACTOR * d0)
    centers = []
    for g in groups:
        ys = xy[g, 1]
        centers.append([float(np.median(ys[r]))
                        for r in split_by_gap(ys, ROW_GAP_FACTOR * d0)])

    gaps = [np.diff(c) for c in centers if len(c) == 3]
    gaps = np.concatenate(gaps) if gaps else np.array([])
    gaps = gaps[(gaps > 0.6 * d0) & (gaps < 1.6 * d0)]
    d_y = float(np.median(gaps)) if len(gaps) else d0
    dx = d_y

    tops = resolve_line_tops(centers, d_y)

    rough = [[xy[g[w], 0] for w in split_by_gap(xy[g, 0], WORD_GAP_FACTOR * d_y)]
             for g in groups]
    P = fit_cell_pitch([xs for words in rough for xs in words], d_y, dx)

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
                cells.append(Cell(cell(""), (0, 0, 0, 0)))
            for i in range(int(k.max()) + 1):
                xl = x0 + i * P
                cells.append(Cell(tuple(grids.get(i, [0] * 6)),
                                  (xl - pad, top - pad, xl + dx + pad, top + 2 * d_y + pad)))
        for cl, ch in zip(cells, decode_cells([c_.pattern for c_ in cells])):
            cl.char = ch
        result.lines.append(cells)

    result.text = "\n".join("".join(c.char for c in line).rstrip() for line in result.lines)
    return result


def process(crop):
    """Obraz (BGR) -> (Result, obraz roboczy, punkty po obrocie)."""
    work = crop
    s = MAX_WORK_SIDE / max(crop.shape[:2])
    if s < 1:
        work = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    dots = detect_dots(preprocess(work))
    if len(dots) < 6:
        raise ValueError(f"Wykryto za mało kropek ({len(dots)}).")
    xy = dots[:, :2]
    d0 = nn_distance(xy)
    angle = estimate_skew(xy, d0)
    xy_rot = rotate_keep_all(work, xy, angle, warp=False)[1]
    result = decode_lattice(xy_rot, d0)
    result.angle = angle
    return result, work, xy_rot


def visualize(work, xy_rot, result):
    """Rysuje kropki i ramki komórek na wyprostowanym obrazie."""
    out = rotate_keep_all(work, xy_rot, result.angle)[0]
    for x, y in xy_rot:
        cv2.circle(out, (int(round(x)), int(round(y))), 3, (0, 255, 0), -1)
    for line in result.lines:
        for c in line:
            if not any(c.pattern):
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in c.bbox]
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = (c.char or SIGN_LABELS.get(c.pattern, "")).translate(_FOLD)
            cv2.putText(out, label, (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 1, cv2.LINE_AA)
    return out


def encode_text(text):
    cells, num = [], False
    for ch in text:
        if ch.isdigit():
            if not num:
                cells.append(NUMBER_SIGN)
                num = True
            cells.append(CHAR_TO_CELL[DIGIT_TO_LETTER[ch]])
            continue
        if ch not in ",.":
            num = False
        if ch.isupper():
            cells.append(CAPITAL_SIGN)
            ch = ch.lower()
        cells.append(CHAR_TO_CELL[ch])
    return cells


def render_synthetic(text, d=14, angle=0.0, jitter=0.5, seed=0,
                     cell_ratio=2.4, space_ratio=1.0):
    """cell_ratio: rozstaw komórek / rozstaw kropek; space_ratio: szerokość spacji / komórki."""
    rng = np.random.default_rng(seed)
    lines = text.split("\n")
    P, LP, margin = cell_ratio * d, 4.0 * d, 4 * d
    enc = [encode_text(line) for line in lines]
    blank = cell("")

    def advances(e):
        xs, cur = [], 0.0
        for pat in e:
            xs.append(cur)
            cur += space_ratio * P if pat == blank else P
        return xs, cur

    width = max(advances(e)[1] for e in enc)
    W = int(2 * margin + width)
    H = int(2 * margin + len(lines) * LP)
    img = np.full((H, W), 205, np.uint8)
    for li, e in enumerate(enc):
        xs, _ = advances(e)
        for ci, pat in enumerate(e):
            for pos, on in enumerate(pat):
                if on:
                    r, c = pos % 3, pos // 3
                    x = margin + xs[ci] + c * d + rng.normal(0, jitter)
                    y = margin + li * LP + r * d + rng.normal(0, jitter)
                    cv2.circle(img, (int(round(x)), int(round(y))), int(d * 0.25),
                               70, -1, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (5, 5), 0).astype(np.float32)
    img *= np.linspace(0.65, 1.0, W, dtype=np.float32)[None, :]
    img += rng.normal(0, 3, img.shape).astype(np.float32)
    img = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if angle:
        img = rotate_keep_all(img, np.zeros((1, 2)), angle)[0]
    return img


def selftest():
    text = "Ala ma kota.\nŻółć 123, gęś!\nz ń ą\nb"
    text2 = "ala ma kota a kot ma\nal"
    cases = []
    for angle in (0, 4, -9, 17):
        for seed in (0, 1):
            cases.append((text, dict(angle=angle, seed=seed)))
    for angle in (0, 5, -7):
        for seed in (0, 1):
            cases.append((text2, dict(angle=angle, seed=seed, d=22,
                                      cell_ratio=3.25, space_ratio=0.79)))
            cases.append((text, dict(angle=angle, seed=seed, cell_ratio=3.0, space_ratio=0.8)))
    ok = True
    for expected, kw in cases:
        img = render_synthetic(expected, **kw)
        try:
            got = process(img)[0].text
        except Exception as e:  # noqa: BLE001
            got = f"<błąd: {e}>"
        good = got == expected
        ok &= good
        print(f"{'OK  ' if good else 'BŁĄD'} {kw}")
        if not good:
            print("  oczekiwano:", repr(expected))
            print("  otrzymano :", repr(got))
    print("WSZYSTKO OK" if ok else "SĄ BŁĘDY")
    return 0 if ok else 1


def dots_of(pattern):
    return "".join(str(i + 1) for i, v in enumerate(pattern) if v) or "_"


def main():
    global SPLIT_BLOBS, DEPTH_FRAC
    ap = argparse.ArgumentParser(description="Dekoder polskiego Braille'a ze zdjęcia")
    ap.add_argument("image", nargs="?", help="ścieżka do zdjęcia")
    ap.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                    help="ROI w pikselach oryginału (pomija okno wyboru)")
    ap.add_argument("--save", help="zapisz wizualizację do pliku zamiast pokazywać okno")
    ap.add_argument("--debug", action="store_true",
                    help="wypisz każdą komórkę jako numery punktów i zapisz maskę")
    ap.add_argument("--no-split", action="store_true", help="nie rozdzielaj dużych plam")
    ap.add_argument("--depth", type=float, default=DEPTH_FRAC,
                    help="głębokość progu 0.2-0.8 (mniej = luźniej, więcej = ostrzej)")
    ap.add_argument("--selftest", action="store_true", help="test na danych syntetycznych")
    args = ap.parse_args()

    SPLIT_BLOBS = not args.no_split
    DEPTH_FRAC = args.depth
    if args.selftest:
        sys.exit(selftest())
    if not args.image:
        ap.error("podaj ścieżkę do zdjęcia albo użyj --selftest")

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Nie można wczytać obrazu: {args.image}")

    if args.roi:
        x, y, w, h = args.roi
        crop = img[y:y + h, x:x + w]
    else:
        crop = select_roi(img)

    try:
        result, work, xy = process(crop)
    except ValueError as e:
        sys.exit(str(e))

    print(f"Kąt prostowania: {result.angle:.2f}°")
    print(f"Rozstaw kropek: {result.d:.2f} px, stosunek komórka/kropka: "
          f"{result.cell_pitch_ratio:.2f} (standard ~2.4)")
    print(f"Linii: {len(result.lines)}, komórek: {sum(len(l) for l in result.lines)}")
    print(f"\nZdekodowany tekst:\n{result.text}\n")

    if args.debug:
        print(f"Wykryte kropki: {len(xy)}  (kąt {result.angle:.2f}°)")
        for i, line in enumerate(result.lines):
            print(f"linia {i}: " + " ".join(dots_of(c.pattern) for c in line))
        base = (args.save or "debug.png").rsplit(".", 1)[0]
        cv2.imwrite(base + "_mask.png", preprocess(work))
        print(f"Zapisano maskę: {base}_mask.png")

    vis = visualize(work, xy, result)
    if args.save:
        cv2.imwrite(args.save, vis)
        print(f"Zapisano: {args.save}")
    else:
        cv2.imshow("Wykryte znaki Braille'a (po wyprostowaniu)", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()