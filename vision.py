"""Przetwarzanie obrazu: wykrywanie kropek, rozstaw, kąt pochylenia, obrót."""
import cv2
import numpy as np
from scipy.spatial import KDTree

from config import Config


def preprocess(image, cfg: Config):
    """Normalizacja oświetlenia (tło liczone na zmniejszonym obrazie) + próg względem szumu tła."""
    h, w = image.shape[:2]
    base = np.sqrt(h * w)
    blur_sz = max(int(base * 0.012), 3)
    blur_sz += 1 - blur_sz % 2
    morph_sz = max(2, int(base * 0.004))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    s = cfg.bg_downscale if min(h, w) >= 200 else 1.0
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    small = cv2.GaussianBlur(small, (0, 0), base * 0.08 * s)
    background = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR) if s < 1 else small
    flat = cv2.divide(gray, background, scale=255)
    flat = cv2.GaussianBlur(flat, (blur_sz, blur_sz), 0)

    med = float(np.median(flat))
    mad = float(np.median(np.abs(flat - med))) * 1.4826
    noise_thr = med - max(8.0, 4.0 * mad)
    core = float(np.percentile(flat, 2))
    half_thr = med - cfg.depth_frac * (med - core)
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


def detect_dots(mask, cfg: Config):
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
        if not cfg.split_blobs:
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


def estimate_skew(xy, d, cfg: Config):
    """Kąt (stopnie) dla cv2.getRotationMatrix2D, który prostuje rzędy kropek."""
    bin_w = d / 3.0
    coarse = np.arange(-cfg.max_skew_deg, cfg.max_skew_deg + 1e-9, 1.0)
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
