"""Dekodowanie zdjęcia: wybór ROI, przetwarzanie, wizualizacja."""
import cv2

from config import Config
from lattice import decode_lattice
from letters import SIGN_LABELS
from vision import detect_dots, estimate_skew, nn_distance, preprocess, rotate_keep_all

_FOLD = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ")


def select_roi(img, cfg: Config = None):
    """Wybór ROI na podglądzie, ale wycinek z ORYGINAŁU ."""
    cfg = cfg or Config()
    scale = cfg.preview_width / img.shape[1]
    preview = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    title = "Zaznacz tekst Braille'a i nacisnij ENTER"
    x, y, w, h = [int(v) for v in cv2.selectROI(
        title, preview, showCrosshair=True, fromCenter=False)]
    cv2.destroyWindow(title)
    if w == 0 or h == 0:
        return img
    return img[int(y / scale):int((y + h) / scale), int(x / scale):int((x + w) / scale)]


def process(crop, cfg: Config = None):
    """Obraz (BGR) -> (Result, obraz roboczy, punkty po obrocie)."""
    cfg = cfg or Config()
    work = crop
    s = cfg.max_work_side / max(crop.shape[:2])
    if s < 1:
        work = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    dots = detect_dots(preprocess(work, cfg), cfg)
    if len(dots) < 6:
        raise ValueError(f"Wykryto za mało kropek ({len(dots)}).")
    xy = dots[:, :2]
    d0 = nn_distance(xy)
    angle = estimate_skew(xy, d0, cfg)
    xy_rot = rotate_keep_all(work, xy, angle, warp=False)[1]
    result = decode_lattice(xy_rot, d0, cfg)
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
