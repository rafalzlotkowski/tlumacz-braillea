"""Dekoder polskiego Braille'a ze zdjęcia: python cli.py zdjecie.jpg [--roi X Y W H] [--save wynik.png]"""
import argparse
import sys

import cv2

from config import Config
from pipeline import process, select_roi, visualize
from vision import preprocess


def dots_of(pattern):
    return "".join(str(i + 1) for i, v in enumerate(pattern) if v) or "_"


def main():
    defaults = Config()
    ap = argparse.ArgumentParser(description="Dekoder polskiego Braille'a ze zdjęcia")
    ap.add_argument("image", help="ścieżka do zdjęcia")
    ap.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                    help="ROI w pikselach oryginału (pomija okno wyboru)")
    ap.add_argument("--save", help="zapisz wizualizację do pliku zamiast pokazywać okno")
    ap.add_argument("--debug", action="store_true",
                    help="wypisz każdą komórkę jako numery punktów i zapisz maskę")
    ap.add_argument("--no-split", action="store_true", help="nie rozdzielaj dużych plam")
    ap.add_argument("--depth", type=float, default=defaults.depth_frac,
                    help="głębokość progu 0.2-0.8 (mniej = luźniej, więcej = ostrzej)")
    args = ap.parse_args()

    cfg = Config(split_blobs=not args.no_split, depth_frac=args.depth)

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Nie można wczytać obrazu: {args.image}")

    if args.roi:
        x, y, w, h = args.roi
        crop = img[y:y + h, x:x + w]
    else:
        crop = select_roi(img, cfg)

    try:
        result, work, xy = process(crop, cfg)
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
        cv2.imwrite(base + "_mask.png", preprocess(work, cfg))
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
