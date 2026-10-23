"""Parametry dekodowania zdjęć (przekazywane jawnie, bez zmiennych globalnych)."""
from dataclasses import dataclass


@dataclass
class Config:
    preview_width: int = 800
    max_work_side: int = 2400
    max_skew_deg: float = 25
    cell_pitch_range: tuple = (2.2, 3.8)
    word_gap_factor: float = 4.0
    line_gap_factor: float = 1.5
    row_gap_factor: float = 0.5
    bg_downscale: float = 0.25
    split_blobs: bool = True
    depth_frac: float = 0.5
