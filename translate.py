"""Tłumaczenie tekstu na Braille'a i eksport do Unicode (⠁⠃⠉) lub BRF (Braille ASCII)."""
from letters import cell, encode_text

BRAILLE_ASCII = ' A1B\'K2L@CIF/MSP"E3H9O6R^DJG>NTQ,*5<-U8V.%[$+X!&;:4\\0Z7(_?W]#Y)='
BLANK = cell("")

assert len(BRAILLE_ASCII) == 64 and len(set(BRAILLE_ASCII)) == 64


def _value(c):
    return sum(bit << i for i, bit in enumerate(c))


def _from_value(v):
    return tuple((v >> i) & 1 for i in range(6))


def _check_layout(cells_per_line, lines_per_page):
    if cells_per_line < 1 or lines_per_page < 1:
        raise ValueError("cells_per_line i lines_per_page muszą być >= 1")


def _wrap_lines(cells, cells_per_line):
    """Jeden akapit -> lista wierszy. Łamie na spacjach; zbyt długi wyraz jest cięty twardo."""
    words, cur = [], []
    for c in cells:
        if c == BLANK:
            words.append(cur)
            cur = []
        else:
            cur.append(c)
    words.append(cur)

    lines, line = [], []
    for w in words:
        while len(w) > cells_per_line:
            if line:
                lines.append(line)
                line = []
            lines.append(w[:cells_per_line])
            w = w[cells_per_line:]
        if not line:
            line = list(w)
        elif len(line) + 1 + len(w) <= cells_per_line:
            line += [BLANK] + w
        else:
            lines.append(line)
            line = list(w)
    if line or not lines:
        lines.append(line)
    return lines


def _paginate(lines, lines_per_page):
    return [lines[i:i + lines_per_page] for i in range(0, len(lines), lines_per_page)]


def wrap_cells(cells, cells_per_line=32, lines_per_page=25):
    """Zwraca strony: listę stron, strona = lista wierszy, wiersz = lista komórek."""
    _check_layout(cells_per_line, lines_per_page)
    return _paginate(_wrap_lines(cells, cells_per_line), lines_per_page)


def wrap_paragraphs(paragraphs, cells_per_line=32, lines_per_page=25):
    """Jak wrap_cells, ale każdy akapit zaczyna nowy wiersz; pusty akapit daje pusty wiersz."""
    _check_layout(cells_per_line, lines_per_page)
    lines = []
    for p in paragraphs:
        lines += _wrap_lines(p, cells_per_line)
    return _paginate(lines, lines_per_page)


def _unicode_pages(pages):
    return "\f".join("\n".join("".join(chr(0x2800 + _value(c)) for c in line)
                               for line in page) for page in pages)


def _brf_pages(pages):
    return "\f".join("\r\n".join("".join(BRAILLE_ASCII[_value(c)] for c in line)
                                 for line in page) for page in pages)


def to_unicode(cells, **layout):
    return _unicode_pages(wrap_cells(cells, **layout))


def to_brf(cells, **layout):
    return _brf_pages(wrap_cells(cells, **layout))


def brf_to_cells(brf):
    out = []
    for ch in brf:
        if ch in "\r\n\f":
            if out and out[-1] != BLANK:
                out.append(BLANK)
        else:
            out.append(_from_value(BRAILLE_ASCII.index(ch.upper())))
    while out and out[-1] == BLANK:
        out.pop()
    return out


def translate(text, fmt="brf", cells_per_line=32, lines_per_page=25):
    """Tekst (także wielowierszowy) -> gotowy plik jako napis. fmt: 'brf' albo 'unicode'."""
    exporters = {"brf": _brf_pages, "unicode": _unicode_pages}
    if fmt not in exporters:
        raise ValueError(f"Nieznany format: {fmt!r} (dozwolone: {', '.join(exporters)})")
    paragraphs = []
    for n, line in enumerate(text.splitlines(), 1):
        try:
            paragraphs.append(encode_text(line))
        except KeyError as e:
            raise ValueError(f"Nieobsługiwany znak: {e.args[0]!r} (linia {n})") from None
    return exporters[fmt](wrap_paragraphs(paragraphs, cells_per_line, lines_per_page))