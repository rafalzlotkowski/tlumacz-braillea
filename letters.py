UNKNOWN_CHAR = "*"


def cell(dots: str) -> tuple:
    """cell('145') -> (1,0,0,1,1,0). Kolejność: punkty 1..6."""
    return tuple(1 if str(i) in dots else 0 for i in range(1, 7))

BLANK = cell("")

_FOLD = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ")

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
    ",": "2", ";": "23", ":": "25", ".": "3", "?": "26", "!": "235",
    "-": "36",
}

CHAR_TO_CELL = {ch: cell(v) for ch, v in {**_LETTERS, **_PUNCT}.items()}
CHAR_TO_CELL[" "] = cell("")
BRAILLE_DICT = {pat: ch for ch, pat in CHAR_TO_CELL.items()}
assert len(BRAILLE_DICT) == len(CHAR_TO_CELL), "Powtórzony wzorzec w tabeli!"

CAPITAL_SIGN = cell("46")
NUMBER_SIGN = cell("3456")
SIGN_LABELS = {CAPITAL_SIGN: "^", NUMBER_SIGN: "#"}

BRAILLE_DIGITS = {"a": "1", "b": "2", "c": "3", "d": "4", "e": "5",
                  "f": "6", "g": "7", "h": "8", "i": "9", "j": "0"}
DIGIT_TO_LETTER = {v: k for k, v in BRAILLE_DIGITS.items()}

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
