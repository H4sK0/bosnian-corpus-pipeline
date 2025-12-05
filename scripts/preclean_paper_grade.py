#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
preclean_paper_grade.py.

- Normalizacija tipografije: » « „ “ → ", —/– → -, elipsa … → ...
- Uklanjanje meta-linija: "izvor:", pipe headeri "| rtrs |", "detektor.ba" i sl.
- Čišćenje fusnota i inline numeričkih markera: [0], (0), *0), [12], (123)…
- Uklanjanje bullets/metaka (•, - na početku) i “forumskih” repova.
- Opcije za latinicu:
    --latin-only         → odbacuje linije koje sadrže ćirilicu
    --cyr2lat            → transliteriše ćirilicu → latinica (sr/bh mapa)
- Jači heuristički “noise” filter + zbijanje whitespace/newline (max 2 NL).
- Ispravljen dupli return, stabilniji progress (% po bajtovima).
"""

from pathlib import Path
import argparse, re, sys, unicodedata, io

try:
    from ftfy import fix_text as _ftfy_fix
except Exception:
    _ftfy_fix = None

MOJIBAKE_FIXES = (
    ("â€ž", "„"), ("â€œ", "“"), ("â€ś", "”"),
    ("â€˜", "‘"), ("â€™", "’"), ("â€š", "‚"),
    ("â€“", "–"), ("â€”", "—"), ("â€¦", "…"), ("â€¢", "•"),
    ("Å¡", "š"), ("Å½", "Ž"), ("Å¾", "ž"), ("Å ", "Š"),
    ("Ä", "č"), ("ÄŒ", "Č"), ("Ä‡", "ć"), ("Ä†", "Ć"),
    ("Ä‘", "đ"), ("Ä", "Đ"),
    ("Ã„", "Ä"), ("Ã„â€¡", "Ä‡"), ("Ã„Â", "č"), ("Ã„Â", "Đ"),
    ("Ã…", "Å"), ("Ã…Â¡", "š"), ("Ã…Â½", "Ž"), ("Ã…Â¾", "ž"),
    ("Ã‘", "Ñ"), ("Ã—", "×"),
    ("Â©", "©"), ("Â®", "®"), ("Â°", "°"), ("Â·", "·"),
    ("Â", ""),
)

def fix_mojibake(s: str) -> str:
    if _ftfy_fix is not None:
        try:
            return _ftfy_fix(s)
        except Exception:
            pass
    for bad, good in MOJIBAKE_FIXES:
        s = s.replace(bad, good)
    return 
# ---Ćirilica → latinica ----
_CYR2LAT_MAP = str.maketrans({
   
    "А":"A","Б":"B","В":"V","Г":"G","Д":"D","Ђ":"Đ","Е":"E","Ж":"Ž","З":"Z","И":"I","Ј":"J",
    "К":"K","Л":"L","Љ":"Lj","М":"M","Н":"N","Њ":"Nj","О":"O","П":"P","Р":"R","С":"S","Т":"T",
    "Ћ":"Ć","У":"U","Ф":"F","Х":"H","Ц":"C","Ч":"Č","Џ":"Dž","Ш":"Š",
    "а":"a","б":"b","в":"v","г":"g","д":"d","ђ":"đ","е":"e","ж":"ž","з":"z","и":"i","ј":"j",
    "к":"k","л":"l","љ":"lj","м":"m","н":"n","њ":"nj","о":"o","п":"p","р":"r","с":"s","т":"t",
    "ћ":"ć","у":"u","ф":"f","х":"h","ц":"c","ч":"č","џ":"dž","ш":"š",
})
_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")

def has_cyrillic(s: str) -> bool:
    return bool(_CYRILLIC_RE.search(s))

def cyr2lat(s: str) -> str:
    return s.translate(_CYR2LAT_MAP)

RE_URL       = re.compile(r"https?://\S+|www\.\S+", re.I)
RE_EMAIL     = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", re.I)
RE_IBAN      = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,}\b")
RE_PHONE     = re.compile(r"\+?\d[\d\s()./\-]{6,}\d")
RE_FILENAME  = re.compile(r"\b\S+\.(jpg|jpeg|png|gif|pdf|docx?|xlsx?|pptx?)\b", re.I)
RE_FORUM_RULES = re.compile(r"(Ne (možete|mozete) .* forumu|Powered by phpBB|MojForum)", re.I)

RE_SOURCE_LINE = re.compile(r"^\s*(izvor\s*:.*|source\s*:.*)$", re.I)
RE_PIPE_HEADER = re.compile(r"^\s*\|[^|]+\|.*$")  # npr. "| rtrs | ..."

RE_PUBLISH  = re.compile(r"^\s*(Objavljeno|Post published)\b.*$", re.I)
RE_ARCHIVE  = re.compile(r"^\s*(Yearly|Monthly)\s+Archive[s]?:.*$", re.I)
RE_CATEGORY = re.compile(r"^\s*Category Archives:.*$|^Entries by .*|^#\w+", re.I)
RE_AUTHOR_ARCH = re.compile(r"^\s*author archives\s*:\s*.*$", re.I)

BHS_DAYS   = r"(Ponedjeljak|Utorak|Srijeda|Četvrtak|Petak|Subota|Nedjelja)"
BHS_MONTHS = r"(januar|februar|mart|april|maj|juni|jun|juli|jul|avgust|septembar|oktobar|novembar|decembar)"
RE_BHS_DATE_LINE = re.compile(
    rf"^\s*({BHS_DAYS}\s*,\s*)?\d{{1,2}}\.\s*{BHS_MONTHS}\s*\d{{4}}(\.\s*u\s+\d{{1,2}}:\d{{2}})?\s*$", re.I
)
RE_BHS_DATE_DOTTED = re.compile(r"^\s*\d{1,2}\.\s*\d{1,2}\.\s*\d{2,4}\.?\s*(\d{1,2}:\d{2}h?)?\s*$", re.I)

EN_DAYS    = r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
EN_MONTHS  = r"(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
RE_EN_DATE_LINE = re.compile(
    rf"^\s*({EN_DAYS}\s*,\s*)?\d{{1,2}}\s+{EN_MONTHS}\s+\d{{4}}(\s+\d{{1,2}}:\d{{2}})?\s*$", re.I
)

RE_ZERO_DATE = re.compile(r"^\s*0{1,2}\.0{1,2}\.0{4}(?:\s+0{1,2}:\d{2}(?::\d{2})?)?\s*$")
RE_ZERO_TIME = re.compile(r"^\s*0{1,2}:\d{2}(?::\d{2})?\s*$")


RE_INLINE_FOOT = re.compile(r"(\[\s*\d+\s*\]|\(\s*\d+\s*\)|\*\s*\d+\))")

RE_BULLET_LINE = re.compile(r"^\s*(•|\-|\*|—|–)\s+")

RE_EMPTY_BRACKETS = re.compile(r"\[\s*\]")

RE_SOLO_USERNAME  = re.compile(r"^\s*[A-Za-z0-9_.\-]{3,}\s*$")

RE_FANCY_QUOTES = re.compile(r"[»«„“]")
RE_LONG_DASHES  = re.compile(r"[—–]")
RE_ELLIPSIS     = re.compile(r"…")

RE_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")
RE_MULTI_SPACE        = re.compile(r"[ \t]{2,}")
RE_MULTI_NL           = re.compile(r"\n{3,}") 

RE_PUNCT_RUNS = re.compile(r"([!?.,;:\-])\1{3,}")

DIGIT_MAP = str.maketrans("0123456789", "0000000000")

def mostly_nonletters(s: str, thresh: float) -> bool:
    stripped = s.replace(" ", "")
    if not stripped:
        return True
    letters = sum(ch.isalpha() for ch in stripped)
    return (1.0 - letters/len(stripped)) >= thresh

def is_noise_line(s: str, nonletter_thresh: float) -> bool:
    if not s.strip():
        return True
    if (RE_PUBLISH.match(s) or RE_ARCHIVE.match(s) or RE_CATEGORY.match(s) or
        RE_AUTHOR_ARCH.match(s) or RE_SOURCE_LINE.match(s) or RE_PIPE_HEADER.match(s)):
        return True
    if RE_BHS_DATE_LINE.match(s) or RE_BHS_DATE_DOTTED.match(s) or RE_EN_DATE_LINE.match(s):
        return True
    if RE_ZERO_DATE.match(s) or RE_ZERO_TIME.match(s):
        return True
    if RE_FORUM_RULES.search(s):
        return True
    if RE_BULLET_LINE.match(s):
        return True
    if RE_SOLO_USERNAME.match(s):
        return True
    if mostly_nonletters(s, nonletter_thresh):
        return True
    return False

def normalize_typography(s: str) -> str:
    s = RE_FANCY_QUOTES.sub('"', s)
    s = RE_LONG_DASHES.sub("-", s)
    s = RE_ELLIPSIS.sub("...", s)
    return s

def clean_line(s: str, digits=False, lower=False, nonletter_thresh=0.6,
               latin_only=False, translit_cyr2lat=False) -> str:
   
    s = unicodedata.normalize("NFC", s)
    s = fix_mojibake(s)

    # Ćirilica → latinica
    if translit_cyr2lat and has_cyrillic(s):
        s = cyr2lat(s)
    # Odbaci linije sa ćirilicom 
    if latin_only and has_cyrillic(s):
        return ""

    s = RE_URL.sub("", s)
    s = RE_EMAIL.sub("", s)
    s = RE_IBAN.sub("", s)
    s = RE_PHONE.sub("", s)
    s = RE_FILENAME.sub("", s)
    if is_noise_line(s, nonletter_thresh):
        return ""

    s = RE_INLINE_FOOT.sub("", s)
    s = RE_EMPTY_BRACKETS.sub("", s)

    s = normalize_typography(s)

    s = RE_SPACE_BEFORE_PUNCT.sub(r"\1", s)
    s = RE_MULTI_SPACE.sub(" ", s).strip()

    s = RE_PUNCT_RUNS.sub(r"\1\1\1", s)


    s = s.replace("' '", "''").replace('" "', '""')

    # Cifre → '0'
    if digits:
        s = s.translate(DIGIT_MAP)
    # Lowercase
    if lower:
        s = s.lower()
    return s

def process_file(src: Path, dst: Path, digits=False, lower=False,
                 nonletter_thresh=0.6, show_progress=True,
                 latin_only=False, translit_cyr2lat=False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    wrote = False
    kept, dropped = 0, 0

    try:
        total_size = max(1, src.stat().st_size)
    except Exception:
        total_size = 1
    last_percent = -1
    line_counter = 0

    with open(src, "rb") as fb:
        text = io.TextIOWrapper(fb, encoding="utf-8", errors="ignore", newline="")
        with open(dst, "w", encoding="utf-8", newline="\n") as fout:
            while True:
                raw = text.readline()
                if not raw:
                    break
                raw = raw.rstrip("\r\n")
                line = clean_line(
                    raw,
                    digits=digits,
                    lower=lower,
                    nonletter_thresh=nonletter_thresh,
                    latin_only=latin_only,
                    translit_cyr2lat=translit_cyr2lat
                )
                if line:
                    fout.write(line + "\n")
                    wrote = True
                    kept += 1
                else:
                    dropped += 1

                line_counter += 1
                if show_progress:
                    try:
                        pos = fb.tell()
                    except Exception:
                        pos = 0
                    percent = int((pos / total_size) * 100)
                    if percent != last_percent:
                        print(f"\r[{src.name}] {percent:3d}%  kept={kept:,}  drop={dropped:,}  lines={line_counter:,}",
                              end="", flush=True)
                        last_percent = percent
    if show_progress:
        print()
    return wrote, kept, dropped

def iter_txt(root: Path):
    if root.is_file():
        yield root
    else:
        for p in root.rglob("*.txt"):
            if p.is_file():
                yield p

def main():
    ap = argparse.ArgumentParser(description="Paper-grade pre-clean (sa % progresom) za entropiju/energiju.")
    ap.add_argument("--in", dest="inp", required=True, help="Ulaz .txt ili direktorij (rekurzivno).")
    ap.add_argument("--out", dest="out", required=True, help="Izlazni direktorij (.clean.txt).")
    ap.add_argument("--digits", action="store_true", help="Sve cifre -> '0'.")
    ap.add_argument("--lower",  action="store_true", help="Sve u lowercase.")
    ap.add_argument("--drop-nonletters", type=float, default=0.60,
                    help="Prag udjela ne-slova za odbacivanje linije (default 0.60).")
    ap.add_argument("--latin-only", action="store_true",
                    help="Odbaci linije koje sadrže ćirilicu.")
    ap.add_argument("--cyr2lat", action="store_true",
                    help="Transliteriši ćirilicu → latinica umjesto odbacivanja.")
    ap.add_argument("--no-progress", action="store_true", help="Isključi % ispis po fajlu.")
    args = ap.parse_args()

    src = Path(args.inp)
    out = Path(args.out)
    base = src if src.is_dir() else src.parent

    files = list(iter_txt(src))
    if not files:
        print("Nema .txt ulaza.", file=sys.stderr); sys.exit(2)

    if _ftfy_fix is None:
        print("[INFO] 'ftfy' nije instaliran — koristi se ugrađeni mojibake-fix. (Preporuka: pip install ftfy)")

    n_files, total_kept, total_dropped = 0, 0, 0
    for f in files:
        try:
            rel = f.relative_to(base)
        except Exception:
            rel = Path(f.name)
        dst = out / rel.with_suffix(".clean.txt")
        dst.parent.mkdir(parents=True, exist_ok=True)

        wrote, kept, dropped = process_file(
            f, dst,
            digits=args.digits,
            lower=args.lower,
            nonletter_thresh=args.drop_nonletters,
            show_progress=(not args.no_progress),
            latin_only=args.latin_only,
            translit_cyr2lat=args.cyr2lat
        )

        status = "OK" if wrote else "EMPTY"
        print(f"{status} -> {dst}   (kept={kept:,}, dropped={dropped:,})")
        n_files += 1
        total_kept += kept
        total_dropped += dropped

    print(f"\nGotovo. Obradjeno fajlova: {n_files}")
    print(f"Zadržano linija: {total_kept:,}")
    print(f"Odbaceno linija: {total_dropped:,}")

if __name__ == "__main__":
    main()
