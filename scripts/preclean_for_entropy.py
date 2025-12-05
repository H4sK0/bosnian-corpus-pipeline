#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preclean_for_entropy.py
Minimalni pre-clean koraci za entropiju/energiju:
1) Unicode NFC + UTF-8 izlaz
2) Uklanjanje "smeće" linija (datumi tipa 19.07.2012., rubrike, Ocjena : N)
3) Popravka razmaka oko interpunkcije (nema razmaka PRIJE ,.;:!?)
4) Smanjenje višestrukih razmaka i praznih redova
5) (opc) Normalizacija cifara -> 0 (--digits)
6) (opc) Lowercase (--lower)
"""

from pathlib import Path
import argparse, re, unicodedata, sys

# --- obrasci ---
RE_DATE_ONLY   = re.compile(r"^\s*\d{1,2}\.\d{1,2}\.\d{2,4}\.\s*$") 
RE_RATING      = re.compile(r"^\s*Ocjena\s*:\s*\d+\s*$", re.I)

DEFAULT_HEADERS = (
    "Najnovije vijesti",
    "Stara priča",
    "Stara prica",
    "Najnovije vesti",
)

RE_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")

RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")

DIGIT_MAP = str.maketrans("0123456789", "0000000000")

def build_header_regex(extra_headers: list[str]|None):
    items = list(DEFAULT_HEADERS)
    if extra_headers:
        items.extend(h for h in extra_headers if h.strip())
   
    pat = r"^\s*(%s)\s*$" % "|".join(re.escape(x) for x in items)
    return re.compile(pat, re.I)

def clean_line(line: str,
               re_header,
               normalize_digits: bool=False,
               to_lower: bool=False) -> str:
    # Unicode NFC
    line = unicodedata.normalize("NFC", line)


    if RE_DATE_ONLY.match(line): 
        return ""
    if re_header.match(line):     
        return ""
    if RE_RATING.match(line):     
        return ""
    
    line = RE_SPACE_BEFORE_PUNCT.sub(r"\1", line)


    line = RE_MULTI_SPACE.sub(" ", line).strip()

    
    if normalize_digits:
        line = line.translate(DIGIT_MAP)

    if to_lower:
        line = line.lower()

    return line

def process_file(src: Path, dst: Path, re_header, digits: bool, lower: bool) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    wrote_any = False
    with src.open("r", encoding="utf-8", errors="ignore") as fin, \
         dst.open("w", encoding="utf-8", newline="\n") as fout:
        for raw in fin:
            raw = raw.rstrip("\n\r")
            line = clean_line(raw, re_header, digits, lower)
            if line:
                fout.write(line + "\n")
                wrote_any = True
    return wrote_any

def iter_txt(root: Path):
    if root.is_file():
        if root.suffix.lower() in {".txt", ".text"}:
            yield root
    else:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".txt", ".text"}:
                yield p

def main():
    ap = argparse.ArgumentParser(description="Pre-clean tekst za entropiju/energiju.")
    ap.add_argument("--in",  dest="inp",  required=True, help="Ulazni .txt ili direktorijum (rekurzivno).")
    ap.add_argument("--out", dest="out",  required=True, help="Izlazni direktorijum za čiste .txt fajlove.")
    ap.add_argument("--digits", action="store_true", help="Normalizuj sve cifre u '0'.")
    ap.add_argument("--lower",  action="store_true", help="Pretvori sav tekst u lowercase.")
    ap.add_argument("--extra-header", action="append",
                    help="Dodatna ime(rubrika) koju treba ukloniti ako je sama u liniji. Može više puta.")
    args = ap.parse_args()

    src = Path(args.inp)
    out = Path(args.out)
    re_header = build_header_regex(args.extra_header)

    files = list(iter_txt(src))
    if not files:
        print("Nema .txt fajlova na ulazu.", file=sys.stderr)
        sys.exit(2)

    base = src if src.is_dir() else src.parent
    count = 0
    for f in files:
        rel = f.relative_to(base) if f.is_relative_to(base) else Path(f.name)
        dst = out / rel.with_suffix(".clean.txt")
        dst.parent.mkdir(parents=True, exist_ok=True)
        process_file(f, dst, re_header, args.digits, args.lower)
        count += 1
        print("OK ->", dst)

    print(f"Gotovo. Obradjeno fajlova: {count}")

if __name__ == "__main__":
    main()
