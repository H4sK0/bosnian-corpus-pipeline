#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
final_sanity_clean.py

Zadnji prolaz prije Zenoda:

- NFC normalizacija
- izbacivanje neštampajućih / "čudnih" znakova
- sabijanje višestrukih razmaka
- drop praznih i prekratkih linija
- opcionalna deduplikacija linija (po fajlu ili globalno)

Primjer:

  python final_sanity_clean.py ^
      --in-root  "C:\\data_data\\genres_merged" ^
      --out-root "C:\\data_data\\genres_merged_final" ^
      --min-len  15 ^
      --dedup    global
"""

import argparse
from pathlib import Path
import unicodedata

def clean_line_basic(s: str) -> str:
    """Blaga normalizacija: NFC + uklanjanje čudnih znakova + sabijanje razmaka."""
    # 1) NFC
    s = unicodedata.normalize("NFC", s)

    out_chars = []
    for ch in s:
        # zadrži obične whitespace (space, tab); izbaci \r
        if ch in ("\n", "\r"):
            # newline rješavamo izvan ove funkcije
            continue
        if ch in ("\t", " "):
            out_chars.append(" ")
            continue

        # izbjegni neštampajuće / kontrolne
        if not ch.isprintable():
            continue

        # ovo ostavlja sve normalne unicode znakove (uklj. šđčćž, navodnike, itd.)
        out_chars.append(ch)

    s = "".join(out_chars)

    # 2) sabij višestruke razmake
    while "  " in s:
        s = s.replace("  ", " ")

    # 3) strip sa strane
    s = s.strip()

    return s

def process_file(
    src: Path,
    dst: Path,
    min_len: int,
    max_len: int | None,
    dedup_mode: str,
    global_seen: set[str] | None = None,
) -> tuple[int,int]:
    """
    Vrati (kept, dropped).
    dedup_mode: "none" | "file" | "global"
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    local_seen: set[str] = set()
    kept = 0
    dropped = 0

    with src.open("r", encoding="utf-8", errors="ignore") as fin, \
         dst.open("w", encoding="utf-8", newline="\n") as fout:

        for raw in fin:
            raw = raw.rstrip("\r\n")
            s = clean_line_basic(raw)

            # drop prazno / prekratko
            if not s or (min_len > 0 and len(s) < min_len):
                dropped += 1
                continue

            if max_len and len(s) > max_len:
                # po defaultu ne režemo, samo pustimo
                # ako želiš, možeš kasnije ovdje dodati rezanje po tački
                pass

            # dedup logika
            if dedup_mode == "file":
                if s in local_seen:
                    dropped += 1
                    continue
                local_seen.add(s)
            elif dedup_mode == "global" and global_seen is not None:
                if s in global_seen:
                    dropped += 1
                    continue
                global_seen.add(s)

            fout.write(s + "\n")
            kept += 1

    return kept, dropped

def main():
    ap = argparse.ArgumentParser(description="Završno čišćenje korpusa prije Zenoda.")
    ap.add_argument("--in-root", required=True, help="Ulazni root (žanrovski folderi, merged ili ne).")
    ap.add_argument("--out-root", required=True, help="Izlazni root (očisćeno).")
    ap.add_argument("--min-len", type=int, default=10,
                    help="Minimalna dužina linije (u znakovima) da bi ostala (default 10).")
    ap.add_argument("--max-len", type=int, default=0,
                    help="Ako >0, možeš koristiti za heuristiku; trenutno se ne reže, samo info.")
    ap.add_argument("--dedup", choices=["none","file","global"], default="file",
                    help="Dedup mod: none / file / global (default: file).")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if not in_root.is_dir():
        print(f"[ERROR] Ne postoji in_root: {in_root}")
        raise SystemExit(1)

    global_seen: set[str] | None = set() if args.dedup == "global" else None

    total_files = 0
    total_kept = 0
    total_dropped = 0

    print(f"[INFO] Krenuo završni prolaz – in_root={in_root}, out_root={out_root}")
    print(f"[INFO] dedup={args.dedup}, min_len={args.min_len}, max_len={args.max_len or 'none'}")

    # prolazi kroz cijelo stablo
    for src in sorted(in_root.rglob("*.txt")):
        if not src.is_file():
            continue
        rel = src.relative_to(in_root)
        dst = out_root / rel

        print(f"[FILE] {rel}")
        kept, dropped = process_file(
            src,
            dst,
            min_len=args.min_len,
            max_len=(args.max_len if args.max_len > 0 else None),
            dedup_mode=args.dedup,
            global_seen=global_seen,
        )
        print(f"       kept={kept:,}, dropped={dropped:,}")
        total_files += 1
        total_kept += kept
        total_dropped += dropped

    print("\n[DONE]")
    print(f"  fajlova obrađeno: {total_files}")
    print(f"  linija zadržano : {total_kept:,}")
    print(f"  linija odbačeno : {total_dropped:,}")

if __name__ == "__main__":
    main()
