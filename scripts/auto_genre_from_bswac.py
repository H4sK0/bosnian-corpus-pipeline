#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_genre_from_bswac.py

Treniraj žanrovski model iz postojećih CLASSLA foldera (po žanrovima),
mapiranim na nekoliko super-žanrova.
Super-žanrovi: NEWS, OPINION, FORUM_CHAT, INFO_HOWTO, LEGAL_ADMIN,
LITERATURE, ADS_PROMO, MIX_OTHER
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys
import unicodedata
import json
import os
import time
import random
from collections import Counter, defaultdict, deque

# ---------------------  SUPER-ŽANR -------------------------

SUPER_MAP = {
    "NEWS":         {"news"},
    "OPINION":      {"opinion"},
    "FORUM_CHAT":   {"forum_chat"},
    "INFO_HOWTO":   {"info_howto"},
    "LEGAL_ADMIN":  {"legal_admin"},
    "LITERATURE":   {"literature"},
    "ADS_PROMO":    {"ads_promo"},
    "MIX_OTHER":    {"mix"},
}
ALL_SUPERS = ["NEWS", "OPINION", "FORUM_CHAT", "INFO_HOWTO",
              "LEGAL_ADMIN", "LITERATURE", "ADS_PROMO", "MIX_OTHER"]

CLASS_SHOTS = {
   
    "NEWS": 5,
    "FORUM_CHAT": 3,
    "LITERATURE": 12,
    "OPINION": 10,
    "INFO_HOWTO": 10,
    "LEGAL_ADMIN": 10,
    "ADS_PROMO": 10,
    "MIX_OTHER": 1,
}

# --------------------------- HEURISTIČKI BOOST -------------------------------

KEYWORDS = {
    "LEGAL_ADMIN": {
        "član", "clan", "stav", "zakon", "službeni glasnik", "sluzbeni glasnik",
        "kazneni", "prekršaj", "prekrsaj", "upravni postupak", "rješenje", "rjesenje",
        "ovlašten", "ovlasten", "ministarstvo", "pravilnik", "službeni list",
    },
    "NEWS": {
        "saopćeno", "saopsteno", "saopšteno", "saopćila", "saopštio", "saopstio",
        "premijer", "vlada", "skupština", "skupstina", "parlament", "ministar",
        "agencija", "izvijestio", "izvjestio", "izvještava", "novinska agencija",
        "hapšenje", "hapsenje", "uhapšen", "uhapsen", "saobraćajna nesreća",
        "sjednica", "konferencija za medije", "u izjavi za medije",
    },
    "INFO_HOWTO": {
        "kako", "upute", "uputstvo", "uputstva", "korak", "koraci", "instalacija",
        "postupak", "recept", "potrebni sastojci", "potrebno je", "uraditi sljedeće",
        "slijedite ove korake", "slijedi nekoliko savjeta",
    },
    "FORUM_CHAT": {
        "lol", "haha", "hehe", ":)", ":D", ";)", "pozz", "pozdrav svima",
        "muskarci", "muškarci", "cura", "djevojka", "dečko", "decko",
        "sta mislite", "šta mislite", "ima li neko", "jel zna neko", "help :)",
    },
    "ADS_PROMO": {
        "akcija", "popust", "cijena", "snizeno", "sniženo", "kupite", "kupovina",
        "ponuda", "specijalna ponuda", "gratis", "besplatno", "samo danas",
        "do isteka zaliha", "poručite", "narucite", "kupi odmah", "uskoro u prodaji",
    },
    "LITERATURE": {
        "lik", "priča", "prica", "roman", "stih", "ponad", "junak", "junakinja",
        "pripovjedač", "pripovjedac", "poglavlje", "proza", "poezija", "stihovi",
        "pričao mi je", "reče", "rece", "kazao je", "šapnu", "sapnu", "duša", "dusa",
    },
    "OPINION": {
        "smatram", "po mom mišljenju", "po mom misljenju", "po mome",
        "po meni", "mislim da", "uvjeren sam", "uvjerena sam",
        "moje je mišljenje", "moje misljenje", "ovo pokazuje", "ovo jasno govori",
        "argument", "argumenti", "kritika", "komentar", "stav", "analiza",
    },
}


def rule_boost(text: str, probs: dict[str, float], delta: float = 0.08) -> dict[str, float]:
    """Blago poguraj klasu ako prepozna ≥2 ključne riječi; renormalizuj."""
    lo = text.lower()
    hit = None
    best_hits = 0
    for lbl, kws in KEYWORDS.items():
        hits = sum(1 for k in kws if k in lo)
        if hits > best_hits and hits >= 2:
            best_hits = hits
            hit = lbl
    if hit and hit in probs:
        probs[hit] = min(1.0, probs[hit] + delta)
    s = sum(probs.values())
    if s > 0:
        for k in probs:
            probs[k] /= s
    return probs

# ------------------------------ UTIL -----------------------------------------


def norm_text(s: str, lower: bool = False) -> str:
    s = unicodedata.normalize("NFC", s.replace("\r", ""))
    return s.lower() if lower else s


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def format_bytes(n: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0
    while n >= 1024 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    return f"{n:.1f} {units[i]}"


def iter_super_folders(root: Path):
    """Vrati (super_label, folder_path) za sve postojeće foldere iz SUPER_MAP."""
    for super_lbl, fine_set in SUPER_MAP.items():
        for fine in fine_set:
            p = root / fine
            if p.is_dir():
                yield super_lbl, p


def check_training_root(root: Path, patterns=("*.txt", "*.clean.txt")) -> int:
    print(f"[CHECK] Pregledam tren. root: {root}")
    total = 0
    for super_lbl, folder in iter_super_folders(root):
        files = []
        for pat in patterns:
            files.extend(folder.rglob(pat))
        files = sorted(set(files))
        count = len(files)
        total += count
        status = "OK" if folder.is_dir() else "MISSING"
        print(f"  - {super_lbl:12s} | {folder} -> {status:8s} | fajlova: {count}")
        for f in files[:3]:
            print(f"      • {f.name}")
    print(f"[CHECK] Ukupno trening fajlova: {total}")
    if total == 0:
        print("[ERROR] Nema trening fajlova! Provjeri --root, nazive foldera i ekstenzije.", file=sys.stderr)
    return total

# -------------------- NASUMIČNO UZORKOVANJE ZA TRENING -----------------------


def sample_snippets_from_file(path: Path, shots: int, shot_size: int, lower: bool) -> list[str]:
    """Iz jednog (velikog) fajla izvuče `shots` nasumičnih isječaka dužine ~shot_size znakova."""
    try:
        data = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    if not data:
        return []

    n = len(data)
    snippets: list[str] = []
    if n <= shot_size:
        snippets.append(norm_text(data, lower=lower))
        return snippets

    for _ in range(shots):
        start = random.randint(0, max(0, n - shot_size))
        end = start + shot_size
        chunk = data[start:end]
        if chunk:
            snippets.append(norm_text(chunk, lower=lower))
    return snippets


def load_training_samples_random(
    root: Path,
    shots: int = 5,
    shot_size: int = 20_000,
    lower: bool = True,
    patterns=("*.txt", "*.clean.txt"),
    max_files_per_class: int | None = None,
    class_min_keep: int = 2,
) -> tuple[list[str], list[str]]:
    """
    Nasumično izvlači više 'shotova' po trening-fajlu. Vraća X, y.
    - shots: default broj isječaka po fajlu (ako klasa nije u CLASS_SHOTS)
    - shot_size: dužina jednog isječka u znakovima
    - class_min_keep: ako klasa dobije premalo uzoraka, preskače se iz modela
    """
    print("[INFO] Učitavam trening uzorke (nasumično uzorkovanje)…")
    X: list[str] = []
    y: list[str] = []
    per_class_count: Counter = Counter()

    for super_lbl, folder in iter_super_folders(root):
        files: list[Path] = []
        for pat in patterns:
            files.extend(folder.rglob(pat))
        files = sorted(set(files))
        if max_files_per_class:
            files = files[:max_files_per_class]

        class_shots = CLASS_SHOTS.get(super_lbl, shots)
        print(f"  {super_lbl:12s} | {folder} | fajlova: {len(files)} | shots/fajl: {class_shots}")

        for f in files:
            snippets = sample_snippets_from_file(
                f, shots=class_shots, shot_size=shot_size, lower=lower
            )
            for s in snippets:
                if s.strip():
                    X.append(s)
                    y.append(super_lbl)
                    per_class_count[super_lbl] += 1


    if per_class_count:
        alive_classes = {k for k, v in per_class_count.items() if v >= class_min_keep}
        if len(alive_classes) < len(per_class_count):
            X2, Y2 = [], []
            for xi, yi in zip(X, y):
                if yi in alive_classes:
                    X2.append(xi)
                    Y2.append(yi)
            X, y = X2, Y2
            per_class_count = Counter(y)

    print(f"[OK] Trening uzoraka: {len(X)}")
    print("[INFO] Po klasama:", dict(per_class_count))
    return X, y

def stream_blocks(path: Path, block_chars: int, overlap: int, lower: bool):
    """
    Generator: vraća (block_text, bytes_increment).
    Blokovi se preklapaju sa 'overlap' znakova (može biti 0).
    """
    buf = ""
    bytes_since_yield = 0
    target = block_chars

    with path.open("r", encoding="utf-8", errors="ignore") as fin:
        while True:
            need = max(0, target - len(buf))
            if need > 0:
                chunk = fin.read(need)
            else:
                chunk = ""

            if not chunk:
                
                if buf:
                   
                    text = norm_text(buf, lower=lower)
                    raw_len = len(buf.encode("utf-8", errors="ignore"))
                    bytes_since_yield += raw_len
                    yield text, bytes_since_yield
                break

            raw_len = len(chunk.encode("utf-8", errors="ignore"))
            bytes_since_yield += raw_len

            buf += chunk
            if len(buf) >= block_chars:
                block_text = buf[:block_chars]
                buf = buf[block_chars - overlap:] if overlap > 0 else ""
                text = norm_text(block_text, lower=lower)
                yield text, bytes_since_yield
                bytes_since_yield = 0

# -------------------------- MODEL (TF-IDF CHAR+WORD) -------------------------


def build_model(min_df: int, max_df: float, max_features: int, class_weight: str | None):
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    feats = FeatureUnion([
        ("char", TfidfVectorizer(
            analyzer="char", ngram_range=(3, 5),
            min_df=min_df, max_df=max_df, max_features=max_features // 2,
            sublinear_tf=True
        )),
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2),
            min_df=max(2, min_df // 2), max_df=max_df,
            max_features=max_features // 2,
            token_pattern=r"\b\w+\b", lowercase=True
        )),
    ])

    lr = LogisticRegression(
        max_iter=2000,
        solver="saga",
        penalty="l2",
        class_weight=(None if (class_weight is None or class_weight == "none") else class_weight)
    )

    clf = Pipeline([
        ("feats", feats),
        ("lr", lr),
    ])
    return clf

# -------------------------- PAKOVANJE BLOKOVA --------------------------------


def pack_blocks_equal_mb(gdir: Path, target_mb: int) -> None:
    """Spaja *.block*.txt u pakete ~target_mb MB: pack_000.txt, pack_001.txt, ..."""
    if target_mb <= 0:
        return
    block_files = sorted(gdir.glob("*.block*.txt"))
    if not block_files:
        return

    target_bytes = target_mb * 1024 * 1024
    pack_idx = 0
    fout = None
    cur_size = 0

    print(f"[PACK] {gdir.name}: {len(block_files)} blokova -> paketi po ≈{target_mb} MB")

    try:
        for bf in block_files:
            try:
                txt = bf.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                print(f"[WARN] pack skip {bf}: {e}", file=sys.stderr)
                continue

            b = len(txt.encode("utf-8", errors="ignore"))
            if fout is None or (cur_size + b > target_bytes and cur_size > 0):
                if fout is not None:
                    fout.close()
                out_path = gdir / f"pack_{pack_idx:03d}.txt"
                fout = out_path.open("w", encoding="utf-8", newline="\n")
                pack_idx += 1
                cur_size = 0

            fout.write(txt)
            if not txt.endswith("\n"):
                fout.write("\n")
            cur_size += b
    finally:
        if fout is not None:
            fout.close()

# --------------------------------- MAIN --------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Root sa žanrovskim folderima (CLASSLA extract / C:\\data_data).")
    ap.add_argument("--out", required=True,
                    help="Izlazni root (auto-žanrovani blokovi).")
    ap.add_argument("--bswac", nargs="+", required=True,
                    help="Spisak BSWaC *.clean.txt fajlova za klasifikaciju.")
    ap.add_argument("--shots", type=int, default=5,
                    help="Default broj nasumičnih uzoraka (shotova) po trening-fajlu.")
    ap.add_argument("--shot-size", type=int, default=20_000,
                    help="Veličina jednog uzorka u znakovima.")
    ap.add_argument("--max-files-per-class", type=int, default=None,
                    help="Opcionalno: ograniči broj fajlova po klasi.")
    ap.add_argument("--class-min-keep", type=int, default=2,
                    help="Min #uzoraka da klasa ostane u treningu.")
    ap.add_argument("--block-chars", type=int, default=100_000,
                    help="Veličina bloka za klasifikaciju (znakovi).")
    ap.add_argument("--overlap", type=int, default=0,
                    help="Preklapanje između blokova (znakovi).")
    ap.add_argument("--vote-window", type=int, default=1,
                    help="Većinski glas preko zadnjih N blokova (1 = bez zaglađivanja).")
    ap.add_argument("--conf-thresh", type=float, default=0.20,
                    help="Minimalna vjerovatnoća za klasu (inače MIX_OTHER).")
    ap.add_argument("--lower", action="store_true",
                    help="Lowercase normalizacija.")
    ap.add_argument("--debug-every", type=int, default=0,
                    help="Svaki N-ti blok ispiši top-3 vjerovatnoće (0=isključeno).")
    ap.add_argument("--min-df", type=int, default=5)
    ap.add_argument("--max-df", type=float, default=0.995)
    ap.add_argument("--max-features", type=int, default=300_000)
    ap.add_argument("--class-weight", type=str, default="balanced",
                    choices=["none", "balanced"])
    ap.add_argument("--pack-equal-mb", type=int, default=0,
                    help="Ako >0, pravi pakete po ~N MB po žanru.")

    ap.add_argument("--random-state", type=int, default=13,
                    help="Za reproducibilnost uzorkovanja.")

    args = ap.parse_args()
    random.seed(args.random_state)

    root = Path(args.root)
    out_root = Path(args.out)
    ensure_dir(out_root)
    _ = check_training_root(root)
    X, y = load_training_samples_random(
        root,
        shots=args.shots,
        shot_size=args.shot_size,
        lower=args.lower,
        max_files_per_class=args.max_files_per_class,
        class_min_keep=args.class_min_keep,
    )
    if not X:
        print("[ERROR] Nema trening uzoraka! Provjeri --root / mapiranja / ekstenzije.", file=sys.stderr)
        sys.exit(2)
    clf = build_model(
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=args.max_features,
        class_weight=(None if args.class_weight == "none" else args.class_weight),
    )
    print("[INFO] Trening modela…")
    t0 = time.time()
    clf.fit(X, y)
    print(f"[OK] Model spreman. (trajanje {time.time() - t0:.1f}s)")
    per_genre_counts: Counter = Counter()
    per_file_stats: dict[str, dict[str, int]] = {}

    for bpath_str in args.bswac:
        bpath = Path(bpath_str.strip('"'))
        if not bpath.is_file():
            print(f"[WARN] Ne postoji: {bpath}", file=sys.stderr)
            continue

        total_bytes = bpath.stat().st_size
        bytes_done = 0
        blocks_done = 0
        file_stats: Counter = Counter()
        start = time.time()

        label_window = deque(maxlen=max(1, args.vote_window))

        print(f"[RUN] {bpath.name}: klasifikacija blokova… ukupno {format_bytes(total_bytes)}")

        for block_text, inc in stream_blocks(
            bpath, args.block_chars, args.overlap, args.lower
        ):
            bytes_done += inc

            probs_vec = clf.predict_proba([block_text])[0]
            labels = clf.classes_
            pdict = {labels[i]: float(probs_vec[i]) for i in range(len(labels))}
            pdict = rule_boost(block_text, pdict, delta=0.08)
            best_lbl = max(pdict, key=pdict.get)
            best_p = pdict[best_lbl]
            raw_lbl = best_lbl if best_p >= args.conf_thresh else "MIX_OTHER"
            label_window.append(raw_lbl)
            if len(label_window) == 1:
                super_lbl = raw_lbl
            else:
                cnt = Counter(label_window)
                super_lbl = max(cnt.items(), key=lambda x: x[1])[0]

            out_dir = out_root / super_lbl
            ensure_dir(out_dir)
            out_file = out_dir / f"{bpath.stem}.block{blocks_done:05d}.txt"
            while out_file.exists():
                blocks_done += 1
                out_file = out_dir / f"{bpath.stem}.block{blocks_done:05d}.txt"

            with out_file.open("w", encoding="utf-8", newline="\n") as fout:
                fout.write(block_text if block_text.endswith("\n") else block_text + "\n")

            per_genre_counts[super_lbl] += 1
            file_stats[super_lbl] += 1
            blocks_done += 1
            if args.debug_every and (blocks_done % args.debug_every == 0):
                top3 = sorted(pdict.items(), key=lambda x: -x[1])[:3]
                print(f"\n[DBG] {bpath.name} block#{blocks_done}: {top3}")
            elapsed = time.time() - start
            speed = bytes_done / elapsed if elapsed > 0 else 0
            eta = (total_bytes - bytes_done) / speed if speed > 0 else 0
            pct = (bytes_done / total_bytes * 100.0) if total_bytes else 100.0
            sys.stdout.write(
                "\r"
                f"[{bpath.name}] {format_bytes(bytes_done)}/{format_bytes(total_bytes)} "
                f"({pct:5.1f}%) | {blocks_done} blokova | "
                f"{format_bytes(speed)}/s | ETA {eta:5.0f}s"
            )
            sys.stdout.flush()

        sys.stdout.write("\n")
        per_file_stats[bpath.name] = dict(file_stats)
        print(f"[OK] {bpath.name} -> {dict(file_stats)}")
    print("[INFO] Spajam blokove u all.txt po žanru…")
    for g in ALL_SUPERS:
        gdir = out_root / g
        if not gdir.exists():
            continue
        all_out = gdir / "all.txt"
        with all_out.open("w", encoding="utf-8", newline="\n") as fout:
            for f in sorted(gdir.glob("*.block*.txt")):
                try:
                    txt = f.read_text(encoding="utf-8", errors="ignore")
                    fout.write(txt)
                    if not txt.endswith("\n"):
                        fout.write("\n")
                except Exception as e:
                    print(f"[WARN] {f} skip: {e}", file=sys.stderr)
    if args.pack_equal_mb > 0:
        for g in ALL_SUPERS:
            gdir = out_root / g
            if gdir.exists():
                pack_blocks_equal_mb(gdir, args.pack_equal_mb)

    report = {
        "per_genre_blocks": dict(per_genre_counts),
        "per_input_file": per_file_stats,
        "classes": [str(c) for c in getattr(clf, "classes_", [])],
        "conf_thresh": args.conf_thresh,
        "block_chars": args.block_chars,
        "overlap": args.overlap,
        "vote_window": args.vote_window,
        "shots_default": args.shots,
        "CLASS_SHOTS": CLASS_SHOTS,
        "shot_size": args.shot_size,
        "min_df": args.min_df,
        "max_df": args.max_df,
        "max_features": args.max_features,
        "class_weight": args.class_weight,
        "lower": args.lower,
        "random_state": args.random_state,
    }
    (out_root / "_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print("[DONE]")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
