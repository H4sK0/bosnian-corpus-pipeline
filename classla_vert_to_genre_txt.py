#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, gzip, re
from pathlib import Path

def open_maybe_gzip(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")

META_RE = re.compile(r'(\w+)="([^"]+)"')

def parse_genre_from_text_tag(line: str) -> str:
    """
    Pokušaj izvući žanr / domen iz <text ...> linije.
    Gleda attribute: genre=, type=, subcorpus=, domain=...
    Ako ništa ne nađe, vrati 'unknown'.
    """
    attrs = dict(META_RE.findall(line))
    for key in ("genre", "type", "subcorpus", "domain", "source"):
        if key in attrs:
            g = attrs[key].strip().lower()
            g = re.sub(r"[^a-z0-9_]+", "_", g)
            g = re.sub(r"_+", "_", g).strip("_")
            if not g:
                continue
            return g
    return "unknown"

def main():
    ap = argparse.ArgumentParser("CLASSLA .vert → TXT po žanrovima")
    ap.add_argument("--vert", required=True, help="putanja do CLASSLA .vert ili .vert.gz")
    ap.add_argument("--out", required=True, help="izlazni folder")
    ap.add_argument("--token-col", type=int, default=0, help="0-based indeks kolone sa tokenom (default 0)")
    ap.add_argument("--shard-sents", type=int, default=0, help="rečenica po shardu (0 = bez shardova, jedan fajl po žanru)")
    ap.add_argument("--prefix", default="classla_bs", help="prefiks izlaznih fajlova")
    args = ap.parse_args()

    vert_path = Path(args.vert)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    current_genre = "unknown"
    buf_tokens = []
    sent_count_per_genre = {}
    handles = {}  

    def get_handle(genre: str):
        shard_id = 0
        key = (genre, shard_id)
        if key in handles:
            return handles[key]
        genre_dir = out_root / f"genre_{genre}"
        genre_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{args.prefix}_{genre}_{shard_id:04d}.txt"
        h = open(genre_dir / fname, "w", encoding="utf-8")
        handles[key] = h
        sent_count_per_genre.setdefault(key, 0)
        return h

    def flush_sentence():
        nonlocal buf_tokens
        if not buf_tokens:
            return
        sent = " ".join(buf_tokens).strip()
        if not sent:
            buf_tokens = []
            return
        h = get_handle(current_genre)
        h.write(sent + "\n")

        key = (current_genre, 0)
        sent_count_per_genre[key] = sent_count_per_genre.get(key, 0) + 1
        buf_tokens = []

    with open_maybe_gzip(vert_path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("<"):
                if line.startswith("<text"):
                    current_genre = parse_genre_from_text_tag(line)
                elif line.startswith("</s"):
                    flush_sentence()
                continue

            parts = line.split("\t")
            if not parts:
                continue
            tok = parts[args.token_col] if len(parts) > args.token_col else parts[0]
            tok = tok.strip()
            if tok:
                buf_tokens.append(tok)
    flush_sentence()
    for h in handles.values():
        h.close()

    print("Gotovo. Rečenica po žanru:")
    for (genre, shard), n in sorted(sent_count_per_genre.items()):
        print(f"  {genre}: {n} rečenica")

if __name__ == "__main__":
    main()
