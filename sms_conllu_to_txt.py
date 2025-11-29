# sms_conllu_to_txt.py
import sys
import gzip
from pathlib import Path

def open_maybe_gzip(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")

def conllu_to_txt(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open_maybe_gzip(src) as fin, dst.open("w", encoding="utf-8") as fout:
        tokens = []
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                
                if tokens:
                    fout.write(" ".join(tokens) + "\n")
                    tokens = []
                continue
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            form = parts[1].strip()
            if form:
                tokens.append(form)
     
        if tokens:
            fout.write(" ".join(tokens) + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sms_conllu_to_txt.py input.conllu[.gz] output.txt")
        sys.exit(1)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    conllu_to_txt(src, dst)
