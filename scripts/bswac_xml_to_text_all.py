from pathlib import Path
import gzip
import re

def open_maybe_gzip(path: Path):
    """Otvori .xml ili .xml.gz fajl transparentno."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


def extract_plain_text(in_path: Path, out_path: Path):
    """
    Iz bsWaC XML/vert fajla izvuče plain tekst:
    - ignoriše XML tagove (<doc>, <p>, <s>...)
    - čita linije sa tokenima (prva kolona = riječ)
    - pravi rečenice (jedan red = jedna rečenica)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open_maybe_gzip(in_path) as fin, out_path.open("w", encoding="utf-8") as fout:
        sentence_tokens = []

        for line in fin:
            line = line.strip()
            if not line:
                if sentence_tokens:
                    fout.write(" ".join(sentence_tokens) + "\n")
                    sentence_tokens = []
                continue
            if line.startswith("<") and line.endswith(">"):
                if sentence_tokens and (
                    line.startswith("</s") or
                    line.startswith("</p") or
                    line.startswith("</text") or
                    line.startswith("</doc")
                ):
                    fout.write(" ".join(sentence_tokens) + "\n")
                    sentence_tokens = []
                continue
            parts = re.split(r"\s+", line)
            if not parts:
                continue

            token = parts[0]
            if token.startswith("<"):
                continue

            sentence_tokens.append(token)
        if sentence_tokens:
            fout.write(" ".join(sentence_tokens) + "\n")

    print(f"Gotovo: {in_path.name} -> {out_path}")


def main():
  
    base_dir = Path(r"e:\data\bswac")
    input_files = [
        base_dir / "bsWaC1.1.01.xml",
        base_dir / "bsWaC1.1.02.xml",
        base_dir / "bsWaC1.1.03.xml",
    ]

    out_dir = base_dir / "plain" 

    for in_path in input_files:
        if not in_path.exists():
            print(f"UPOZORENJE: ne postoji fajl {in_path}")
            continue

        out_name = in_path.name.replace(".xml.gz", ".txt").replace(".xml", ".txt")
        out_path = out_dir / out_name

        extract_plain_text(in_path, out_path)


if __name__ == "__main__":
    main()
