import argparse
from pathlib import Path


def merge_files(in_dir: Path, out_path: Path, pattern: str) -> None:
    in_dir = in_dir.resolve()
    files = sorted(in_dir.glob(pattern))

    if not files:
        raise SystemExit(f"No files matching pattern '{pattern}' in {in_dir}")

    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fout:
        for path in files:
            with path.open("r", encoding="utf-8", errors="ignore") as fin:
                for line in fin:
                    fout.write(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True, help="Input directory with genre files")
    parser.add_argument("--out", required=True, help="Output merged corpus path")
    parser.add_argument(
        "--pattern",
        default="*.txt",
        help="Glob pattern for input files (default: *.txt)",
    )
    args = parser.parse_args()

    merge_files(Path(args.in_dir), Path(args.out), args.pattern)


if __name__ == "__main__":
    main()
