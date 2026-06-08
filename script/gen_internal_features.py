#!/usr/bin/env python
"""Generate internal net-only feature pickles for a block.

For a block such as ``cp0``, this script reads matching pickle files from
``db/cp0/cp0_input``, ``db/cp0/cp0_net``, and ``db/cp0/cp0_output``. It drops
from each net dataframe any columns that also appear in the input or output
dataframes, then writes the result to ``db/cp0/cp0_internal``.

Example:
python script/gen_internal_features.py cp0

"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
BLOCK_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create internal net-only feature pickles for one db block."
    )
    parser.add_argument(
        "block",
        help="Block name under db/, for example: cp0",
    )
    parser.add_argument(
        "--db-root",
        type=Path,
        default=REPO_ROOT / "db",
        help="Root directory containing block folders (default: <repo>/db).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing files in <block>_internal. By default they are skipped.",
    )
    return parser.parse_args()


def validate_block(block: str) -> str:
    if not BLOCK_RE.fullmatch(block):
        raise SystemExit(
            "block must contain only letters, numbers, underscores, or hyphens"
        )
    return block


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise SystemExit(f"{label} directory not found: {path}")


def read_dataframe(path: Path) -> pd.DataFrame:
    obj = pd.read_pickle(path)
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"expected pandas DataFrame in {path}, got {type(obj).__name__}")
    return obj


def matching_net_pickles(input_dir: Path, net_dir: Path, output_dir: Path) -> list[Path]:
    net_pickles = sorted(net_dir.glob("*.pkl"))
    if not net_pickles:
        raise SystemExit(f"no .pkl files found in net directory: {net_dir}")

    missing: list[str] = []
    for net_path in net_pickles:
        if not (input_dir / net_path.name).is_file():
            missing.append(str(input_dir / net_path.name))
        if not (output_dir / net_path.name).is_file():
            missing.append(str(output_dir / net_path.name))

    if missing:
        formatted = "\n".join(f"  {path}" for path in missing)
        raise SystemExit(f"missing matching input/output pickle files:\n{formatted}")

    return net_pickles


def process_pickle(
    net_path: Path,
    input_dir: Path,
    output_dir: Path,
    internal_dir: Path,
    overwrite: bool,
) -> tuple[str, int, int, int, bool]:
    output_path = internal_dir / net_path.name
    if output_path.exists() and not overwrite:
        return net_path.name, 0, 0, 0, True

    input_df = read_dataframe(input_dir / net_path.name)
    net_df = read_dataframe(net_path)
    output_df = read_dataframe(output_dir / net_path.name)

    excluded_cols = set(input_df.columns) | set(output_df.columns)
    keep_cols = [col for col in net_df.columns if col not in excluded_cols]
    internal_df = net_df.loc[:, keep_cols]
    internal_df.to_pickle(output_path)

    dropped_count = len(net_df.columns) - len(keep_cols)
    return net_path.name, len(net_df.columns), dropped_count, len(keep_cols), False


def main() -> int:
    args = parse_args()
    block = validate_block(args.block)
    db_root = args.db_root.resolve()
    block_dir = db_root / block
    input_dir = block_dir / f"{block}_input"
    net_dir = block_dir / f"{block}_net"
    output_dir = block_dir / f"{block}_output"
    internal_dir = block_dir / f"{block}_internal"

    require_dir(block_dir, "block")
    require_dir(input_dir, "input")
    require_dir(net_dir, "net")
    require_dir(output_dir, "output")

    net_pickles = matching_net_pickles(input_dir, net_dir, output_dir)
    internal_dir.mkdir(parents=False, exist_ok=True)

    written = 0
    skipped = 0
    print(f"block: {block}")
    print(f"output: {internal_dir}")
    for net_path in net_pickles:
        name, net_cols, dropped_cols, internal_cols, was_skipped = process_pickle(
            net_path=net_path,
            input_dir=input_dir,
            output_dir=output_dir,
            internal_dir=internal_dir,
            overwrite=args.overwrite,
        )
        if was_skipped:
            skipped += 1
            print(f"{name}: skipped existing output")
            continue

        written += 1
        print(
            f"{name}: net_cols={net_cols} "
            f"dropped={dropped_cols} internal_cols={internal_cols}"
        )

    print(f"done: wrote={written} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
