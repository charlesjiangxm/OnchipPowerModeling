#!/usr/bin/env python
"""Generate all feature pickles for a block.

For a block such as ``cp0``, this script reads matching pickle files from
``db/cp0/cp0_input``, ``db/cp0/cp0_internal``, and ``db/cp0/cp0_output``. It
concatenates their columns in input/internal/output order, then writes the
result to ``db/cp0/cp0_all``.

Example:
python script/gen_all_features.py cp0

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
        description="Create combined input/internal/output feature pickles for one db block."
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
        help="Replace existing files in <block>_all. By default they are skipped.",
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


def matching_feature_pickles(
    input_dir: Path, internal_dir: Path, output_dir: Path
) -> list[Path]:
    input_pickles = sorted(input_dir.glob("*.pkl"))
    if not input_pickles:
        raise SystemExit(f"no .pkl files found in input directory: {input_dir}")

    paths_by_label = {
        "input": input_dir,
        "internal": internal_dir,
        "output": output_dir,
    }
    names_by_label = {
        label: {path.name for path in directory.glob("*.pkl")}
        for label, directory in paths_by_label.items()
    }
    expected_names = names_by_label["input"]

    mismatches: list[str] = []
    for label, names in names_by_label.items():
        missing = sorted(expected_names - names)
        extra = sorted(names - expected_names)
        if missing:
            mismatches.extend(
                f"  missing from {label}: {name}" for name in missing
            )
        if extra:
            mismatches.extend(f"  extra in {label}: {name}" for name in extra)

    if mismatches:
        formatted = "\n".join(mismatches)
        raise SystemExit(f"mismatched input/internal/output pickle files:\n{formatted}")

    return input_pickles


def require_matching_indexes(
    name: str,
    input_df: pd.DataFrame,
    internal_df: pd.DataFrame,
    output_df: pd.DataFrame,
) -> None:
    if input_df.index.equals(internal_df.index) and input_df.index.equals(output_df.index):
        return

    raise ValueError(
        f"row indexes do not match for {name}; cannot concatenate feature columns safely"
    )


def require_unique_columns(name: str, all_df: pd.DataFrame) -> None:
    duplicated = all_df.columns[all_df.columns.duplicated()].unique()
    if len(duplicated) == 0:
        return

    sample = ", ".join(repr(col) for col in duplicated[:10])
    suffix = "" if len(duplicated) <= 10 else ", ..."
    raise ValueError(f"duplicate columns after concatenating {name}: {sample}{suffix}")


def process_pickle(
    input_path: Path,
    internal_dir: Path,
    output_dir: Path,
    all_dir: Path,
    overwrite: bool,
) -> tuple[str, int, int, int, int, bool]:
    output_path = all_dir / input_path.name
    if output_path.exists() and not overwrite:
        return input_path.name, 0, 0, 0, 0, True

    input_df = read_dataframe(input_path)
    internal_df = read_dataframe(internal_dir / input_path.name)
    output_df = read_dataframe(output_dir / input_path.name)

    require_matching_indexes(input_path.name, input_df, internal_df, output_df)

    all_df = pd.concat([input_df, internal_df, output_df], axis=1)
    require_unique_columns(input_path.name, all_df)
    all_df.to_pickle(output_path)

    return (
        input_path.name,
        len(input_df.columns),
        len(internal_df.columns),
        len(output_df.columns),
        len(all_df.columns),
        False,
    )


def main() -> int:
    args = parse_args()
    block = validate_block(args.block)
    db_root = args.db_root.resolve()
    block_dir = db_root / block
    input_dir = block_dir / f"{block}_input"
    internal_dir = block_dir / f"{block}_internal"
    output_dir = block_dir / f"{block}_output"
    all_dir = block_dir / f"{block}_all"

    require_dir(block_dir, "block")
    require_dir(input_dir, "input")
    require_dir(internal_dir, "internal")
    require_dir(output_dir, "output")

    input_pickles = matching_feature_pickles(input_dir, internal_dir, output_dir)
    all_dir.mkdir(parents=False, exist_ok=True)

    written = 0
    skipped = 0
    print(f"block: {block}")
    print(f"output: {all_dir}")
    for input_path in input_pickles:
        name, input_cols, internal_cols, output_cols, all_cols, was_skipped = (
            process_pickle(
                input_path=input_path,
                internal_dir=internal_dir,
                output_dir=output_dir,
                all_dir=all_dir,
                overwrite=args.overwrite,
            )
        )
        if was_skipped:
            skipped += 1
            print(f"{name}: skipped existing output")
            continue

        written += 1
        print(
            f"{name}: input_cols={input_cols} internal_cols={internal_cols} "
            f"output_cols={output_cols} all_cols={all_cols}"
        )

    print(f"done: wrote={written} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
