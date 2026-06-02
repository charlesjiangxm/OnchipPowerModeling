#!/usr/bin/env python
"""Submit all YAML regression configs to Slurm with bounded concurrency.

The runner discovers configs under ``configs/``, routes torch-based models to
GPU nodes and the remaining models to CPU nodes, then keeps at most
``--max-active`` jobs from this sweep in Slurm at a time.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
GPU_ALGORITHMS = {"MLP", "FT-Transformer"}
TERMINAL_PLACEHOLDER = "LEFT_QUEUE"
DONE_LOG_RE = re.compile(r"INFO\s+pipeline\s+::\s+done:\s+(.+?report\.md)\s*$")


@dataclass(frozen=True)
class ConfigJob:
    config: Path
    algorithm: str
    partition: str
    needs_gpu: bool


@dataclass(frozen=True)
class FinishedRun:
    run_dir: Path
    report: Path


@dataclass(frozen=True)
class SkippedJob:
    config: Path
    algorithm: str
    run_dir: Path
    report: Path


@dataclass
class SubmittedJob:
    config: str
    algorithm: str
    partition: str
    job_id: str
    sbatch: str
    state: str
    submitted_at: str
    updated_at: str


def main() -> int:
    args = parse_args()
    configs_dir = resolve_repo_path(args.configs_dir)
    output_root = resolve_repo_path(args.output_root)
    python_exe = Path(sys.executable).resolve()
    scheduler_dir = output_root / "slurm_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")

    all_jobs = discover_configs(
        configs_dir=configs_dir,
        gpu_partition=args.gpu_partition,
        cpu_partition=args.cpu_partition,
    )
    if not all_jobs:
        raise SystemExit(f"No YAML configs found under {configs_dir}")

    pending_jobs, skipped_jobs = skip_finished_jobs(all_jobs, output_root)
    unfinished_count = len(pending_jobs)
    if args.limit is not None:
        pending_jobs = pending_jobs[: args.limit]

    print_summary(
        jobs=pending_jobs,
        skipped_jobs=skipped_jobs,
        total_configs=len(all_jobs),
        unfinished_count=unfinished_count,
        args=args,
        python_exe=python_exe,
    )
    print_skipped_jobs(skipped_jobs)
    if not pending_jobs:
        if args.dry_run:
            print("dry run: no sbatch commands will be executed")
        print("no unfinished configs to submit")
        return 0

    if args.dry_run:
        print_dry_run(pending_jobs, args, scheduler_dir, output_root, python_exe)
        return 0

    scheduler_dir.mkdir(parents=True, exist_ok=True)
    (scheduler_dir / "sbatch").mkdir(parents=True, exist_ok=True)
    (scheduler_dir / "logs").mkdir(parents=True, exist_ok=True)

    submitted: list[SubmittedJob] = []
    pending = list(pending_jobs)
    active_ids: set[str] = set()
    manifest_path = scheduler_dir / "manifest.json"

    def submit_until_full() -> None:
        while pending and len(active_ids) < args.max_active:
            job = pending.pop(0)
            submitted_job = submit_job(
                job=job,
                args=args,
                scheduler_dir=scheduler_dir,
                output_root=output_root,
                python_exe=python_exe,
            )
            submitted.append(submitted_job)
            active_ids.add(submitted_job.job_id)
            write_manifest(
                manifest_path=manifest_path,
                scheduler_dir=scheduler_dir,
                jobs=submitted,
                args=args,
                python_exe=python_exe,
            )
            print(
                f"submitted {submitted_job.job_id} "
                f"({submitted_job.partition}) {submitted_job.config}"
            )

    submit_until_full()
    while active_ids:
        print(
            f"active={len(active_ids)} pending_configs={len(pending)} "
            f"submitted={len(submitted)}/{len(pending_jobs)}"
        )
        time.sleep(args.poll_interval)
        states = query_squeue(active_ids)
        now = timestamp()
        for record in submitted:
            if record.job_id not in active_ids:
                continue
            state = states.get(record.job_id, TERMINAL_PLACEHOLDER)
            record.state = state
            record.updated_at = now
        active_ids = set(states)
        submit_until_full()
        write_manifest(
            manifest_path=manifest_path,
            scheduler_dir=scheduler_dir,
            jobs=submitted,
            args=args,
            python_exe=python_exe,
        )

    print(f"all jobs have left the Slurm queue; manifest: {manifest_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit every YAML config under configs/ to Slurm."
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=Path("configs"),
        help="Directory containing YAML configs (default: configs).",
    )
    parser.add_argument(
        "--max-active",
        type=positive_int,
        default=50,
        help="Maximum active jobs from this sweep (default: 50).",
    )
    parser.add_argument(
        "--poll-interval",
        type=positive_int,
        default=180,
        help="Seconds between Slurm queue polls (default: 180).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Parent directory for model outputs and Slurm metadata.",
    )
    parser.add_argument(
        "--time",
        default="1-00:00:00",
        help="SBATCH wall time, e.g. 12:00:00 or 1-00:00:00 (default: 1-00:00:00).",
    )
    parser.add_argument(
        "--gpu-partition",
        default="gpu-share",
        help="Partition for MLP and FT-Transformer configs (default: gpu-share).",
    )
    parser.add_argument(
        "--cpu-partition",
        default="cpu-share",
        help="Partition for non-torch configs (default: cpu-share).",
    )
    parser.add_argument(
        "--gpu-gres",
        default="gpu:1",
        help="GRES request for GPU jobs (default: gpu:1).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned submissions and an example sbatch script without sbatch.",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=None,
        help="Only process the first N configs, useful for smoke tests.",
    )
    return parser.parse_args()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def resolve_repo_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def discover_configs(
    *,
    configs_dir: Path,
    gpu_partition: str,
    cpu_partition: str,
) -> list[ConfigJob]:
    if not configs_dir.is_dir():
        raise FileNotFoundError(f"configs directory not found: {configs_dir}")

    paths = sorted({*configs_dir.glob("*.yaml"), *configs_dir.glob("*.yml")})
    jobs: list[ConfigJob] = []
    for config_path in paths:
        algorithm = read_algorithm(config_path)
        needs_gpu = algorithm in GPU_ALGORITHMS
        jobs.append(
            ConfigJob(
                config=config_path,
                algorithm=algorithm,
                partition=gpu_partition if needs_gpu else cpu_partition,
                needs_gpu=needs_gpu,
            )
        )
    return jobs


def skip_finished_jobs(
    jobs: list[ConfigJob],
    output_root: Path,
) -> tuple[list[ConfigJob], list[SkippedJob]]:
    pending: list[ConfigJob] = []
    skipped: list[SkippedJob] = []
    for job in jobs:
        finished = find_finished_run(job.config, output_root)
        if finished is None:
            pending.append(job)
            continue
        skipped.append(
            SkippedJob(
                config=job.config,
                algorithm=job.algorithm,
                run_dir=finished.run_dir,
                report=finished.report,
            )
        )
    return pending, skipped


def find_finished_run(config_path: Path, output_root: Path) -> FinishedRun | None:
    if not output_root.is_dir():
        return None

    prefix = f"{config_path.stem}_"
    candidates = [
        path for path in output_root.iterdir()
        if path.is_dir() and path.name.startswith(prefix)
    ]
    for run_dir in sorted(candidates, key=lambda path: path.name, reverse=True):
        report = finished_report_from_log(run_dir)
        if report is not None:
            return FinishedRun(run_dir=run_dir, report=report)
    return None


def finished_report_from_log(run_dir: Path) -> Path | None:
    log_path = run_dir / "fit.log"
    if not log_path.is_file():
        return None

    done_report: str | None = None
    with log_path.open(errors="replace") as fh:
        for line in fh:
            match = DONE_LOG_RE.search(line)
            if match:
                done_report = match.group(1).strip()

    if done_report is None:
        return None

    report_path = Path(done_report)
    if not report_path.is_absolute():
        report_path = run_dir / report_path
    if report_path.is_file():
        return report_path

    # Keep runs portable if output directories move after completion.
    local_report = run_dir / "report.md"
    if local_report.is_file() and report_path.name == "report.md":
        return local_report
    return None


def read_algorithm(config_path: Path) -> str:
    with config_path.open() as fh:
        cfg = yaml.safe_load(fh)
    try:
        algorithm = cfg["regression"]["algorithm"]
    except (TypeError, KeyError) as exc:
        raise ValueError(f"{config_path}: missing regression.algorithm") from exc
    if not isinstance(algorithm, str) or not algorithm:
        raise ValueError(f"{config_path}: regression.algorithm must be a non-empty string")
    return algorithm


def print_summary(
    *,
    jobs: list[ConfigJob],
    skipped_jobs: list[SkippedJob],
    total_configs: int,
    unfinished_count: int,
    args: argparse.Namespace,
    python_exe: Path,
) -> None:
    gpu_count = sum(job.needs_gpu for job in jobs)
    cpu_count = len(jobs) - gpu_count
    print(f"repo root: {REPO_ROOT}")
    print(f"python: {python_exe}")
    print(
        f"configs: {total_configs} total; {len(skipped_jobs)} finished skipped; "
        f"{unfinished_count} unfinished"
    )
    if args.limit is not None:
        print(f"limit: first {len(jobs)} unfinished config(s)")
    print(f"to submit: {len(jobs)} total ({gpu_count} gpu, {cpu_count} cpu)")
    print(f"max active jobs: {args.max_active}")


def print_skipped_jobs(skipped_jobs: list[SkippedJob], max_items: int = 20) -> None:
    if not skipped_jobs:
        return

    print(f"skipped already-finished configs: {len(skipped_jobs)}")
    for skipped in skipped_jobs[:max_items]:
        rel_config = display_path(skipped.config)
        rel_run_dir = display_path(skipped.run_dir)
        print(f"  skip {rel_config} (finished: {rel_run_dir})")
    remaining = len(skipped_jobs) - max_items
    if remaining > 0:
        print(f"  ... {remaining} more skipped")


def display_path(path: Path) -> Path:
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


def print_dry_run(
    jobs: list[ConfigJob],
    args: argparse.Namespace,
    scheduler_dir: Path,
    output_root: Path,
    python_exe: Path,
) -> None:
    print("dry run: no sbatch commands will be executed")
    for job in jobs:
        rel_config = job.config.relative_to(REPO_ROOT)
        gres = f" --gres={args.gpu_gres}" if job.needs_gpu else ""
        print(f"{rel_config} -> -p {job.partition}{gres}")

    first = jobs[0]
    script = render_sbatch(
        job=first,
        args=args,
        scheduler_dir=scheduler_dir,
        output_root=output_root,
        python_exe=python_exe,
    )
    print("\nexample sbatch script for first config:\n")
    print(script)


def submit_job(
    *,
    job: ConfigJob,
    args: argparse.Namespace,
    scheduler_dir: Path,
    output_root: Path,
    python_exe: Path,
) -> SubmittedJob:
    sbatch_path = scheduler_dir / "sbatch" / f"{safe_stem(job.config)}.sbatch"
    sbatch_path.write_text(
        render_sbatch(
            job=job,
            args=args,
            scheduler_dir=scheduler_dir,
            output_root=output_root,
            python_exe=python_exe,
        )
    )

    result = subprocess.run(
        ["sbatch", str(sbatch_path)],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"sbatch failed for {job.config}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    job_id = parse_sbatch_job_id(result.stdout)
    now = timestamp()
    return SubmittedJob(
        config=str(job.config.relative_to(REPO_ROOT)),
        algorithm=job.algorithm,
        partition=job.partition,
        job_id=job_id,
        sbatch=str(sbatch_path),
        state="SUBMITTED",
        submitted_at=now,
        updated_at=now,
    )


def render_sbatch(
    *,
    job: ConfigJob,
    args: argparse.Namespace,
    scheduler_dir: Path,
    output_root: Path,
    python_exe: Path,
) -> str:
    rel_config = job.config.relative_to(REPO_ROOT)
    log_dir = scheduler_dir / "logs"
    log_base = safe_stem(job.config)
    job_name = safe_job_name(f"opm_{job.config.stem}")
    python_dir = python_exe.parent
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if not conda_prefix and python_dir.name == "bin":
        conda_prefix = str(python_dir.parent)

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={log_dir / (log_base + '.%j.out')}",
        f"#SBATCH --error={log_dir / (log_base + '.%j.err')}",
        f"#SBATCH -p {job.partition}",
        f"#SBATCH --time={args.time}",
    ]
    if job.needs_gpu:
        lines.append(f"#SBATCH --gres={args.gpu_gres}")

    lines.extend(
        [
            "",
            "set -euo pipefail",
            f"cd {shlex.quote(str(REPO_ROOT))}",
            "export PYTHONUNBUFFERED=1",
            f"export PATH={shlex.quote(str(python_dir))}:$PATH",
        ]
    )
    if conda_prefix:
        lines.append(f"export CONDA_PREFIX={shlex.quote(conda_prefix)}")

    lines.append(
        " ".join(
            [
                shlex.quote(str(python_exe)),
                shlex.quote(str(REPO_ROOT / "script" / "run_fit.py")),
                "--config",
                shlex.quote(str(rel_config)),
                "--output-root",
                shlex.quote(str(output_root)),
            ]
        )
    )
    return "\n".join(lines) + "\n"


def parse_sbatch_job_id(stdout: str) -> str:
    match = re.search(r"Submitted batch job\s+(\d+)", stdout)
    if not match:
        raise RuntimeError(f"Could not parse sbatch job id from: {stdout!r}")
    return match.group(1)


def query_squeue(job_ids: Iterable[str]) -> dict[str, str]:
    ids = sorted(set(job_ids), key=int)
    if not ids:
        return {}
    result = subprocess.run(
        ["squeue", "-h", "-j", ",".join(ids), "-o", "%i|%T"],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"squeue failed for job ids {','.join(ids)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    states: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        job_id, state = line.split("|", 1)
        states[job_id.strip()] = state.strip()
    return states


def write_manifest(
    *,
    manifest_path: Path,
    scheduler_dir: Path,
    jobs: list[SubmittedJob],
    args: argparse.Namespace,
    python_exe: Path,
) -> None:
    payload = {
        "created_by": "script/run_all.py",
        "updated_at": timestamp(),
        "repo_root": str(REPO_ROOT),
        "scheduler_dir": str(scheduler_dir),
        "python": str(python_exe),
        "max_active": args.max_active,
        "poll_interval": args.poll_interval,
        "jobs": [asdict(job) for job in jobs],
    }
    tmp_path = manifest_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(manifest_path)


def safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)


def safe_job_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return cleaned[:120]


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
