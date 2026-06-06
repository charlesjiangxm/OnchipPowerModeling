#!/usr/bin/env python
"""Submit unfinished YAML regression configs as node-scoped Slurm workers.

The runner discovers configs under ``configs/`` and assigns them across four
user-configurable CPU/GPU workers. Each worker is one Slurm job on a single node
and runs up to ``--jobs-per-node`` configs concurrently.

Example - CPU only:
python script/run_all.py --node-partitions cpu-share,cpu-share,cpu-share,cpu-share --jobs-per-node 2

Example - CPU & GPU:
python script/run_all.py --node-partitions cpu-share,cpu-share,gpu-share,gpu-share --gpu-algorithms "MLP,FT-Transformer"

"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
# GPU_ALGORITHMS = {"MLP", "FT-Transformer"}
GPU_ALGORITHMS: set[str] = set()
CPU_PARTITION = "cpu-share"
GPU_PARTITION = "gpu-share"
GPU_GRES = "gpu:1"
NODE_COUNT = 4
DEFAULT_NODE_PARTITIONS = (
    GPU_PARTITION,
    GPU_PARTITION,
    CPU_PARTITION,
    CPU_PARTITION,
)
DEFAULT_JOBS_PER_NODE = 1
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


@dataclass(frozen=True)
class WorkerBatch:
    name: str
    partition: str
    needs_gpu: bool
    index: int
    jobs: list[ConfigJob]


@dataclass
class SubmittedBatch:
    name: str
    partition: str
    needs_gpu: bool
    index: int
    assigned_count: int
    assigned_configs: list[str]
    job_id: str
    sbatch: str
    submitted_at: str


def main() -> int:
    args = parse_args()
    configs_dir = resolve_repo_path(args.configs_dir)
    output_root = resolve_repo_path(args.output_root)
    python_exe = Path(sys.executable).resolve()
    scheduler_dir = output_root / "slurm_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")

    all_jobs = discover_configs(
        configs_dir=configs_dir,
        gpu_algorithms=args.gpu_algorithms,
        node_partitions=args.node_partitions,
    )
    if not all_jobs:
        raise SystemExit(f"No YAML configs found under {configs_dir}")

    pending_jobs, skipped_jobs = skip_finished_jobs(all_jobs, output_root)
    unfinished_count = len(pending_jobs)
    batches = build_worker_batches(
        jobs=pending_jobs,
        node_partitions=args.node_partitions,
    )

    print_summary(
        batches=batches,
        skipped_jobs=skipped_jobs,
        total_configs=len(all_jobs),
        unfinished_count=unfinished_count,
        python_exe=python_exe,
        jobs_per_node=args.jobs_per_node,
        node_partitions=args.node_partitions,
        gpu_algorithms=args.gpu_algorithms,
    )
    print_skipped_jobs(skipped_jobs)
    if not pending_jobs:
        if args.dry_run:
            print("dry run: no sbatch commands will be executed")
        print("no unfinished configs to submit")
        return 0

    if args.dry_run:
        print_dry_run(batches, args, scheduler_dir, output_root, python_exe)
        return 0

    scheduler_dir.mkdir(parents=True, exist_ok=True)
    (scheduler_dir / "sbatch").mkdir(parents=True, exist_ok=True)
    (scheduler_dir / "logs").mkdir(parents=True, exist_ok=True)

    submitted: list[SubmittedBatch] = []
    manifest_path = scheduler_dir / "manifest.json"
    for batch in batches:
        if not batch.jobs:
            continue
        submitted_batch = submit_batch(
            batch=batch,
            args=args,
            scheduler_dir=scheduler_dir,
            output_root=output_root,
            python_exe=python_exe,
        )
        submitted.append(submitted_batch)
        write_manifest(
            manifest_path=manifest_path,
            scheduler_dir=scheduler_dir,
            batches=submitted,
            args=args,
            python_exe=python_exe,
        )
        print(
            f"submitted {submitted_batch.job_id} "
            f"({submitted_batch.partition}) {submitted_batch.name}: "
            f"{submitted_batch.assigned_count} config(s)"
        )

    print(f"submitted {len(submitted)} worker job(s); manifest: {manifest_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit unfinished YAML configs as four node-scoped Slurm workers."
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=Path("configs"),
        help="Directory containing YAML configs (default: configs).",
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
        "--dry-run",
        action="store_true",
        help="Print planned submissions and an example sbatch script without sbatch.",
    )
    parser.add_argument(
        "--jobs-per-node",
        type=positive_int,
        default=DEFAULT_JOBS_PER_NODE,
        help=(
            "Maximum configs to run concurrently inside each worker "
            f"(default: {DEFAULT_JOBS_PER_NODE})."
        ),
    )
    parser.add_argument(
        "--node-partitions",
        type=parse_node_partitions,
        default=DEFAULT_NODE_PARTITIONS,
        help=(
            "Comma-separated partitions for the four worker nodes. Each entry "
            f"must be {CPU_PARTITION!r} or {GPU_PARTITION!r} "
            f"(default: {','.join(DEFAULT_NODE_PARTITIONS)})."
        ),
    )
    parser.add_argument(
        "--gpu-algorithms",
        type=parse_gpu_algorithms,
        default=set(GPU_ALGORITHMS),
        help=(
            "Comma-separated regression.algorithm names to route to GPU nodes "
            "when both CPU and GPU nodes are configured, e.g. "
            "MLP,FT-Transformer. Defaults to the GPU_ALGORITHMS constant."
        ),
    )
    return parser.parse_args()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def parse_node_partitions(value: str) -> tuple[str, ...]:
    partitions = tuple(part.strip() for part in value.split(","))
    if len(partitions) != NODE_COUNT:
        raise argparse.ArgumentTypeError(
            f"must contain exactly {NODE_COUNT} comma-separated partitions"
        )
    if any(not partition for partition in partitions):
        raise argparse.ArgumentTypeError("partitions must not be empty")

    allowed = {CPU_PARTITION, GPU_PARTITION}
    invalid = sorted(set(partitions) - allowed)
    if invalid:
        allowed_text = ", ".join(sorted(allowed))
        invalid_text = ", ".join(invalid)
        raise argparse.ArgumentTypeError(
            f"invalid partition(s): {invalid_text}; allowed: {allowed_text}"
        )
    return partitions


def parse_gpu_algorithms(value: str) -> set[str]:
    if not value.strip():
        return set()

    algorithms = [algorithm.strip() for algorithm in value.split(",")]
    if any(not algorithm for algorithm in algorithms):
        raise argparse.ArgumentTypeError("algorithm names must not be empty")
    return set(algorithms)


def resolve_repo_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def discover_configs(
    *,
    configs_dir: Path,
    gpu_algorithms: set[str],
    node_partitions: tuple[str, ...],
) -> list[ConfigJob]:
    if not configs_dir.is_dir():
        raise FileNotFoundError(f"configs directory not found: {configs_dir}")

    paths = sorted({*configs_dir.glob("*.yaml"), *configs_dir.glob("*.yml")})
    jobs: list[ConfigJob] = []
    for config_path in paths:
        algorithm = read_algorithm(config_path)
        needs_gpu = route_to_gpu(
            algorithm=algorithm,
            gpu_algorithms=gpu_algorithms,
            node_partitions=node_partitions,
        )
        jobs.append(
            ConfigJob(
                config=config_path,
                algorithm=algorithm,
                partition=GPU_PARTITION if needs_gpu else CPU_PARTITION,
                needs_gpu=needs_gpu,
            )
        )
    return jobs


def route_to_gpu(
    *,
    algorithm: str,
    gpu_algorithms: set[str],
    node_partitions: tuple[str, ...],
) -> bool:
    mode = gpu_routing_mode(node_partitions)
    if mode == "all":
        return True
    if mode == "none":
        return False
    return algorithm in gpu_algorithms


def gpu_routing_mode(node_partitions: tuple[str, ...]) -> str:
    has_gpu_node = GPU_PARTITION in node_partitions
    has_cpu_node = CPU_PARTITION in node_partitions
    if has_gpu_node and not has_cpu_node:
        return "all"
    if has_cpu_node and not has_gpu_node:
        return "none"
    return "selected"


def describe_gpu_routing(
    *,
    gpu_algorithms: set[str],
    node_partitions: tuple[str, ...],
) -> str:
    mode = gpu_routing_mode(node_partitions)
    if mode == "all":
        return f"all algorithms (no {CPU_PARTITION} nodes configured)"
    if mode == "none":
        return f"none (no {GPU_PARTITION} nodes configured)"
    if not gpu_algorithms:
        return "(none)"
    return ", ".join(sorted(gpu_algorithms))


def build_worker_batches(
    *,
    jobs: list[ConfigJob],
    node_partitions: tuple[str, ...],
) -> list[WorkerBatch]:
    buckets: list[list[ConfigJob]] = [[] for _ in node_partitions]
    node_indexes = {
        True: [
            idx for idx, partition in enumerate(node_partitions)
            if partition == GPU_PARTITION
        ],
        False: [
            idx for idx, partition in enumerate(node_partitions)
            if partition == CPU_PARTITION
        ],
    }
    assigned_counts = {True: 0, False: 0}

    for job in jobs:
        indexes = node_indexes[job.needs_gpu]
        if not indexes:
            raise ValueError(f"no worker node available for {job.partition}")
        bucket_idx = indexes[assigned_counts[job.needs_gpu] % len(indexes)]
        buckets[bucket_idx].append(job)
        assigned_counts[job.needs_gpu] += 1

    return [
        WorkerBatch(
            name=f"node-{idx + 1}-{'gpu' if partition == GPU_PARTITION else 'cpu'}",
            partition=partition,
            needs_gpu=partition == GPU_PARTITION,
            index=idx + 1,
            jobs=bucket,
        )
        for idx, (partition, bucket) in enumerate(zip(node_partitions, buckets))
    ]


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
    batches: list[WorkerBatch],
    skipped_jobs: list[SkippedJob],
    total_configs: int,
    unfinished_count: int,
    python_exe: Path,
    jobs_per_node: int,
    node_partitions: tuple[str, ...],
    gpu_algorithms: set[str],
) -> None:
    gpu_count = sum(len(batch.jobs) for batch in batches if batch.needs_gpu)
    cpu_count = sum(len(batch.jobs) for batch in batches if not batch.needs_gpu)
    worker_count = sum(1 for batch in batches if batch.jobs)
    configured_gpu_workers = sum(
        1 for partition in node_partitions if partition == GPU_PARTITION
    )
    configured_cpu_workers = sum(
        1 for partition in node_partitions if partition == CPU_PARTITION
    )
    gpu_routing_text = describe_gpu_routing(
        gpu_algorithms=gpu_algorithms,
        node_partitions=node_partitions,
    )
    print(f"repo root: {REPO_ROOT}")
    print(f"python: {python_exe}")
    print(
        f"configs: {total_configs} total; {len(skipped_jobs)} finished skipped; "
        f"{unfinished_count} unfinished"
    )
    print(f"pending pools: {gpu_count} gpu config(s), {cpu_count} cpu config(s)")
    print(
        f"worker slots: {configured_gpu_workers} {GPU_PARTITION} + "
        f"{configured_cpu_workers} {CPU_PARTITION}; {jobs_per_node} jobs/worker"
    )
    print(f"node layout: {', '.join(node_partitions)}")
    print(f"gpu routing: {gpu_routing_text}")
    print(f"worker jobs to submit: {worker_count}")
    for batch in batches:
        gres = f", gres={GPU_GRES}" if batch.needs_gpu else ""
        print(
            f"  {batch.name}: {len(batch.jobs)} config(s) "
            f"on {batch.partition}{gres}"
        )


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
    batches: list[WorkerBatch],
    args: argparse.Namespace,
    scheduler_dir: Path,
    output_root: Path,
    python_exe: Path,
) -> None:
    print("dry run: no sbatch commands will be executed")
    for batch in batches:
        action = "submit" if batch.jobs else "skip empty"
        gres = f" --gres={GPU_GRES}" if batch.needs_gpu else ""
        print(
            f"{action}: {batch.name} -> -p {batch.partition}{gres} "
            f"({len(batch.jobs)} config(s))"
        )
        for job in batch.jobs[:5]:
            rel_config = display_path(job.config)
            print(f"  {rel_config} ({job.algorithm})")
        remaining = len(batch.jobs) - 5
        if remaining > 0:
            print(f"  ... {remaining} more")

    first = next((batch for batch in batches if batch.jobs), None)
    if first is None:
        return

    script = render_sbatch(
        batch=first,
        args=args,
        scheduler_dir=scheduler_dir,
        output_root=output_root,
        python_exe=python_exe,
    )
    print(f"\nexample sbatch script for {first.name}:\n")
    print(script)


def submit_batch(
    *,
    batch: WorkerBatch,
    args: argparse.Namespace,
    scheduler_dir: Path,
    output_root: Path,
    python_exe: Path,
) -> SubmittedBatch:
    sbatch_path = scheduler_dir / "sbatch" / f"{safe_job_name(batch.name)}.sbatch"
    sbatch_path.write_text(
        render_sbatch(
            batch=batch,
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
            f"sbatch failed for {batch.name}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    job_id = parse_sbatch_job_id(result.stdout)
    now = timestamp()
    return SubmittedBatch(
        name=batch.name,
        partition=batch.partition,
        needs_gpu=batch.needs_gpu,
        index=batch.index,
        assigned_count=len(batch.jobs),
        assigned_configs=[str(display_path(job.config)) for job in batch.jobs],
        job_id=job_id,
        sbatch=str(sbatch_path),
        submitted_at=now,
    )


def render_sbatch(
    *,
    batch: WorkerBatch,
    args: argparse.Namespace,
    scheduler_dir: Path,
    output_root: Path,
    python_exe: Path,
) -> str:
    log_dir = scheduler_dir / "logs"
    log_base = safe_job_name(batch.name)
    job_name = safe_job_name(f"opm_{batch.name}")
    python_dir = python_exe.parent
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if not conda_prefix and python_dir.name == "bin":
        conda_prefix = str(python_dir.parent)

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={log_dir / (log_base + '.%j.out')}",
        f"#SBATCH --error={log_dir / (log_base + '.%j.err')}",
        f"#SBATCH -p {batch.partition}",
        "#SBATCH --nodes=1",
        f"#SBATCH --ntasks={args.jobs_per_node}",
        f"#SBATCH --time={args.time}",
    ]
    if batch.needs_gpu:
        lines.append(f"#SBATCH --gres={GPU_GRES}")

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

    lines.extend(["", "configs=("])
    for job in batch.jobs:
        lines.append(f"  {shlex.quote(str(display_path(job.config)))}")
    lines.extend(
        [
            ")",
            "",
            f"max_parallel={args.jobs_per_node}",
            "active=0",
            "status=0",
            "",
            "run_config() {",
            "  local config=\"$1\"",
            "  echo \"[$(date --iso-8601=seconds)] start ${config}\"",
            "  if "
            + " ".join(
                [
                    shlex.quote(str(python_exe)),
                    shlex.quote(str(REPO_ROOT / "script" / "run_fit.py")),
                    "--config",
                    "\"$config\"",
                    "--output-root",
                    shlex.quote(str(output_root)),
                ]
            )
            + "; then",
            "    echo \"[$(date --iso-8601=seconds)] done ${config}\"",
            "  else",
            "    rc=$?",
            "    echo \"[$(date --iso-8601=seconds)] failed ${config} rc=${rc}\" >&2",
            "    return \"${rc}\"",
            "  fi",
            "}",
            "",
            "for config in \"${configs[@]}\"; do",
            "  run_config \"${config}\" &",
            "  active=$((active + 1))",
            "  echo \"launched ${config}; running=$(jobs -rp | wc -l)\"",
            "  if (( active >= max_parallel )); then",
            "    if ! wait -n; then",
            "      status=1",
            "    fi",
            "    active=$((active - 1))",
            "  fi",
            "done",
            "",
            "while (( active > 0 )); do",
            "  if ! wait -n; then",
            "    status=1",
            "  fi",
            "  active=$((active - 1))",
            "done",
            "",
            "exit \"${status}\"",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_sbatch_job_id(stdout: str) -> str:
    match = re.search(r"Submitted batch job\s+(\d+)", stdout)
    if not match:
        raise RuntimeError(f"Could not parse sbatch job id from: {stdout!r}")
    return match.group(1)


def write_manifest(
    *,
    manifest_path: Path,
    scheduler_dir: Path,
    batches: list[SubmittedBatch],
    args: argparse.Namespace,
    python_exe: Path,
) -> None:
    payload = {
        "created_by": "script/run_all.py",
        "updated_at": timestamp(),
        "repo_root": str(REPO_ROOT),
        "scheduler_dir": str(scheduler_dir),
        "python": str(python_exe),
        "configs_dir": str(resolve_repo_path(args.configs_dir)),
        "output_root": str(resolve_repo_path(args.output_root)),
        "time": args.time,
        "node_count": NODE_COUNT,
        "node_partitions": list(args.node_partitions),
        "jobs_per_node": args.jobs_per_node,
        "partitions": {
            "cpu": CPU_PARTITION,
            "gpu": GPU_PARTITION,
        },
        "gpu_algorithms": sorted(args.gpu_algorithms),
        "gpu_routing": {
            "mode": gpu_routing_mode(args.node_partitions),
            "description": describe_gpu_routing(
                gpu_algorithms=args.gpu_algorithms,
                node_partitions=args.node_partitions,
            ),
        },
        "gpu_gres": GPU_GRES,
        "batches": [asdict(batch) for batch in batches],
    }
    tmp_path = manifest_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(manifest_path)


def safe_job_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return cleaned[:120]


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
