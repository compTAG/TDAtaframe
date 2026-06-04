#!/usr/bin/env python3
"""Trigger a GitHub Actions workflow, watch it, and download its logs."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = ROOT / ".artifacts" / "ci-logs"


def run(
    cmd: list[str],
    *,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(
        cmd,
        cwd=ROOT,
        check=check,
        text=True,
        capture_output=capture,
    )


def require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Missing required command: {name}")


def current_branch() -> str:
    result = run(["git", "branch", "--show-current"], capture=True)
    branch = result.stdout.strip()
    if not branch:
        raise SystemExit("Could not determine the current git branch.")
    return branch


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def find_newest_run(
    workflow: str,
    ref: str,
    started_at: datetime,
) -> dict[str, object] | None:
    result = run(
        [
            "gh",
            "run",
            "list",
            "--workflow",
            workflow,
            "--branch",
            ref,
            "--limit",
            "10",
            "--json",
            "databaseId,status,conclusion,createdAt,url,name,headSha",
        ],
        capture=True,
    )
    runs = json.loads(result.stdout)
    for item in runs:
        if parse_time(item["createdAt"]) >= started_at:
            return item
    return runs[0] if runs else None


def download_logs(run_id: str, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)

    summary = run(["gh", "run", "view", run_id], check=False, capture=True)
    (log_dir / f"run-{run_id}-summary.txt").write_text(summary.stdout + summary.stderr)

    log = run(["gh", "run", "view", run_id, "--log"], check=False, capture=True)
    (log_dir / f"run-{run_id}.log").write_text(log.stdout + log.stderr)

    artifacts_dir = log_dir / f"run-{run_id}-artifacts"
    artifacts = run(
        ["gh", "run", "download", run_id, "--dir", str(artifacts_dir)],
        check=False,
        capture=True,
    )
    (log_dir / f"run-{run_id}-artifact-download.txt").write_text(
        artifacts.stdout + artifacts.stderr
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", default="CI.yml")
    parser.add_argument(
        "--ref",
        default=None,
        help="Git ref to run. Defaults to current branch.",
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument(
        "--no-trigger",
        action="store_true",
        help="Watch the newest existing run instead of dispatching a new run.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_command("gh")
    require_command("git")

    ref = args.ref or current_branch()
    started_at = datetime.now(UTC)

    if not args.no_trigger:
        run(["gh", "workflow", "run", args.workflow, "--ref", ref])
        time.sleep(5)
    else:
        started_at = datetime.fromtimestamp(0, UTC)

    run_info = None
    deadline = time.monotonic() + args.timeout_seconds
    while time.monotonic() < deadline:
        run_info = find_newest_run(args.workflow, ref, started_at)
        if run_info:
            break
        time.sleep(5)

    if not run_info:
        raise SystemExit(f"No workflow run found for {args.workflow} on {ref}")

    run_id = str(run_info["databaseId"])
    print(f"Watching {run_info['url']}", flush=True)
    watch = run(["gh", "run", "watch", run_id, "--exit-status"], check=False)
    download_logs(run_id, args.log_dir.resolve())
    print(f"Saved logs under {args.log_dir.resolve()}", flush=True)
    sys.exit(watch.returncode)


if __name__ == "__main__":
    main()
