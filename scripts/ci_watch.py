#!/usr/bin/env python3
"""Trigger a GitHub Actions workflow, watch it, and download its logs."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = ROOT / ".artifacts" / "ci-logs"
API_ROOT = "https://api.github.com/repos/compTAG/TDAtaframe"


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


def token() -> str | None:
    return os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")


def api_request(
    path: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    accept: str = "application/vnd.github+json",
) -> bytes:
    headers = {
        "Accept": accept,
        "User-Agent": "tdataframe-ci-watch",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if current_token := token():
        headers["Authorization"] = f"Bearer {current_token}"
    request = urllib.request.Request(
        f"{API_ROOT}{path}",
        data=body,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def api_json(path: str, **kwargs: object) -> dict[str, object]:
    return json.loads(api_request(path, **kwargs).decode())


def current_branch() -> str:
    result = run(["git", "branch", "--show-current"], capture=True)
    branch = result.stdout.strip()
    if not branch:
        raise SystemExit("Could not determine the current git branch.")
    return branch


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def current_sha() -> str:
    result = run(["git", "rev-parse", "HEAD"], capture=True)
    return result.stdout.strip()


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


def api_find_newest_run(
    workflow: str,
    ref: str,
    started_at: datetime,
    head_sha: str | None = None,
) -> dict[str, object] | None:
    data = api_json(f"/actions/runs?per_page=20&branch={ref}")
    runs = data.get("workflow_runs", [])
    if not isinstance(runs, list):
        return None
    for item in runs:
        if item.get("path", "").endswith(workflow) and (
            head_sha is None or item.get("head_sha") == head_sha
        ):
            created_at = str(item["created_at"])
            if parse_time(created_at) >= started_at:
                return item
    return None


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


def api_download_logs(run_id: str, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    run_data = api_json(f"/actions/runs/{run_id}")
    (log_dir / f"run-{run_id}-summary.json").write_text(
        json.dumps(run_data, indent=2, sort_keys=True)
    )

    jobs = api_json(f"/actions/runs/{run_id}/jobs?per_page=100")
    (log_dir / f"run-{run_id}-jobs.json").write_text(
        json.dumps(jobs, indent=2, sort_keys=True)
    )

    if not token():
        print(
            "No GH_TOKEN/GITHUB_TOKEN is set; saved status JSON, but GitHub "
            "requires authentication for raw job logs.",
            flush=True,
        )
        return

    try:
        logs_zip = api_request(
            f"/actions/runs/{run_id}/logs",
            accept="application/zip",
        )
    except urllib.error.HTTPError as error:
        print(f"Could not download run logs: HTTP {error.code}", flush=True)
        return

    archive_path = log_dir / f"run-{run_id}-logs.zip"
    archive_path.write_bytes(logs_zip)
    try:
        with zipfile.ZipFile(io.BytesIO(logs_zip)) as archive:
            archive.extractall(log_dir / f"run-{run_id}-logs")
    except zipfile.BadZipFile:
        (log_dir / f"run-{run_id}.log").write_bytes(logs_zip)


def api_dispatch(workflow: str, ref: str) -> None:
    if not token():
        raise SystemExit("Set GH_TOKEN or GITHUB_TOKEN to dispatch workflows.")
    body = json.dumps({"ref": ref}).encode()
    api_request(
        f"/actions/workflows/{workflow}/dispatches",
        method="POST",
        body=body,
    )


def api_watch(args: argparse.Namespace) -> None:
    ref = args.ref or current_branch()
    head_sha = current_sha()
    started_at = datetime.now(UTC)

    if not args.no_trigger:
        api_dispatch(args.workflow, ref)
        time.sleep(5)
    else:
        started_at = datetime.fromtimestamp(0, UTC)
        head_sha = None

    run_info = None
    deadline = time.monotonic() + args.timeout_seconds
    while time.monotonic() < deadline:
        run_info = api_find_newest_run(args.workflow, ref, started_at, head_sha)
        if run_info:
            break
        time.sleep(5)

    if not run_info:
        raise SystemExit(f"No workflow run found for {args.workflow} on {ref}")

    run_id = str(run_info["id"])
    print(f"Watching {run_info['html_url']}", flush=True)
    conclusion = None
    while time.monotonic() < deadline:
        run_data = api_json(f"/actions/runs/{run_id}")
        status = run_data.get("status")
        conclusion = run_data.get("conclusion")
        print(f"{status} {conclusion}", flush=True)
        if status == "completed":
            break
        time.sleep(30)

    api_download_logs(run_id, args.log_dir.resolve())
    print(f"Saved logs under {args.log_dir.resolve()}", flush=True)
    sys.exit(0 if conclusion == "success" else 1)


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
    require_command("git")
    if shutil.which("gh") is None:
        api_watch(args)
        return

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
