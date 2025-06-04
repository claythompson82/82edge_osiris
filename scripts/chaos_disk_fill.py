#!/usr/bin/env python3
import argparse
import os
import subprocess
import time
from datetime import datetime


def log(msg: str) -> None:
    print(f"[{datetime.utcnow().isoformat()}] {msg}", flush=True)


def run(cmd: str) -> None:
    subprocess.run(cmd, shell=True, check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill disk space inside a container")
    parser.add_argument(
        "--container", default=os.environ.get("DISK_CHAOS_CONTAINER", "llm-sidecar")
    )
    parser.add_argument(
        "--path", default=os.environ.get("DISK_CHAOS_PATH", "/app/lancedb_data")
    )
    parser.add_argument(
        "--size",
        type=int,
        default=int(os.environ.get("DISK_CHAOS_SIZE_MB", "500")),
        help="Size in MB",
    )
    parser.add_argument(
        "--duration", type=int, default=int(os.environ.get("DISK_CHAOS_DURATION", "30"))
    )
    args = parser.parse_args()

    file_path = os.path.join(args.path, "diskfill.tmp")
    log(
        f"Filling {args.size}MB at {file_path} in {args.container} for {args.duration}s"
    )
    run(
        f"docker exec {args.container} dd if=/dev/zero of={file_path} bs=1M count={args.size} || true"
    )
    time.sleep(args.duration)
    run(f"docker exec {args.container} rm -f {file_path} || true")
    log("Disk fill chaos complete")


if __name__ == "__main__":
    main()
