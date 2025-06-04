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


def ensure_tc(container: str) -> None:
    check = subprocess.run(
        f"docker exec {container} which tc",
        shell=True,
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        log("'tc' not found, attempting to install iproute2")
        run(f"docker exec {container} apt-get update -y")
        run(f"docker exec {container} apt-get install -y iproute2")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Introduce network latency and packet loss"
    )
    parser.add_argument(
        "--container", default=os.environ.get("NET_CHAOS_CONTAINER", "llm-sidecar")
    )
    parser.add_argument(
        "--delay-ms", default=os.environ.get("NET_CHAOS_DELAY_MS", "200")
    )
    parser.add_argument(
        "--loss", default=os.environ.get("NET_CHAOS_LOSS_PERCENT", "10")
    )
    parser.add_argument(
        "--duration", type=int, default=int(os.environ.get("NET_CHAOS_DURATION", "30"))
    )
    args = parser.parse_args()

    log(
        f"Applying netem to {args.container}: delay {args.delay_ms}ms loss {args.loss}% for {args.duration}s"
    )
    ensure_tc(args.container)
    run(
        f"docker exec {args.container} tc qdisc add dev eth0 root netem delay {args.delay_ms}ms loss {args.loss}%"
    )
    time.sleep(args.duration)
    run(f"docker exec {args.container} tc qdisc del dev eth0 root netem")
    log("Network chaos complete")


if __name__ == "__main__":
    main()
