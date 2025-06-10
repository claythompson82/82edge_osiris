#!/usr/bin/env python3
import os
import subprocess
import time
from datetime import datetime


def log(msg: str) -> None:
    print(f"[{datetime.utcnow().isoformat()}] {msg}", flush=True)


def restart_redis(container_id: str, downtime: int) -> None:
    log(f"Stopping Redis container {container_id}")
    subprocess.run(["docker", "stop", container_id], check=True)
    log(f"Redis stopped. Sleeping for {downtime}s")
    time.sleep(downtime)
    log(f"Starting Redis container {container_id}")
    subprocess.run(["docker", "start", container_id], check=True)
    log("Redis container restarted")


def main() -> None:
    cid = os.environ.get("REDIS_CONTAINER_ID")
    if not cid:
        log("REDIS_CONTAINER_ID env var not set; exiting")
        return
    cycles = int(os.environ.get("CHAOS_REDIS_CYCLES", "3"))
    downtime = int(os.environ.get("CHAOS_REDIS_DOWNTIME", "5"))
    sleep_between = int(os.environ.get("CHAOS_REDIS_SLEEP", "10"))

    log(
        f"Starting Redis chaos: cycles={cycles} downtime={downtime}s sleep={sleep_between}s"
    )
    for i in range(cycles):
        log(f"Cycle {i+1}/{cycles}")
        restart_redis(cid, downtime)
        if i < cycles - 1:
            log(f"Sleeping {sleep_between}s before next cycle")
            time.sleep(sleep_between)
    log("Redis chaos test complete")


if __name__ == "__main__":
    main()
