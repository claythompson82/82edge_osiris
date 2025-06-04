#!/usr/bin/env python3
import argparse
from datetime import datetime
import subprocess
import lancedb


def container_running(name: str) -> bool:
    result = subprocess.run([
        "docker",
        "inspect",
        "-f",
        "{{.State.Running}}",
        name,
    ], capture_output=True, text=True)
    return result.stdout.strip() == "true"


def process_running(pattern: str) -> bool:
    result = subprocess.run(["pgrep", "-f", pattern], capture_output=True)
    return result.returncode == 0


def advice_count(db_path: str) -> int:
    try:
        db = lancedb.connect(db_path)
        tbl = db.open_table("advice")
        return tbl.count_rows()
    except Exception:
        return -1


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate chaos test report")
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--output", default="chaos_report.txt")
    args = parser.parse_args()

    orchestrator_ok = process_running("osiris_policy/orchestrator.py")
    sidecar_ok = container_running("llm-sidecar")
    redis_ok = container_running("redis")
    adv_count = advice_count(args.db_path)

    lines = [
        "Chaos Test Report",
        f"Timestamp: {datetime.utcnow().isoformat()}",
        f"Orchestrator running: {orchestrator_ok}",
        f"LLM Sidecar running: {sidecar_ok}",
        f"Redis running: {redis_ok}",
        f"Advice entries: {adv_count}",
    ]

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
