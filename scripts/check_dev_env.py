#!/usr/bin/env python3
"""Check local development environment prerequisites for Osiris."""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class CheckResult:
    name: str
    ok: bool
    message: str


def parse_version(text: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", text))


def run_command(cmd: List[str]) -> Tuple[str, str | None]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        return "", "not installed"
    except subprocess.CalledProcessError as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        return output.strip(), "command failed"
    return (result.stdout or result.stderr).strip(), None


def check_tool(name: str, cmd: List[str], min_version: str | None = None) -> CheckResult:
    output, error = run_command(cmd)
    if error:
        return CheckResult(name, False, error)
    if min_version:
        match = re.search(r"(\d+\.\d+\.\d+)", output)
        if match:
            version = match.group(1)
            if parse_version(version) < parse_version(min_version):
                return CheckResult(name, False, f"version {version} < {min_version}")
        else:
            return CheckResult(name, False, f"unable to parse version from '{output}'")
    return CheckResult(name, True, output.splitlines()[0])


def check_port(port: int) -> CheckResult:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("localhost", port))
        except OSError:
            return CheckResult(f"port {port}", False, "in use")
    return CheckResult(f"port {port}", True, "available")


def check_docker() -> CheckResult:
    try:
        subprocess.run(
            ["docker", "ps"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return CheckResult("docker", True, "docker ps succeeded")
    except FileNotFoundError:
        return CheckResult("docker", False, "not installed")
    except subprocess.CalledProcessError as exc:
        return CheckResult("docker", False, f"docker ps failed: {exc}")


def check_env_var(var: str) -> CheckResult:
    return CheckResult(var, var in os.environ, os.environ.get(var, "not set"))


def main() -> None:
    checks = [
        check_tool("python", ["python3", "--version"], "3.11.0"),
        check_tool("docker", ["docker", "--version"]),
        check_tool("docker compose", ["docker", "compose", "version"]),
        check_tool("make", ["make", "--version"]),
        check_tool("git", ["git", "--version"]),
        check_tool("pre-commit", ["pre-commit", "--version"]),
        check_tool("kubectl", ["kubectl", "version", "--client", "--short"]),
        check_tool("helm", ["helm", "version", "--short"]),
        check_tool("terraform", ["terraform", "version"]),
        check_docker(),
        check_port(8000),
        check_port(6379),
        check_env_var("OSIRIS_SIDECAR_URL"),
    ]

    failures = [c for c in checks if not c.ok]
    for c in checks:
        status = "OK" if c.ok else "FAIL"
        print(f"{status:4} {c.name}: {c.message}")

    if failures:
        print(f"\n{len(failures)} check(s) failed.")
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
