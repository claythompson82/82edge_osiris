import json
import os
import re
from collections import deque
from packaging.requirements import Requirement
import requests

REQUIREMENTS_FILES = [
    "requirements.txt",
    "requirements-tests.txt",
    "requirements-docs.txt",
    "requirements-ci.txt",
    "services/azr_planner/requirements.txt",
]

PACKAGE_RE = re.compile(r"^[A-Za-z0-9_.-]+")


def parse_req_line(line):
    line = line.strip()
    if not line or line.startswith("#") or line.startswith("-r"):
        return None
    if "#" in line:
        line = line.split("#", 1)[0].strip()
    req = Requirement(line)
    return req.name, req.specifier


def gather_root_packages():
    pkgs = {}
    for path in REQUIREMENTS_FILES:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                parsed = parse_req_line(line)
                if not parsed:
                    continue
                name, spec = parsed
                version = None
                if spec:
                    # Use first exact version if pinned with ==
                    for s in spec:
                        if s.operator == "==":
                            version = s.version
                            break
                pkgs[name] = version
    return pkgs


class LicenseCollector:
    def __init__(self):
        self.packages = {}

    def fetch_metadata(self, name, version=None):
        if version:
            url = f"https://pypi.org/pypi/{name}/{version}/json"
        else:
            url = f"https://pypi.org/pypi/{name}/json"
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

    def extract_license(self, info):
        license_str = info.get("license") or ""
        if not license_str:
            for c in info.get("classifiers", []):
                if c.startswith("License ::"):
                    license_str = c.split("::")[-1].strip()
                    break
        return license_str or "UNKNOWN"

    def add(self, name, version=None):
        if name in self.packages:
            return
        meta = self.fetch_metadata(name, version)
        if not meta:
            self.packages[name] = {"version": version or "", "license": "UNKNOWN"}
            return
        info = meta["info"]
        license_name = self.extract_license(info)
        self.packages[name] = {
            "version": version or info.get("version", ""),
            "license": license_name,
        }

    def collect(self, roots):
        for name, version in roots.items():
            self.add(name, version)
        return self.packages


def main():
    roots = gather_root_packages()
    collector = LicenseCollector()
    data = collector.collect(roots)
    with open("LICENSES.json", "w") as f:
        json.dump(data, f, indent=2)
    with open("LICENSES.md", "w") as f:
        f.write("# Third-Party Licenses\n\n")
        for name, info in sorted(data.items()):
            ver_display = info.get("version", "latest")
            f.write(f"## {name} ({ver_display})\n")
            f.write(f'License: {info.get("license", "UNKNOWN")}\n\n')


if __name__ == "__main__":
    main()
