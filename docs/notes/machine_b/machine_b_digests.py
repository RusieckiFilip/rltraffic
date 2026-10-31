"""Machine B (docs/notes/MACHINE_B.md): digests that compare a Python environment with the laptop's without copying it.

Run with any Python 3.12 or later, standard library only:  python3 -I machine_b_digests.py <command> ...

  record <site-packages>         One line per installed distribution: ``name==version digest count``. The digest is the
                                 sha256 over the sorted ``path,hash,size`` lines of its RECORD, keeping only files inside
                                 site-packages that carry a hash: the package's own library files, exactly as its wheel
                                 shipped them. Left out: pip's bookkeeping (INSTALLER, REQUESTED, direct_url.json, RECORD),
                                 the .pyc caches (no hash) and files installed outside site-packages (console scripts).
  verify-record <site-packages>  Hashes every file ``record`` keeps and compares it with its RECORD entry; a file changed
                                 after installation is printed and fails the command.
  deb <package> ...              One line per Debian package: ``package digest count``. The digest is the sha256 over the
                                 sorted ``path sha256`` lines of the package's installed regular files (``dpkg -L``).
"""
from __future__ import annotations

import base64
import csv
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

#: pip's own files in a .dist-info directory: they record how a package was installed, not what it contains.
BOOKKEEPING = {"INSTALLER", "REQUESTED", "direct_url.json", "RECORD"}


def normalised(name: str) -> str:
    """PEP 503's normalised project name, as pip freeze's names compare."""
    return re.sub(r"[-_.]+", "-", name).lower()


def distributions(site: Path):
    """Each ``*.dist-info`` under *site*: its normalised name, version and the RECORD rows ``record`` keeps."""
    for info in sorted(site.glob("*.dist-info")):
        metadata = (info / "METADATA").read_text(encoding="utf-8", errors="replace")
        name = re.search(r"^Name: *(.+)$", metadata, re.M).group(1).strip()
        version = re.search(r"^Version: *(.+)$", metadata, re.M).group(1).strip()
        rows = []
        with open(info / "RECORD", newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if len(row) < 3 or not row[1] or row[0].startswith("../"):
                    continue
                if row[0].startswith(info.name + "/") and row[0].rsplit("/", 1)[-1] in BOOKKEEPING:
                    continue
                rows.append((row[0], row[1], row[2]))
        yield normalised(name), version, rows


def record(site: Path) -> int:
    for name, version, rows in distributions(site):
        text = "\n".join(sorted(f"{path},{digest},{size}" for path, digest, size in rows))
        print(f"{name}=={version} {hashlib.sha256(text.encode('utf-8')).hexdigest()} {len(rows)}")
    return 0


def verify_record(site: Path) -> int:
    files = changed = 0
    for name, _, rows in distributions(site):
        for path, digest, size in rows:
            algorithm, _, expected = digest.partition("=")
            data = (site / path).read_bytes()
            actual = base64.urlsafe_b64encode(hashlib.new(algorithm, data).digest()).rstrip(b"=").decode("ascii")
            files += 1
            if actual != expected or (size and int(size) != len(data)):
                changed += 1
                print(f"CHANGED {name}: {path}")
    print(f"verify-record: {files} files, {changed} changed since installation")
    return 1 if changed else 0


def deb(packages: list[str]) -> int:
    for package in packages:
        listing = subprocess.run(["dpkg", "-L", package], capture_output=True, text=True, check=True).stdout
        lines = []
        for path in sorted({line for line in listing.splitlines() if line}):
            if os.path.isfile(path) and not os.path.islink(path):
                lines.append(f"{path} {hashlib.sha256(Path(path).read_bytes()).hexdigest()}")
        print(f"{package} {hashlib.sha256(chr(10).join(lines).encode('utf-8')).hexdigest()} {len(lines)}")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "record":
        return record(Path(argv[1]))
    if len(argv) >= 2 and argv[0] == "verify-record":
        return verify_record(Path(argv[1]))
    if len(argv) >= 2 and argv[0] == "deb":
        return deb(argv[1:])
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
