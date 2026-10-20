#!/usr/bin/env python3
"""Call the installed package exporter; keep one implementation for every runtime."""
from __future__ import annotations

import subprocess
import sys

if __name__ == "__main__":
    raise SystemExit(subprocess.call([sys.executable, "-m", "paper2", "export", *sys.argv[1:]]))
