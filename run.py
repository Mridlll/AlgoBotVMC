#!/usr/bin/env python3
"""Simple runner script for VMC Trading Bot."""

import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from main import cli

if __name__ == "__main__":
    cli()
