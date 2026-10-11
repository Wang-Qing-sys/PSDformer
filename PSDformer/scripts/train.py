#!/usr/bin/env python3
"""Convenience wrapper; equivalent to python -m psdformer train."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from psdformer.__main__ import main

if __name__ == "__main__":
    main(["train", *sys.argv[1:]])
