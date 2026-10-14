"""CLI for compiling a portable DAEHN predictor on the current target."""
from __future__ import annotations

import argparse
from pathlib import Path

from .executable import compile_portable_to_native


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("portable", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = compile_portable_to_native(args.portable, args.output)
    print(f"native_executable={args.output}")
    print(f"os={manifest['os']} architecture={manifest['architecture']}")
    print(f"backend={manifest['backend']} device={manifest['device_kind']}")


if __name__ == "__main__":
    main()
