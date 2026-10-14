"""Install optional compatible KKTHN source before the declared DAEHN extras."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tomllib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extras", default="dev,docs")
    parser.add_argument("--native-only", action="store_true", help="Test native code without the unavailable shared dependency; not a full supported installation")
    args = parser.parse_args()
    if args.native_only:
        root = Path(__file__).resolve().parents[1]
        metadata = tomllib.loads((root/"pyproject.toml").read_text(encoding="utf-8"))
        dependencies = [d for d in metadata["project"]["dependencies"] if not d.startswith("kkt-hardnet")]
        for extra in filter(None, args.extras.split(",")):
            dependencies.extend(metadata["project"]["optional-dependencies"][extra])
        subprocess.run([sys.executable, "-m", "pip", "install", *dependencies], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", str(root)], check=True)
        print("Native-only CI installation: KKTHN integration and full dependency validation are not covered.")
        return
    source = os.environ.get("KKTHN_SOURCE", "").strip()
    if source:
        subprocess.run([sys.executable, "-m", "pip", "install", source], check=True)
    target = str(Path(__file__).resolve().parents[1]) + ("["+args.extras+"]" if args.extras else "")
    subprocess.run([sys.executable, "-m", "pip", "install", target], check=True)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)


if __name__ == "__main__":
    main()
