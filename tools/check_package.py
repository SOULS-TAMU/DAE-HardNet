"""Build artifacts and test each in an independent temporary environment."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import venv

ROOT = Path(__file__).resolve().parents[1]


def run(command, cwd):
    subprocess.run([str(x) for x in command], cwd=cwd, check=True)


def main():
    packages = tomllib.loads((ROOT/"pyproject.toml").read_text())["tool"]["setuptools"]["packages"]
    discovered = {"daehn" + ("."+str(p.parent.relative_to(ROOT/"src")).replace("\\", ".").replace("/", ".") if p.parent != ROOT/"src" else "") for p in (ROOT/"src").rglob("__init__.py")}
    if set(packages) != discovered:
        raise RuntimeError("Setuptools package inventory mismatch: " + repr(set(packages)^discovered))
    with tempfile.TemporaryDirectory(prefix="daehn-package-") as temporary:
        temporary = Path(temporary);artifacts = temporary/"artifacts"
        run([sys.executable, "-m", "build", "--outdir", artifacts, ROOT], temporary)
        archives = sorted(artifacts.iterdir())
        run([sys.executable, "-m", "twine", "check", "--strict", *archives], temporary)
        for index, archive in enumerate(archives):
            directory = temporary/("environment-"+str(index))
            venv.EnvBuilder(with_pip=True).create(directory)
            python = directory/("Scripts/python.exe" if os.name == "nt" else "bin/python")
            source = os.environ.get("KKTHN_SOURCE", "").strip()
            if source:
                run([python, "-m", "pip", "install", source], temporary)
            run([python, "-m", "pip", "install", archive, "pytest"], temporary)
            run([python, "-m", "pip", "check"], temporary)
            code = "import importlib,json; packages=json.loads("+repr(json.dumps(packages))+"); [importlib.import_module(p) for p in packages]; import daehn; print(daehn.__file__); assert 'site-packages' in daehn.__file__"
            run([python, "-c", code], temporary)
            run([python, "-m", "pytest", "--import-mode=importlib", ROOT/"tests"], temporary)
        print("Wheel and source-distribution clean-install checks passed.")


if __name__ == "__main__":
    main()
