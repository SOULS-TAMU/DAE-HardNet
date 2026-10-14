# Installation from GitHub

DAEHN is distributed as source through GitHub. There is no PyPI publication workflow. Pip is used to install a checked-out repository into a virtual environment.

Create a virtual environment with Python 3.11, 3.12 or 3.13 for the initial CI baseline. The package currently declares Python >=3.10, but that minimum and GPU platforms are not covered by the initial CI matrix. JAX wheels determine which interpreter/platform combinations can be installed; see the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

```bash
git clone https://github.com/SOULS-TAMU/DAE-HardNet.git daehn
cd daehn
python -m venv .venv
```

Windows PowerShell activation: `.\.venv\Scripts\Activate.ps1`. Linux/macOS activation: `source .venv/bin/activate`.

Use the Git revision containing this library's pyproject.toml and src/ directory. Earlier upstream revisions use a different installation layout. Do not assume that the current remote default branch already contains these local changes.

## KKTHN dependency

The required shared library has Python import name `kkthn` and distribution name `kkt-hardnet`, version >=0.8.0,<0.9. Install it from your configured package index or from a compatible source checkout supplied by its maintainers:

```bash
python -m pip install --upgrade pip
python -m pip install /path/to/compatible/kkthn-checkout
python -m pip install .
python -m pip check
python -c "import daehn, kkthn; print(daehn.__version__)"
```

Replace the source path with your actual checkout. If kkt-hardnet 0.8 is available from your configured index, `python -m pip install .` resolves it automatically. Its public source URL is not configured in this local repository; no guessed dependency URL is embedded here.

For your existing sibling layout, install `../kkthn` before installing `.`. For development use `python -m pip install -e ".[dev,docs]"`, or `python -m pip install -r requirements-dev.txt`.

`requirements.txt` installs this checkout and resolves runtime dependencies from pyproject.toml. Avoid --no-deps for a new environment. CPU is the automated-test baseline. Configure CUDA packages separately using JAX's platform instructions; do not assume a CPU CI pass verifies GPU execution.
