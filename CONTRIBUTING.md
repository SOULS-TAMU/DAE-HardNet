# Contributing

Install a supported Python interpreter and create a virtual environment as described in [installation](docs/installation.md). Install compatible KKTHN first when it is supplied as a source checkout, then run `python -m pip install -e ".[dev,docs]"`.

Before submitting a change:

```bash
python -m pytest
python -m ruff check src tests tools
python -m mkdocs build --strict
python tools/check_package.py
```

Describe the problem, resulting behavior and validation in the pull request. Add numerical tests for mathematical changes: use analytical reference problems and tolerances, and check gradients away from nonsmooth switching points. Preserve historical imports and saved-model behavior or document a deliberate compatibility change in CHANGELOG.md. Do not update numerical reference values merely to make a failing test pass.

Source is directly in src/, mapped to the daehn package by setuptools. When adding a package, update the explicit package inventory in pyproject.toml. Tests must work against an installed distribution from outside the checkout. Avoid tests that depend on local benchmark folders, absolute paths or large trained models.

Keep full case studies, datasets and models in the separate benchmark project. Runnable usage snippets belong in the documentation. No full benchmark training runs are required for documentation-only changes. Discuss substantial API or numerical changes in an issue first. Optional pre-commit checks can be installed with `pre-commit install`.

For local or GitHub release artifacts, see [maintenance](docs/maintenance.md). Publication to PyPI is not part of this repository's workflow.
