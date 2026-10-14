# Maintenance and automation

## CI

ci.yml tests CPU execution on Linux, Windows and macOS with Python 3.11/3.12/3.13. It enables JAX x64, runs numerical tests, and runs correctness-oriented Ruff rules in a separate job. It does not run full benchmark training. The declared Python 3.10 minimum is not yet verified by this matrix.

package.yml builds wheel/source artifacts, checks metadata and imports/subpackage inventory, then tests clean installations from each artifact outside the checkout. docs.yml builds documentation strictly and uploads it as an artifact; it does not publish a website.

All dependency-installing workflows accept an optional repository variable KKTHN_SOURCE containing a pip-installable compatible source path or URL. On hosted runners use an accessible immutable Git URL, for example a maintainer-provided URL pinned to a commit. Do not put credentials in this variable. Without it, dependency resolution uses the declared kkt-hardnet requirement from the configured package index. A missing compatible distribution will fail installation visibly. The appropriate public source still needs to be supplied by maintainers if the index does not host it.

Local validation can use the already installed compatible library or `python -m pip install ../kkthn`. Passing local tests does not mean GitHub's three operating systems have already passed. Inspect actual Actions results after pushing the files.

## GitHub releases

release.yml runs a build-only preflight on pull requests. Actual release creation is manually dispatched. It builds and checks source artifacts and creates a GitHub release for an existing tag pointing to the dispatched commit. The job uses a `release` environment: configure approval protection in GitHub before using it. Ensure CI passed for the tagged commit. There is no PyPI upload, token or Trusted Publisher configuration.

Do not overwrite an existing release/tag to hide a compatibility change. Update version metadata and src/__init__.py together, record changes in CHANGELOG.md, and preserve relevant checkpoint/import tests. No tag, release, remote or repository settings are created by adding these files.

## Citation and provenance

LICENSE preserves the upstream MIT notice. CITATION.cff identifies the software repository. The paper URL currently mentioned in the original upstream README resolves to the KKT-HardNet paper, so this checkout does not invent a DAE-HardNet manuscript DOI or author list. Add a verified manuscript citation when supplied.

The CPU matrix runs native-library tests independently of KKTHN. A separate integration.yml shared-backend workflow and the full installation job require the compatible kkt-hardnet 0.8 distribution or a configured KKTHN_SOURCE. Native-only installation uses --no-deps after installing other declared dependencies and is a CI isolation technique, not the recommended user installation. Until the compatible shared package is available, those dependency jobs remain blocked.
