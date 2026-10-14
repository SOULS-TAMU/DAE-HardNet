# Quality checks and badge scope

README workflow badges report the jax-library-ci review branch. Switch their branch query to main after the library PR is merged. Green workflow status means that the named job ran successfully; it is not a proof of mathematical correctness or absence of security vulnerabilities.

- Native CPU tests: Linux, Windows, macOS on Python 3.11/3.12/3.13, excluding the separate shared integration tests.
- Repository quality: source/metadata/citation agreement, complete package inventory, local Markdown links, YAML structure, Ruff correctness checks, extended Ruff checks for tests/tools and actionlint workflow checks.
- Coverage: full native-library line/branch measurements, published as an artifact. It excludes KKTHN integration and does not claim comprehensive coverage of the library. No arbitrary coverage score or percentage badge is supplied.
- CodeQL: Python security-and-quality queries. Inspect code-scanning alerts in addition to workflow success. The scan can complete successfully while reporting findings.
- Native dependency audit: known vulnerabilities in freshly resolved native runtime dependencies and audit tooling. It cannot audit the unpublished DAEHN distribution or unavailable compatible KKTHN package; skipped packages are reported by pip-audit. No advisory is suppressed to make a badge green.
- Documentation: strict MkDocs build and artifact upload.
- Release preflight: source build, metadata checks and version agreement; it creates no release on a pull request.
- Shared integration and full installation: still blocked until kkt-hardnet 0.8 is supplied through a supported index or KKTHN_SOURCE, as previously agreed.

Tests include analytical gradients, parameter sensitivities, randomized feasibility/idempotence/normal corrections, dense/sparse definition agreement, float32 tolerances, invalid preparations, DAE residual reconstruction, serialization round trips and malformed-artifact rejection. Projection formulas, solver defaults, training routines and checkpoints are unchanged; the CodeQL cleanup removes an unused import.

Run `python tools/check_repository.py`, `python -m pytest`, and `python -m ruff check src tests tools` locally. Native-only CI is an isolation technique, not a supported substitute for installing all required dependencies. CodeQL and dependency audit findings should be reviewed before describing the repository as secure.
