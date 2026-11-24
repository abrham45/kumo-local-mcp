# Release procedure

Do not claim a PyPI release based on a successful local build. The local Engine CPU variant is a
runtime artifact and must not be uploaded. Publishing is performed by `.github/workflows/publish.yml`
using GitHub trusted publishing; a maintainer starts the workflow after reviewing CI and artifacts.

## Trusted publishing setup

Register a pending publisher at https://pypi.org/manage/account/publishing/ with these exact values:

- PyPI project: `kumo-local-mcp`
- GitHub owner: `abrham45`
- Repository: `kumo-local-mcp`
- Workflow filename: `publish.yml`
- Environment: `pypi`

Then run **Publish to PyPI** from the repository's Actions tab on `main`. The workflow tests and
builds before its dedicated publishing job receives a short-lived OIDC identity. No PyPI token is
stored in the repository. Protect the `pypi` environment as appropriate for repository maintainers.

## Release checklist

1. Check the normalized name `kumo-local-mcp` on PyPI and decide ownership. A missing project page is
   only a point-in-time observation; it does not reserve the name or settle trademark questions.
2. Run unit and stdio tests on the supported Python/OS matrix. Run the real-model smoke runner for
   all four families, classification and regression, reusable contexts, explanations, fine-tuning,
   and Engine graph/snapshot/training/prediction on supported hardware. Record actual versions.
3. Review the feature matrix and experimental limitations. Reconcile upstream SDK changes with
   adapter schemas and rerun the relevant real-model checks.
4. Set the version in `pyproject.toml` and `src/kumo_local_mcp/__init__.py`, then update CHANGELOG.
5. Inspect `git status --short` and `git ls-files` before staging. Keep local configurations,
   credentials, environments, datasets, weights, results, and verification artifacts out of Git.
   `.gitignore` does not remove files already tracked; explicitly untrack such files before pushing.
   Keep the sanitized `examples/config.example.json` and two curated example CSV files.
   Build with `python -m build`. Check both artifacts with `python -m twine check --strict dist/*` and inspect
   their contents. No datasets, environment files, tokens, weights, or user configuration belong in
   the package. Install the wheel into a clean environment and verify `setup` and stdio startup.
6. Configure PyPI trusted publishing for the eventual repository, or supply a scoped publishing
   credential through your local credential tooling. Credentials must never be pasted into source.
7. Upload the exact reviewed artifacts to TestPyPI first, validate installation, then publish to PyPI:

   ```bash
   python -m twine upload --repository testpypi dist/*
   python -m twine upload dist/*
   ```

8. Verify the public package version and install it in a fresh environment. Record the public URL
   and release artifact hashes. Only then mark the PyPI release milestone complete.

The CI workflow runs unit/stdio checks and builds distributions. The publish workflow remains manual
and requires the matching PyPI trusted publisher before an upload can succeed.
