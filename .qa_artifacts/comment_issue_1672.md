Re-tested against current `main` / version 1.19.0. The explicit `.venv*`, `venv/`, `.tox/`, and cache exclusions now address the original virtual-environment example, but the broader dirty-worktree leak remains.

### Current reproduction

```powershell
python -m build --sdist --no-isolation --outdir .qa_artifacts/dist-audit
python -m twine check .qa_artifacts/dist-audit/arnio-1.19.0.tar.gz
```

The resulting archive passed `twine check`, but contained:

- **1,827 total entries**
- **1,537 entries** under the untracked `.qa_artifacts/` directory
- untracked repository-audit files such as:
  - `open_prs_latest.json` (3,509,890 bytes uncompressed)
  - `issues_all_scan23.json` (2,991,725 bytes)
  - `open_issues_latest.json` (2,861,642 bytes)
- a previously built wheel nested inside the sdist
- generated CMake logs and pytest fixtures

The compressed sdist was **4,619,399 bytes**, compared with a **398,229-byte** Windows wheel. This confirms that current `sdist.exclude` rules are a denylist and do not prevent unrelated untracked workspace content from entering a release archive.

### Updated impact

This is not limited to virtual environments. Any untracked local report, fixture, credential-like file, or generated artifact outside the current denylist may be published in an sdist. Release validation does not currently detect it.

### Suggested acceptance criteria

- Build sdists from an explicit allowlist / VCS-tracked file set, or enforce a clean worktree before release.
- Add an archive-content validation step that rejects unexpected top-level paths and generated artifacts.
- Add a regression fixture containing an arbitrary untracked sentinel and assert that it is absent from the sdist.
- Keep the existing virtual-environment exclusions as defense in depth.
