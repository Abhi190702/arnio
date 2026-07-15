## Summary

The dedicated `Windows MSVC Wheel Smoke Test` workflow starts `pip wheel .` without first initializing the Visual Studio developer environment. The latest main-branch check fails during this job, and the workflow is inconsistent with the regular Windows C++ CI job, which explicitly runs `ilammy/msvc-dev-cmd` before invoking CMake.

## Environment / context

- Repository: current `main`
- Workflow: `.github/workflows/windows-msvc-wheel-smoke.yml`
- Runner: `windows-latest`
- Python: 3.12
- Build backend: `scikit-build-core`
- Local comparison environment: Windows with Visual Studio Community/MSVC installed

## Steps to reproduce

1. Start from a regular Windows PowerShell session where the Visual Studio developer environment has not been initialized.
2. Run:

   ```powershell
   python -m pip wheel . --no-deps -w dist/
   ```

3. Observe CMake attempting to configure a native build without a usable MSVC/NMake environment.
4. Compare the workflow with `.github/workflows/ci.yml`, whose Windows native job includes:

   ```yaml
   - name: Set up MSVC
     uses: ilammy/msvc-dev-cmd@0b201ec74fa43914dc39ae48a89fd1d8cb592756
   ```

## Expected result

The Windows wheel workflow initializes a deterministic MSVC build environment before building the native extension, and the wheel build/smoke test completes successfully.

## Actual result

The workflow only configures Python and then invokes `pip wheel`. In a non-developer shell this can leave CMake without `cl`, `nmake`, or another configured generator. A local reproduction failed during CMake configuration with NMake unavailable and `CMAKE_CXX_COMPILER` unset; the latest corresponding GitHub check also fails quickly at the wheel-smoke stage.

## Severity / impact

**High.** This leaves the Windows native-wheel release gate red and prevents the smoke workflow from providing reliable packaging confidence after C++ or binding changes.

## Likely affected area

- `.github/workflows/windows-msvc-wheel-smoke.yml`
- Windows wheel packaging / native extension builds

## Suggested next action

Initialize MSVC before `pip wheel` using the same pinned `ilammy/msvc-dev-cmd` action as the normal Windows C++ job. For additional determinism, install Ninja and configure the generator explicitly. After the build is restored, reuse `tests/smoke_wheel_install.py` or otherwise exercise a C++-backed operation in the installed wheel (tracked separately in #1897).

## Acceptance criteria

- The dedicated Windows wheel workflow initializes MSVC explicitly.
- A wheel builds successfully on `windows-latest` from a clean runner.
- The wheel installs into a fresh environment and passes its smoke test.
- A regression run confirms the job remains green after native C++/binding changes.
