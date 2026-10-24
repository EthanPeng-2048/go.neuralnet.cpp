# Security Policy

## Supported versions

The project is at pre-1.0 development status. Security fixes are applied to
the latest commit on `main`; older snapshots are not separately supported.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use GitHub's private
vulnerability reporting through the repository's **Security** tab and include:

- affected commit or version;
- operating system and toolchain;
- a minimal reproducer or malformed input;
- expected and observed behavior;
- whether the issue can cause memory corruption, arbitrary file access, code
  execution, or a denial of service.

Malformed SGF, `.nngames`, `.nnvocab`, and model files are security-relevant
inputs. If private reporting is unavailable, contact the maintainer through
the GitHub profile linked from the repository and ask for a private channel
without disclosing exploit details publicly.

## Dependency vulnerabilities

Report vulnerabilities in neuralnet.cpp or KataGo to the respective upstream
project when the issue is in upstream code. This project will still track any
version bump, integration impact, and mitigation required on its side.
