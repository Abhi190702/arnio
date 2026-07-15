# Governance

## Decision Making

Arnio uses a **BDFL** (Benevolent Dictator For Life) model.
[Anish Raj](https://github.com/im-anishraj) has final say on all decisions.

## Contributor Ladder

| Role | Responsibility | How to reach |
|------|---------------|--------------|
| **Contributor** | Submit PRs, report issues | Open a PR |
| **Reviewer** | Review PRs, triage issues | Sustained quality contributions |
| **Committer** | Merge PRs, manage releases | Invitation from BDFL |

## Major Changes

Significant changes (new adapters, new field types, API changes) require
an RFC (Request for Comments) posted as a GitHub Discussion before implementation.

## Versioning

- Semantic versioning (semver) strictly enforced
- Public API (`arnio/__init__.py`) follows semver
- Private API (`_` prefixed modules) has no stability guarantee
- Deprecation cycle: warn for 2 minor versions, remove in next major
