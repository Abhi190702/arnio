# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] — 2026-07-15

### 🔥 Breaking Changes

Complete architecture rewrite. No backwards compatibility with v1.x.

- Removed C++ extension (`_arnio_cpp`)
- Removed ArFrame container class
- Removed monolithic API — replaced with focused subsystems
- Removed `scikit-build-core` — now a pure Python package (`hatchling`)

### ✨ Added

- **Schema system** — class-based and dict-based schema definition
  - 12 built-in field types: `Int`, `Float`, `String`, `Bool`, `Date`, `DateTime`
  - 6 semantic types: `Email`, `URL`, `PhoneNumber`, `IPAddress`, `UUID`, `Regex`
  - Schema inference: `ar.infer_schema(df)`
  - Schema diff: `ar.diff_schemas(a, b)`
  - Schema serialization: JSON and YAML
- **Validation engine** — `ar.validate(df, schema)` returns structured results
  - Rich `ValidationResult` with `.to_dict()`, `.to_json()`, `.to_html()`, `.to_markdown()`, `.to_pandas()`
  - Never raises on bad data — always returns results
  - Per-value semantic validation (email format, URL format, etc.)
  - Constraint validation: nullable, unique, min/max, allowed values, patterns
- **Profiling engine** — `ar.profile(df)` returns comprehensive quality report
  - Quality score (0–100) based on completeness, uniqueness, consistency, validity
  - Per-column statistics: null rate, unique ratio, value counts, numeric stats, string lengths
  - Quality warnings: all_null, high_null_rate, constant, high_cardinality
- **Cleaning engine** — `ar.clean(df, steps)` applies declarative cleaning
  - 11 built-in cleaning steps
  - Custom step registration: `ar.register_step("name", fn)`
  - Reusable `Pipeline` class with JSON/YAML serialization
  - Type-preserving: pandas in → pandas out, dicts in → dicts out
- **Suggest engine** — `ar.suggest(df)` auto-suggests cleaning steps
  - Profile-driven recommendations with confidence scores
- **Quality gates** — `ar.check(df, schema)` for CI/CD assertions
- **Adapter layer** — protocol-based DataFrame abstraction
  - `PandasAdapter` — full pandas support
  - `DictAdapter` — dict/list-of-dicts support
- **pandas accessor** — `df.arnio.validate()`, `.profile()`, `.clean()`, `.suggest()`
- PEP 561 typed package (`py.typed`)

### 🏗 Architecture

- Pure Python — no C++ dependencies, no binary builds
- Protocol-based adapter pattern — engines never touch raw DataFrames
- Layered design: Public API → Engines → Adapters → Data
- Frozen dataclasses for field types — immutable, hashable, serializable
- Thread-safe step registry with `register_step()` / `unregister_step()`

## [1.x] — Legacy

See the `main` branch for v1.x history.
