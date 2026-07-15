# Contributing to Arnio

Thank you for your interest in contributing to Arnio!

## Development Setup

```bash
git clone https://github.com/im-anishraj/arnio.git
cd arnio
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ --cov=arnio --cov-report=term-missing
```

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check arnio/ tests/
ruff format arnio/ tests/
```

## Type Checking

```bash
mypy arnio/
```

## Architecture

Arnio follows a layered architecture:

1. **Public API** (`arnio/__init__.py`) — curated, stable surface
2. **Core Engines** (`validate/`, `profile/`, `clean/`) — business logic
3. **Adapter Layer** (`adapt/`) — abstracts DataFrame operations
4. **Schema System** (`schema/`) — field types and schema definition

All implementation modules use a leading underscore (`_engine.py`, `_fields.py`).
Public API is only exported through `__init__.py` files.

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Write tests for your changes
4. Ensure all tests pass
5. Submit a pull request

## Adding a New Field Type

1. Create a new class in `arnio/schema/_fields.py` or `arnio/schema/_semantic.py`
2. Subclass `Field` and override `validate_value()`
3. Export it in `arnio/schema/__init__.py`
4. Re-export in `arnio/__init__.py`
5. Add serialization support in `arnio/schema/_serde.py`
6. Write tests

## Adding a New Cleaning Step

1. Add the step function in `arnio/clean/_steps.py`
2. Register it in `arnio/clean/_registry.py`
3. Write tests

## Code of Conduct

We follow the [Contributor Covenant](https://www.contributor-covenant.org/).
Be respectful, constructive, and inclusive.
