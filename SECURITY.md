# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 2.0.x   | ✅         |
| < 2.0   | ❌         |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it privately:

1. **Do NOT** open a public issue
2. Email: anishrajyadav97@gmail.com
3. Include: description, steps to reproduce, potential impact

We will respond within 48 hours and provide a fix timeline.

## Security Considerations

Arnio processes user-provided data and schemas. Key security measures:

- **Regex patterns**: Validated against catastrophic backtracking
- **No code execution**: Schemas are data, not code
- **No network access**: Arnio never makes network requests
- **No file I/O**: Arnio does not read or write files (schemas are serialized to strings)