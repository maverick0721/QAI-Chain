# API Compatibility Policy

## Versioning

- Current API version: v0 (prototype)
- Compatibility target: backward compatibility within a minor release line

## Stability Rules

- Existing endpoint paths and required request fields are not removed without a deprecation window.
- New optional fields may be added at any time.
- Response payloads may gain additive fields but existing keys keep semantics.
- Validation tightening that would reject previously-valid payloads requires a version bump note.

## Contract Source of Truth

- OpenAPI schema artifact: [docs/api_openapi.json](docs/api_openapi.json)
- Human-readable endpoint summary: [docs/API_SCHEMA.md](docs/API_SCHEMA.md)

## Deprecation Process

1. Mark endpoint/field as deprecated in docs and changelog.
2. Keep support for at least one release cycle.
3. Remove only after versioned migration guidance is available.
