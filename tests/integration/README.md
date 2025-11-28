# Integration Tests

## HEADER
- **Purpose**: Hold integration tests that exercise external services or multi-component flows
- **Status**: Active
- **Date**: 2025-11-28
- **Dependencies**: External services, I/O systems, and project code
- **Target**: Developers and CI systems

## Content
Use this directory for tests that interact with databases, object stores, external APIs, or complex workflows. Tests should handle missing dependencies gracefully (e.g., via pytest markers or skips).

