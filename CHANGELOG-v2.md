# Changelog

## [2.0.0-alpha.2](https://github.com/google/adk-python/compare/v2.0.0-alpha.1...v2.0.0-alpha.2) (2026-03-26)


### Bug Fixes

* add agent name validation to prevent arbitrary module imports ([be094f7](https://github.com/google/adk-python/commit/be094f7e50e3976cb69468c7a8d17f3385132c41))
* add protection for arbitrary module imports ([0019801](https://github.com/google/adk-python/commit/0019801f6080a371a71b09cbd8291f2e79a49336)), closes [#4947](https://github.com/google/adk-python/issues/4947)
* Default to ClusterIP so GKE deployment isn't publicly exposed by default ([27aed23](https://github.com/google/adk-python/commit/27aed23edddcc8c69c4c4563198c8570aa40743e))
* enforce allowed file extensions for GET requests in the builder API ([68ece4e](https://github.com/google/adk-python/commit/68ece4e04fb3829e61847dae123cf4bc79a0a4c2))
* Exclude compromised LiteLLM versions from dependencies pin to 1.82.6 ([79ed953](https://github.com/google/adk-python/commit/79ed95383dd0bc13984b6499913babef232d4dab))
* gate builder endpoints behind web flag ([584283e](https://github.com/google/adk-python/commit/584283e2d8209af9db719fdeae506892d4a44ca2))
* Update eval extras to Vertex SDK package version with constrained LiteLLM upperbound ([a86d0aa](https://github.com/google/adk-python/commit/a86d0aa7d70d1c70346eceef591d255ed96711c8))

## [2.0.0-alpha.1](https://github.com/google/adk-python/compare/v2.0.0-alpha.0...v2.0.0-alpha.1) (2026-03-18)

### Features

Introduces two major capabilities:
* Workflow runtime: graph-based execution engine for composing
  deterministic execution flows for agentic apps, with support for
  routing, fan-out/fan-in, loops, retry, state management, dynamic
  nodes, human-in-the-loop, and nested workflows
* Task API: structured agent-to-agent delegation with multi-turn
  task mode, single-turn controlled output, mixed delegation
  patterns, human-in-the-loop, and task agents as workflow nodes
