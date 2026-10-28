# Fixture provenance

`crypto_bitcoin/`, `ga4/`, `stackoverflow/`, `acme_retail/` are vendored verbatim from

- **Repository:** https://github.com/GoogleCloudPlatform/open-knowledge-format
- **Commit:** `ad30107`
- **Path:** `bundles/`
- **License:** Apache-2.0 (see `LICENSE.md` in this directory)

They are checked in rather than downloaded so the test suite stays offline and
deterministic, and so the golden tests pin the §11 conformance facts against the
real corpus rather than against a fixture we wrote ourselves.

Two deliberate deviations from verbatim:

- Each bundle's `viz.html` viewer is omitted; it is not part of the bundle
  semantics and nothing here reads it.
- The repository's pre-commit hooks normalise trailing whitespace and final
  newlines in the `.md` files. The concept frontmatter and bodies are otherwise
  untouched.

Do not edit these files. If the upstream bundles change, re-vendor and update
`tests/unittests/integrations/okf/test_bundles.py` deliberately.
