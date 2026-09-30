# Documentation and Comments

## Public API Documentation

- **Classes**: explain the intended usage, with a concise example when the
  usage is not obvious from the signature. Document every public attribute.
  For Pydantic models, document fields with an attribute docstring under the
  field — see the Pydantic reference.
- **Methods and functions**: document every argument, the return value, and
  each exception raised.

## Internal Implementation Comments

- **Explain why, not what.** The code says what it does; a comment earns its
  place by saying why it does it that way — the constraint, the bug, or the
  ordering requirement that is not visible from the code.
- **No links to RFCs, design docs, issues, or pull requests.** They rot faster
  than the code, and a reader who cannot open the link is left with nothing.
  Put the reasoning in the comment itself and the link in the pull request.
