# Publish a retained ADK reference only after admission

This standalone consumer sample runs an already installed reader twice before
publishing its admission receipt. It is for the public twelve-case
[Probity ADK plugin reference](https://github.com/probityai/agent-evidence-observer/tree/c774e0711e4a30c31cdfb184c1e4495a249fbdd0/interop/adk-ticket-2026-10-02),
which exercises plugin callback ordering, protected ticket dispatch, errors after
committed effects and incomplete capture. It does not accept arbitrary ADK logs,
judge model quality or establish independent custody of effects.

The operator selects the packet, external policy bytes and their SHA-256 digest,
absolute installed interpreter and reader entrypoint before execution. Keep the
policy and output outside the packet. Use a trusted installation; Python `-B`
prevents writing caches but does not reject existing cached code. The reference's
source-installed reader is separate from the producer SDK. There is no registry
release implied by this sample.

```sh
python publish_reference.py /absolute/retained-packet \
  --policy /absolute/host-policy.json \
  --policy-sha256 "$SELECTED_POLICY_SHA256" \
  --python /absolute/reader-venv/bin/python \
  --reader /absolute/reader-venv/bin/probity-adk-read \
  --output /absolute/new-publication-receipt
```

Both installed gate calls must succeed and emit identical literal decisions for
the selected reference population. Only then does `publication-receipt.json`
appear. Errors, timeouts, changed repeated output or a mismatched external policy
leave no publication receipt. Raw stdout, stderr and exit/timeout status are
retained for each attempted gate call. The existing reader's admission decision
is preserved; this sample adds no ADK plugin or framework API.

Credential-free current-SDK execution, source selection, normal reader wheel
installation and semantic refusal controls are maintained in the
[public reference repository](https://github.com/probityai/agent-evidence-observer).
This sample's unit tests live at
`tests/unittests/plugins/test_report_publication_sample.py` and run as part of the
ordinary ADK test suite. Run the full supported-version `tox` matrix before
submitting changes.
