---
type: Skill
title: Run an Attested Computation on BigQuery
description: "Executor skill for `Attested Computation` concepts with `runtime: bigquery`. Binds declared parameters, submits the job, and returns a receipt the attester will verify."
tags: [skill, executor, bigquery]
generated: { by: human:kliu@acme, at: 2026-06-30T14:00:00Z }
status: stable
---

# Skill: run on BigQuery

## When to use

The `executor.resource` field of an `Attested Computation` points here when the computation's `runtime` is `bigquery`.

## Preconditions

- Caller has a service account with `bigquery.jobs.create` on the concept's `resource` project.
- Caller has read access on every table referenced by the computation's `# Computation` fence (or the file at `computation:`).
- The concept declares its `parameters:` list. Every required parameter has been supplied a value by the caller.

## Steps

1. **Load the computation.** Read the `# Computation` fence from the concept body, or the file at `computation:` if the field is set. The result is a SQL string containing `@name`-style bind variables for each declared parameter.
2. **Bind parameters.** Pass the caller-supplied values as BigQuery [named query parameters](https://cloud.google.com/bigquery/docs/parameterized-queries). Do NOT string-interpolate; the attester will reject a receipt whose `executed_sql` shows literal substitution.
3. **Submit the job.** Use `jobs.query` (or `jobs.insert` with a `Query` configuration). Set `useLegacySql: false`. Set the job's `labels` to include `okf_concept: <bundle-relative-path>` for auditability.
4. **Wait for completion.** Poll `jobs.get` until `status.state = 'DONE'`. If `status.errorResult` is present, return the receipt with `result: null` and `error: <errorResult>` so the attester can distinguish "sanctioned SQL that failed at runtime" from "the executor ran the wrong SQL."
5. **Assemble the receipt.** Return exactly the fields declared in `executor.receipt`. For this skill:

    ```json
    {
      "job_id": "bq://<project>/us/<jobId>",
      "executed_sql": "<queryConfig.query with parameters shown as @name>",
      "result": "<the first result row's cell values, in declared select-order>"
    }
    ```

6. **Never modify the computation.** If the caller-supplied parameters cannot be bound (missing required, wrong type), refuse and return an error receipt. Do NOT rewrite the SQL to work around missing parameters.

## Post-conditions

The receipt is handed to the concept's `attester.resource`. Do not display the value to the user until the attester returns `verdict: ok`.
