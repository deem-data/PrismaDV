<!--
This file IS the prompt for the `codex` baseline. Edit it freely;
nothing in the Python needs to change when you do.

Everything outside an HTML comment is sent to the agent verbatim, as `task.txt`
in its workspace. Comments like this one are stripped before sending, so you can
leave notes here for yourself.

Placeholders are `{{NAME}}`. They are substituted by plain text replacement --
there is no format-string escaping, so braces, backslashes and JSON in the body
below need no special treatment. Any `{{...}}` not on this list is an error
rather than something silently passed through, so a typo fails loudly instead of
reaching the agent:

  {{CODE}}          REQUIRED -- the pipeline source, with line numbers
  {{DATA_SAMPLE}}   REQUIRED -- sampled rows per input table
  {{TABLES_DESC}}   profiles of the input tables the pipeline reads
  {{TASK}}          what the pipeline is for, downstream of the data
  {{DATA_DIR}}      workspace-relative directory holding the staged observed tables
  {{CODE_DIR}}      workspace-relative directory holding the same source files as
                    {{CODE}}, in their repository layout
  {{CHECKER_CMD}}   the command that runs a constraint file through PyDeequ

The two REQUIRED ones must appear somewhere below, or the agent would be asked
to constrain a pipeline it was never shown. The others are yours to use, move, or
leave out.

This document is the EIDBench-real counterpart of
`eid_bench_baselines/prompts/codex.md`. The two rows are the same method on
different benchmarks, so a change to how one instructs the agent probably wants
making in both.
-->

# Task

You are writing a **data unit test**: a set of PyDeequ constraints describing what
the pipeline below assumes about its input data. If new data arrived that violated
one of your constraints, the pipeline should be expected to misbehave -- crash,
silently produce a wrong number, or drop records it should have kept.

Your output is a single file, `constraints.json`, in the current folder.

# The pipeline

This code consumes the input tables.

```python
{{CODE}}
```

The same files are on disk under `{{CODE_DIR}}`, laid out as in the repository,
so you can open, search and trace them rather than working from the listing
alone. The `# File:` headers above are paths inside that directory.

What it is for, downstream of the data:

{{TASK}}

# The data

Profiles of the input tables the pipeline reads:

{{TABLES_DESC}}

Rows sampled from each of them:

{{DATA_SAMPLE}}

The full tables are on disk at `{{DATA_DIR}}`, one CSV per table. You have
complete access to them: load them with pandas, profile them, plot them, run
whatever analysis helps. They are throwaway copies -- nothing you do to them has
any effect outside this workspace.

# Verifying your constraints

You can execute PyDeequ constraint code against a staged table and read back the
result of every check:

```
{{CHECKER_CMD}}
```

where the table is named by its file stem and the constraints file is plain text
with one PyDeequ `Check` call per line, for example:

```
.isComplete("age")
.isNonNegative("age")
.isContainedIn("status", ["active", "closed"])
```

Each line comes back as `[HOLDS]` or `[FAILS]`, and a failure prints the reason,
including compilation errors. Use it as often as you like.

# Output format

Write `constraints.json` in the current folder, containing the union of what you
settled on for every column:

{
    "constraints": [
        {
            "table_name": "some_table",
            "column_name": "some_column",
            "code_for_constraint": ".isComplete('some_column')"
        },
        {
            "table_name": "another_table",
            "column_name": "another_column",
            "code_for_constraint": ".isPositive('another_column')"
        }
    ]
}

Use only the PyDeequ `Check` object's methods in `code_for_constraint`. Each
entry names the single table and column it constrains; for a constraint spanning
several columns of one table, name the one it is primarily about. Constraints are
checked against one table at a time, so do not write one that spans two tables.
