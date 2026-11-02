<!--
To see exactly what the agent will receive, render it without spending a run:

  python -m workflow_prismadv.icd_bench_experiments.codex \
      --case access_dependency_forloop --dump-prompt /tmp/task.txt
-->

# Task

You are writing a **data unit test**: a set of PyDeequ constraints describing what
the program below assumes about its input data. If new data arrived that violated
one of your constraints, the program should be expected to misbehave -- crash,
silently produce a wrong number, or drop records it should have kept.

Your output is a single file, `constraints.json`, in the current folder.

# The program

This code consumes the dataset as the dataframe `df`. It is an excerpt of a
larger program, so names it uses but does not define come from that program.

```python
{{CODE}}
```

The same code is on disk at `{{CODE_PATH}}`, so you can search it or run parts
of it against the staged table rather than working from the listing alone.

# The data

A sample of data that the program can process:

{{DATA_SAMPLE}}

The full sample is on disk at `{{TABLE_PATH}}`. You have complete access to it:
load it with pandas, profile it, run whatever analysis helps. It is a throwaway
copy -- nothing you do to it has any effect outside this workspace.

# Verifying your constraints

You can execute PyDeequ constraint code against the staged table and read back
the result of every check:

```
{{CHECKER_CMD}}
```

where the constraints file is plain text with one PyDeequ `Check` call per line,
for example:

```
.isComplete("age")
.isNonNegative("age")
.isContainedIn("status", ["active", "closed"])
```

Each line comes back as `[HOLDS]` or `[FAILS]`, and a failure prints the reason,
including compilation errors. Use it as often as you like.

# Output format

Write `constraints.json` in the current folder, in this format:

{
    "constraints": [
        {
            "column_name": "some_column",
            "code_for_constraint": ".isComplete('some_column')"
        },
        {
            "column_name": "some_column",
            "code_for_constraint": ".isPositive('some_column')"
        }
    ]
}

Return constraints for the `{{TARGET_COLUMN}}` column only.

Use only the PyDeequ `Check` object's methods in `code_for_constraint`. Each
entry names the single column it constrains; for a constraint spanning several
columns, name the one it is primarily about.
