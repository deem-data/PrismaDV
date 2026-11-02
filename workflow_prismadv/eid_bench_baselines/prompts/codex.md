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

  {{CODE}}          REQUIRED -- the program under analysis, assertions stripped
  {{CODE_PATH}}     workspace-relative path of that same program, staged as a .py file
  {{DATA_SAMPLE}}   REQUIRED -- three sampled rows, as a list of dicts
  {{TABLE_PATH}}    workspace-relative path of the staged CSV
  {{TABLE_NAME}}    the table's name, as the runner's --table wants it
  {{CHECKER_CMD}}   the command that runs a constraint file through PyDeequ

The two REQUIRED ones must appear somewhere below, or the agent would be asked
to constrain a program it was never shown. The others are yours to use, move, or
leave out.

To see exactly what the agent will receive, render it without spending a run:

  python -m workflow_prismadv.eid_bench_baselines.constraint_inference_codex \
      --dataset sleep_health --script general_task_0 --dump-prompt /tmp/task.txt
-->

# Task

You are writing a **data unit test**: a set of PyDeequ constraints describing what
the program below assumes about its input data. If new data arrived that violated
one of your constraints, the program should be expected to misbehave -- crash,
silently produce a wrong number, or drop records it should have kept.

Your output is a single file, `constraints.json`, in the current folder.

# The program

This program consumes the dataset.

```python
{{CODE}}
```

The same program is on disk at `{{CODE_PATH}}`, so you can run it against the
staged table, step through it, or search it, rather than working from the
listing alone.

# The data

Three rows sampled from the dataset the program consumes:

{{DATA_SAMPLE}}

The full table is on disk at `{{TABLE_PATH}}`. You have complete access to it:
load it with pandas, profile it, plot it, run whatever analysis helps. It is a
throwaway copy -- nothing you do to it has any effect outside this workspace.

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

Write `constraints.json` in the current folder, containing the union of what you
settled on for every column:

{
    "constraints": [
        {
            "column_name": "some_column",
            "code_for_constraint": ".isComplete('some_column')"
        },
        {
            "column_name": "another_column",
            "code_for_constraint": ".isPositive('another_column')"
        }
    ]
}

Use only the PyDeequ `Check` object's methods in `code_for_constraint`. Each
entry names the single column it constrains; for a constraint spanning several
columns, name the one it is primarily about.
