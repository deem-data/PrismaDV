from inspect import cleandoc

MULTI_COLUMN_ASSUMPTION_GENERATION_PROMPT = cleandoc("""
You are part of a task-aware data validation system. You serve as the *Multi-Column Assumption Generation* component.
When people write code, they often make assumptions about the data they are working with. These assumptions are not always explicitly stated, but they are crucial for the code to work correctly.
For example, if the code doesn't fillna on a column and performs operations that require the column to have no null values, it assumes the column should not have null values.

Given a dataset and the downstream code. Your goal is to extract assumptions that involve multiple columns based on data characteristics and the downstream task.
We will provide you with the code that highlights the code where multiple columns are used together. You need to provide assumptions that involve multiple columns based on the code and data characteristics.

The assumption should be used as the basis for generating multi-column constraints written in Deequ or other constraint languages. Thus, please ensure that the assumptions are precise and actionable.

Your response should be a JSON-formatted string with the following structure:
```json
{{"assumptions": [
    {{
        "text": "<Assumption 1>",
        "sources":
            [{{
                "start_line": 1,
                "end_line": 3 # should be integers representing the line numbers, don't add 0 as a prefix.
            }},
            {{
                ...
            }}],
    }}
    {{
        "text": "<Assumption 2>",
        "sources": [...]
    }}
]}}
```

Here is one example of an assumption with multiple columns we expect:
    Given the code snippet below:
    ```python
        0001: df["full_name"] = df["first_name"] + " " + df["last_name"]
        0002: assert df["full_name"].is_unique, "Full names must be unique"
    ```
    The generated assumption should be:
    ```json
    {{"assumptions": [
        {{
        "text": "The combination of 'first_name' and 'last_name' columns should be unique across all records in the dataset.
        We could represent this assumption as a deequ constraint like this:
        .isUnique(["first_name", "last_name"])
        ",
        "sources": [{{
            "start_line": 1,
            "end_line": 2
        }}],
        }}
    ]}}
    ```
    
Here are the information you will need:

Target columns:
{target_columns}

Dataset columns description:
{target_columns_desc}

The user writes the code snippet below:
{focused_code}

Downstream task description:
{downstream_task_description}

**Important:** JSON numbers **must not have leading zeros**. For example, use `1` instead of `0001`.

""")
