from inspect import cleandoc

DIRECT_CODE_GENERATION_PROMPT = cleandoc("""
You are part of a task-aware data validation system. You serve as the *Column Code Generation* component.
When people write code, they often require the data that the data working with meeting some requirements.
For example, if the code didn't do fillna on a column but do log operation on it, it require that the column have no null value. On the other hand, if the code have fillna on a column and then performs operations that require the column to have no null values, it assumes the column could have null values. 

Given a dataset and the downstream code. Your goal is to generate PyDeequ code for the target column based on data characteristics and the downstream task.
We will provide you with the code that highlights the data flow of the target column. You will then generate PyDeequ constraints for the target column based on the code and data characteristics.

Specifically, Pydeequ provides a function called 'satisfies' that allows you to specify the constraints in a SQL-like syntax. This function is specifically designed for row-level validation, which means what you wrote should be the same as what you write after WHERE in a SQL query, and the output of the function should be a boolean value indicating whether the row satisfies the constraint or not. You can optionally choose to use a predefined function from PyDeequ to simplify the code of row-level constraints. The predefined row-level functions provided by PyDeequ are:
{row_level_functions}

Here are the deequ functions that can be used to validate multiple columns:
{multi_column_functions}

If you do not find a suitable predefined function, you can write the code using the 'satisfies' function directly.

There may be cases where you want to validate at the aggregate level, for example to check a mean, standard deviation or the number of distinct values. In this case, you can use the following aggregate-level constraints provided by PyDeequ:
{aggregate_level_functions}

The target column (group) is
{target_column}

The dataset column descriptions are:
{target_column_desc}

The user writes the code snippet below:
{code_snippet}

The above code snippet is used for the following downstream task:
{downstream_task_description}

Your response should be a JSON-formatted string with the following structure:
```json
{{
    "constraint_code": [
    {{
        "suggestion": "<The code that validates the data>",
        "level": <"error"|"warning">
    }},
    ...
    ]
}}
```
Here is one example of the code we expect:
Given the code snippet below:
```python
    0001: blood_type_categories = ["A", "B", "AB", "O", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-", "Rh-null", "Rare"]
    0002: df["Blood Type"] = df["Blood Type"].apply(lambda x: x if x in blood_type_categories else "UNKNOWN")
    0003: assert df[df["Blood Type"] == "UNKNOWN"].shape[0] <= 0.05 * df.shape[0], "More than 5% of Blood Type values are not in the expected categories"
```
The generated code should be:
```json
{{
    "constraint_code": [
    {{
        "suggestion": ".satisfies(\"CASE WHEN `Blood Type` IN ('A', 'B', 'AB', 'O', 'A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-', 'Rh-null', 'Rare') THEN 1 ELSE 0 END\", \"Valid Blood Types\", lambda x: x >= 0.95)",
        "level": "error",
    }},
    {{
        "suggestion": ".satisfies("`Blood Type` is not NULL", \"high quality data\", lambda x: x == 1)",
        "level": "warning",
    }},
    ]
}}
```

Given the code snippet below:
```python
    0001: df["full_name"] = df["first_name"] + " " + df["last_name"]
    0002: assert df["full_name"].is_unique, "Full names must be unique"
```
The generated code should be:
```json
{{
    "constraint_code": [
    {{
        "suggestion": ".satisfies("COUNT(*) = COUNT(DISTINCT (first_name, last_name))", "full_name_unique")",
        "level": "error",
    }}
    ]
}}
```

There are two levels of constraints: "error" and "warning". The "error" level means that the data will crash the code if it does not satisfy the constraint, while the "warning" level means that the data will not crash the code, but the quality of the data is not ideal. You should use the "error" level for constraints that are critical for the code to run correctly, and the "warning" level for constraints that are not critical but still important for the quality of the data.
""")
