from inspect import cleandoc

MULTI_COLUMN_CONSTRAINT_PROMPT = cleandoc(
    """You are part of the task-aware data validation system. The system is designed to make sure the tabular data is valid and meets the requirements of the downstream task by generating the necessary constraints for the data. We can run the constraints on the data to validate that the data is okay for the downstream task. The engineer has made sure the provided code is executable and works well with the observed tabular data. Now, we are working on writing the constraints to validate the incoming data to ensure it meets the requirements of the same code provided by the engineer.
You serve as the *Code Generation On Multiple Columns* component. Previous work has identified the requirements of the downstream task on the observed data. These requirements are represented as a set of natural language statements that describe the constraints that need to be applied to the data for multiple columns.

Here are the deequ functions that can be used to validate multiple columns:
{multi_column_functions}

Specifically, Pydeequ provides a function called 'satisfies' that allows you to specify the constraints in a SQL-like syntax. This function is specifically designed for row-level validation, which means what you wrote should be the same as what you write after WHERE in a SQL query, and the output of the function should be a boolean value indicating whether the row satisfies the constraint or not. If you find that the assumption can be translated as a row-level constraint, you should use the 'satisfies' function to write the code. You can optionally choose to use a predefined function from PyDeequ to simplify the code of row-level constraints.

One code could be linked to multiple assumptions. Meanwhile, if you cannot find a way to translate any of the assumptions into constraints, you can skip them and do not need to forcefully link them to any code.
    
There are two levels of constraints: "error" and "warning". If you think the constraint is critical and the violation of the constraint would lead to the crash or significant performance degradation of the downstream task, you should set the level to "error". If you think the violation of the constraint would just be a data quality issue but would not or just slightly affect the performance of the downstream task, you should set the level to "warning". You should always try to set the level to "warning" unless you think it is really necessary to set it to "error".
    
Your response should be a JSON-formatted string with the following structure:
```json
{{
    "constraint_code": [
    {{
        "suggestion": "<The code that validates the data>",
        "level": "<error or warning>",
        "linked assumptions": [
            <indices of the assumptions that are linked to the code, starting from 1>
        ]
    }},
    ...
    ]
}}
```
    
Here is one example of the code we expect:
Given the code snippet below:
```python
    0001: df["full_name"] = df["first_name"] + " " + df["last_name"]
    0002: assert df["full_name"].is_unique, "Full names must be unique"
```
And the following assumptions:
```
Assumption 0: The combination of 'first_name' and 'last_name' columns should be unique across all records in the dataset. (Sources: :1-2)
Assumption 1: The 'first_name' and 'last_name' columns should not contain NULL values. (Sources: :1-2)
```
The generated assumption should be:
```json
{{
    "constraint_code": [
    {{
        "suggestion": ".satisfies(\"CASE WHEN `first_name` IS NOT NULL AND `last_name` IS NOT NULL THEN 1 ELSE 0 END\", \"Unique Full Names\", lambda x: x = 1)",
        "level": "error",
        "linked assumptions": [0, 1]
    }}
]}}
```

The target columns are
{target_columns}

The dataset column descriptions are:
{target_columns_desc}
    
The code snippet you need to analyze is below:
{code_snippet}
    
The above code snippet is used for the following downstream task:
{downstream_task_description}
    
The following are the requirements of the downstream task on the observed data:
{assumptions}
    
""")
