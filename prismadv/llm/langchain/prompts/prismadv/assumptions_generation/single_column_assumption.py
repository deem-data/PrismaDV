from inspect import cleandoc

SINGLE_COLUMN_ASSUMPTION_GENERATION_PROMPT = cleandoc("""
You are part of a task-aware data validation system. You serve as the *Column Assumption Generation* component.
When people write code, they often make assumptions about the data they are working with. These assumptions are not always explicitly stated, but they are crucial for the code to work correctly.
For example, if the code does fillna on a column, it assumes that the column could have null values. On the other hand, if the code doesn't fillna on a column and performs operations that require the column to have no null values, it assumes the column should not have null values. 

Given a dataset and the downstream code. Your goal is to generate assumptions for the target column based on data characteristics and the downstream task.
We will provide you with the code that highlights the data flow of the target column. You will then generate data assumptions for the target column based on the code and data characteristics.

Target column:
{target_column}

Dataset columns description:
{target_column_desc}

The user writes the code snippet below:
{focused_code}

Downstream task description:
{downstream_task_description}

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

Here is one example of an assumption we expect:
    Given the code snippet below:
    ```python
        0001: blood_type_categories = ["A", "B", "AB", "O", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-", "Rh-null", "Rare"]
        0002: df["Blood Type"] = df["Blood Type"].apply(lambda x: x if x in blood_type_categories else "UNKNOWN")
        0003: assert df[df["Blood Type"] == "UNKNOWN"].shape[0] <= 0.05 * df.shape[0], "More than 5% of Blood Type values are not in the expected categories"
    ```
    The generated assumption should be:
    ```json
    {{"assumptions": [
        {{
        "text": "The 'Blood Type' column should not contain more than 5% of values which are not in A, B, AB, O, A+, A-, B+, B-, AB+, AB-, O+, O-, Rh-null, or Rare categories.
        We could represent this assumption as a deequ constraint like this:
        .satisfies("CASE WHEN `Blood Type` IN ('A', 'B', 'AB', 'O', 'A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-', 'Rh-null', 'Rare') THEN 1 ELSE 0 END", "Valid Blood Types", lambda x: x >= 0.95)
        ",
        "sources": [{{
            "start_line": 1,
            "end_line": 3
            }}]
        }}
  ni]}}
    ```
    Given the code snippet below:
    ```python
        0001: # fill missing values in the 'Age' column with the mean age
        0002: df["Age"] = df["Age"].fillna(df["Age"].mean())
        0003: assert df["Age"].min() >= 18, "Age should be at least 18"
    ```
    The generated assumption should be:
    ```json
    {{"assumptions": [
        {{
        "text": "The 'Age' column could be null because the code fills missing values with the mean age during the data processing. However, not all values can be null, as some are required to calculate the mean age.
        We could represent this assumption as a deequ constraint like this:
        .satisfies("CASE WHEN `Age` IS NOT NULL THEN 1 ELSE 0 END", "Null Age Values", lambda x: x > 0)
        "sources": [{{
            "start_line": 1,
            "end_line": 2
            }}]
        }},
        {{
        "text": "The 'Age' column should have a minimum value of 18, as the code asserts that the minimum age is at least 18.
        We could represent this assumption as a deequ constraint like this:
        .satisfies("CASE WHEN `Age` <= 18 AND `Age` IS NOT NULL THEN 1 ELSE 0 END", "Minimum Age", lambda x: x == 0) 
        "sources": [{{
            "start_line": 3,
            "end_line": 3
            }}]
        }}
    ]}}
    ```

The assumptions will be used to generate validation rules to ensure the *input data* meets the code's expectations and requirements. Thus, they should be assumptions on the *input data* rather than the intermediate data on the data flow.
If you find some constraints on the intermediate data, you should convert them into assumptions on the input data. Please describe the assumptions in a way that makes it possible to convert them into Deequ or Great Expectations validation rules.
Please ensure the JSON is *valid* as we will parse it programmatically.

**Important:** JSON numbers **must not have leading zeros**. For example, use `1` instead of `0001`.
""")
