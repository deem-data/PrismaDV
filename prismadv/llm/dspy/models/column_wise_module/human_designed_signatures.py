"""Human-designed DSPy signatures with detailed prompts.

These signatures preserve the full prompt content from the original LangChain implementation,
providing more detailed instructions compared to the auto-optimizable signatures.
"""

from typing import Any, Dict, List

import dspy

from prismadv.ir_translator.deequ_constraints.function_manager import DeequFunctionManager


class HumanDesignedDataFlowInspectorSig(dspy.Signature):
    """You are an expert in data-flow analysis and code inspection.
Given a code snippet, a column name, and (optionally) a sink variable, list every line that touches the column—reads it, writes to it, filters on it, transforms it, merges it, or passes it along.
If a sink variable is supplied, include only the lines that ultimately affect that sink. If no sink is given, include every reference through to the end of the snippet.

Output (1-based line numbers)
Return only a JSON object whose "sources" field contains the contiguous line-number ranges that reference or manipulate the column:
```json
{
"sources": [
    { "start_line": 2, "end_line": 2 },
    { "start_line": 5, "end_line": 7 }
    ]
}
```

Rules
1. Use 1-based line numbers exactly as they appear in the snippet.
2. Be exhaustive: any line that interacts with the column must appear in at least one range.
3. Adjacent lines go in the same range; non-adjacent lines go in separate ranges.
4. The order inside "sources" does not matter.
5. If sink_variable is empty or omitted, consider the entire snippet.
6. Output only the JSON—no extra text or formatting.

Mini-example

1:  df = pd.read_csv("employees.csv")
2:  df["age"] = df["age"].fillna(0)
3:  df["age_group"] = pd.cut(df["age"], bins=[0,17,64,120], labels=["child","adult","senior"])
4:  df["salary_eur"] = df["salary_usd"] * 0.9
5:  engineering = df[df["department"] == "Engineering"]
6:  high_earners = engineering[engineering["salary_eur"] > 80000]
7:  result = high_earners[["name","age","salary_eur","age_group"]].copy()
8:  result.reset_index(drop=True, inplace=True)
9:  final_df = result

-> Column: age and Sink: result_df
```json
{
    "sources": [
        {"start_line": 1, "end_line": 3},
        {"start_line": 7, "end_line": 9},
    ]
}
```

We will use a JSON parser to parse your output, so please ensure that your output is valid JSON. The line number should be valid 1-based line numbers.
For example, 0073 is not a valid line number, but 73 is.

Inputs:
- code_script: The code snippet to analyze with line numbers.
- target_column: The column name to track.
- sink_variable: Optional sink variable to focus on. If empty, consider the whole snippet.

Output:
- sources: Return only valid JSON: { "sources": [{ "start_line": int, "end_line": int }, ... ] }.
  Use exact 1-based line numbers from the snippet.
  Include every line that reads, writes, filters, transforms, merges, or passes through the target column.
  Merge adjacent lines into one range; non-adjacent lines must be separate ranges.
  If sink_variable is not empty, include only lines that ultimately affect that sink.
"""

    code_script: str = dspy.InputField()
    target_column: str = dspy.InputField()
    sink_variable: str = dspy.InputField()
    sources: List[Dict[str, int]] = dspy.OutputField()


class HumanDesignedAssumptionGenerationSig(dspy.Signature):
    """You are part of a task-aware data validation system. You serve as the *Column Assumption Generation* component.
When people write code, they often make assumptions about the data they are working with. These assumptions are not always explicitly stated, but they are crucial for the code to work correctly.
For example, if the code does fillna on a column, it assumes that the column could have null values. On the other hand, if the code doesn't fillna on a column and performs operations that require the column to have no null values, it assumes the column should not have null values.

Given a dataset and the downstream code. Your goal is to generate assumptions for the target column based on data characteristics and the downstream task.
We will provide you with the code that highlights the data flow of the target column. You will then generate data assumptions for the target column based on the code and data characteristics.

Here is one example of an assumption we expect:
    Given the code snippet below:
    ```python
        0001: blood_type_categories = ["A", "B", "AB", "O", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-", "Rh-null", "Rare"]
        0002: df["Blood Type"] = df["Blood Type"].apply(lambda x: x if x in blood_type_categories else "UNKNOWN")
        0003: assert df[df["Blood Type"] == "UNKNOWN"].shape[0] <= 0.05 * df.shape[0], "More than 5% of Blood Type values are not in the expected categories"
    ```
    The generated assumption should be:
    ```json
    {"assumptions": [
        {
        "text": "The 'Blood Type' column should not contain more than 5% of values which are not in A, B, AB, O, A+, A-, B+, B-, AB+, AB-, O+, O-, Rh-null, or Rare categories.
        We could represent this assumption as a deequ constraint like this:
        .satisfies(\"CASE WHEN `Blood Type` IN ('A', 'B', 'AB', 'O', 'A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-', 'Rh-null', 'Rare') THEN 1 ELSE 0 END\", \"Valid Blood Types\", lambda x: x >= 0.95)
        ",
        "sources": [{
            "start_line": 1,
            "end_line": 3
            }]
        }
    ]}
    ```
    Given the code snippet below:
    ```python
        0001: # fill missing values in the 'Age' column with the mean age
        0002: df["Age"] = df["Age"].fillna(df["Age"].mean())
        0003: assert df["Age"].min() >= 18, "Age should be at least 18"
    ```
    The generated assumption should be:
    ```json
    {"assumptions": [
        {
        "text": "The 'Age' column could be null because the code fills missing values with the mean age during the data processing. However, not all values can be null, as some are required to calculate the mean age.
        We could represent this assumption as a deequ constraint like this:
        .satisfies(\"CASE WHEN `Age` IS NOT NULL THEN 1 ELSE 0 END\", \"Null Age Values\", lambda x: x > 0)
        "sources": [{
            "start_line": 1,
            "end_line": 2
            }]
        },
        {
        "text": "The 'Age' column should have a minimum value of 18, as the code asserts that the minimum age is at least 18.
        We could represent this assumption as a deequ constraint like this:
        .satisfies(\"CASE WHEN `Age` <= 18 AND `Age` IS NOT NULL THEN 1 ELSE 0 END\", \"Minimum Age\", lambda x: x == 0)
        "sources": [{
            "start_line": 3,
            "end_line": 3
            }]
        }
    ]}
    ```

The assumptions will be used to generate validation rules to ensure the *input data* meets the code's expectations and requirements. Thus, they should be assumptions on the *input data* rather than the intermediate data on the data flow.
If you find some constraints on the intermediate data, you should convert them into assumptions on the input data. Please describe the assumptions in a way that makes it possible to convert them into Deequ or Great Expectations validation rules.
Please ensure the JSON is *valid* as we will parse it programmatically.

**Important:** JSON numbers **must not have leading zeros**. For example, use `1` instead of `0001`.

Inputs:
- target_column: The target column name to generate assumptions for.
- target_column_desc: YAML-formatted description of the target column including its schema and metadata.
- focused_code: The code snippet with highlighted line numbers showing the data flow of the target column.
- downstream_task_description: Description of the downstream task that the code is designed to perform.

Output:
- assumptions: List of assumption objects in JSON format: { "assumptions": [{ "text": str, "sources": [{ "start_line": int, "end_line": int }] }, ...] }.
  Each assumption should describe a data requirement that can be converted to a Deequ constraint.
  JSON numbers must not have leading zeros.
"""

    target_column: str = dspy.InputField()
    target_column_desc: str = dspy.InputField()
    focused_code: str = dspy.InputField()
    downstream_task_description: str = dspy.InputField()
    assumptions: List[Dict[str, Any]] = dspy.OutputField()


class HumanDesignedIRGenerationSig(dspy.Signature):
    __doc__ = f"""You are part of the task-aware data validation system. The system is designed to make sure the tabular data is valid and meets the requirements of the downstream task by generating the necessary constraints for the data. We can run the constraints on the data to validate that the data is okay for the downstream task. The engineer has made sure the provided code is executable and works well with the observed tabular data. Now, we are working on writing the constraints to validate the incoming data to ensure it meets the requirements of the same code provided by the engineer.
You serve as the *IR Generation* component. Previous work has identified the requirements of the downstream task on the observed data. These requirements are represented as a set of natural language statements that describe the constraints that need to be applied to the data for each column.
You need to translate these natural language statements into executable data validation code that can be run on the data. The code will be run with PyDeequ, a library for data validation in Python. Pydeequ provides a set of functions that can be used to validate the data. You should use these functions to write the code that will validate the data.

The functions provided by PyDeequ include:

Row-Level Functions:
{chr(10).join(DeequFunctionManager().get_constraints(is_row_level=True))}

Aggregate-Level Functions:
{chr(10).join(DeequFunctionManager().get_constraints(is_row_level=False))}

Here is one example of the code we expect:
Given the code snippet below:
```python
    0001: blood_type_categories = ["A", "B", "AB", "O", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-", "Rh-null", "Rare"]
    0002: df["Blood Type"] = df["Blood Type"].apply(lambda x: x if x in blood_type_categories else "UNKNOWN")
    0003: assert df[df["Blood Type"] == "UNKNOWN"].shape[0] <= 0.05 * df.shape[0], "More than 5% of Blood Type values are not in the expected categories"
```
And the following assumptions:
```
Assumption 0: The 'Blood Type' column should not contain more than 5% of values which are not in A, B, AB, O, A+, A-, B+, B-, AB+, AB-, O+, O-, Rh-null, or Rare categories. (Sources: :1-3)
Assumption 1: The 'Blood Type' column should not contain NULL values. (Sources: :1-3)
```
The generated code should be:
```json
{{
    "constraint_code": [
    {{
        "suggestion": ".satisfies(\\"CASE WHEN `Blood Type` IN ('A', 'B', 'AB', 'O', 'A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-', 'Rh-null', 'Rare') THEN 1 ELSE 0 END\\", \\"Valid Blood Types\\", lambda x: x >= 0.95)",
        "level": "error",
        "linked assumptions": [0]
    }},
    {{
        "suggestion": ".satisfies(\\"`Blood Type` is not NULL\\", \\"high quality data\\", lambda x: x == 1)",
        "level": "warning",
        "linked assumptions": [1]
    }},
    ]
}}
```

One code could be linked to one or more assumptions. Please make sure the index of the linked assumptions is correct. The index starts from 0. If the code does not reflect any assumptions, you should leave the linked assumption as an empty list. An index that is out of range will lead to an out-of-range error when we parse the output.

There are two levels of constraints: "error" and "warning". The "error" level means that the data will crash the code if it does not satisfy the constraint, while the "warning" level means that the data will not crash the code, but the quality of the data is not ideal. You should use the "error" level for constraints that are critical for the code to run correctly, and the "warning" level for constraints that are not critical but still important for the quality of the data.
Linked assumptions are the indices of the assumptions that are linked to the constraint code. It could be one or more indices, depending on which assumptions are reflected in the code. If the code does not reflect any assumptions, you should leave the linked assumption as an empty list.

Inputs:
- target_column: The target column name to generate constraints for.
- target_column_desc: YAML-formatted description of the target column including its schema and metadata.
- code_snippet: The full code snippet with line numbers.
- downstream_task_description: Description of the downstream task that the code is designed to perform.
- assumptions: The natural language assumptions generated in the previous step, formatted as text with indices.

Output:
- constraint_code: List of constraint objects in JSON format: {{"constraint_code": [{{"suggestion": PyDeequ code (starting with "." or a function), "level": "warning"|"error", "linked assumptions": [indices]}}, ...]}}.
  The suggestion should be valid PyDeequ constraint code.
  The level should be "error" for critical constraints and "warning" for quality constraints.
"""

    target_column: str = dspy.InputField()
    target_column_desc: str = dspy.InputField()
    code_snippet: str = dspy.InputField()
    downstream_task_description: str = dspy.InputField()
    assumptions: Any = dspy.InputField()
    constraint_code: List[Dict[str, Any]] = dspy.OutputField()
