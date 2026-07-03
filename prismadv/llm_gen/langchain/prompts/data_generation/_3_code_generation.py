CODE_SYNTHESIS_PROMPT = """You are on step 3 in the pipeline. Given a dataset, a task description, and task-level assumptions for one or more target column groups (each group may consist of a single column or multiple columns), write downstream task code that realistically could exist in industry and that implicitly relies on those assumptions. This code will be the input to an assumption-extraction model.

The code must include executable data assertions to verify whether the incoming data satisfies the assumptions. Assertions should be implemented as `assert` statements or explicit checks that raise `AssertionError`. Do not use comments to describe the assumptions; the checks must be inferred from the code logic.

### About the code

- The code must reflect the assertions. If an assertion is violated, the code must fail or misbehave in a way that reflects the violated assumption. Don't just add assertions that reflect the assumptions, but don't leverage the assumptions in the code.

### How to add assertions
- Every assumption check must be enclosed between `# ASSERTION_START` and `# ASSERTION_END`.
- Inside that block, you may write a single `assert` line or multiple lines of code culminating in an `assert`.
- The exact form of the assertion is flexible (simple boolean assert, multi-line precomputation + assert, etc.).
- Avoid natural-language comments inside; the block itself signals that the code expresses an assumption.
- Post-processing can generate a version without assertions by stripping all lines between `# ASSERTION_START` and `# ASSERTION_END`.

For example. Here is a code snippet that processes a dataset with columns `age`, `salary`, and `monthly_income`.
```
import pandas as pd
import numpy as np

df = pd.read_csv("previous_data.csv")

# ASSERTION_START
# Single-line check
assert (df["age"] >= 0).all()
# ASSERTION_END

log_age = np.log(df["age"])

# ASSERTION_START
# Multi-line check with precomputation
def check_positive_salary(salary_series):
    positive_salary = df["salary"] > 0
assert check_positive_salary(df["salary"]).all()
# ASSERTION_END

income_to_age_ratio = df["monthly_income"] / df["salary"]
```

### Requirements
- Use a realistic stack, like Python with Pandas or Polars. Do not use external databases or data sources.
- Implement the described task end-to-end at a minimal viable level.
- Include assertion checks that fail when assumptions are violated and pass otherwise.
- The code must really require the assumptions to run correctly on real-world data.
- Do not include any comments that reveal assumptions; the assertions themselves must encode them. If you want to add comments about the assertion. Add them insert the block. Any comments about the assertion with lead to leakage.
    Instead of the following code
    ```
    # Single-line check
    # ASSERTION_START
    assert (df["age"] >= 0).all()
    # ASSERTION_END
    ```
    Do it this way
    ```
    # ASSERTION_START
    # Single-line check
    assert (df["age"] >= 0).all()
    # ASSERTION_END
    ```
- Don't put all assertions at the start; distribute them logically throughout the code. An advantage of this is you can use intermediate variables to express complex assumptions.
- The code should be executable when user remove the assertion block. So don't only define some parameter inside the assertion block but still use it out of the assertion block.
- Code equivalence with and without assertions:
    - Removing every `# ASSERTION_START … # ASSERTION_END` block must not break the program.
    - Variables or functions used in the main logic cannot be defined only inside an assertion block.
    - The semantics of the program outside the assertions must remain unchanged.


#### Assertion quality rules
- *Guard before use*. Place assertions immediately before code that would fail without them. Don't put all assertions at the start of the script. Keep assertions close to the code that depends on them.
- Boundaries explicit. Encode inclusive/exclusive ends exactly as stated in assumptions (e.g., >= 55 goes to “55+”).
- Keep ASSERT blocks self-contained. If intermediate steps are needed, define a local check_* function inside the block and end with a single assert.
- Keep the code clean. Put assertion logic, including helper functions, and **related comments** inside the ASSERT block. Do not define helper functions outside the block as we need to be able to remove all assertion-related code easily and keep the main code clean to test downstream logic.
- Avoid assertions that are unnecessarily strict and reject valid data that the code can handle.
- Balance tightness and looseness. Assertions must reflect exactly what the code needs for correct execution and valid outputs.
    - *Not too loose*: Do not allow values that silently break downstream logic. This will lead to the erroneous error being distinguished as the safe error. For example, 
        ```
        # ASSERTION_START
        # BAD: Allows None values, which propagate silently and cause issues downstream
        assert (df['age'].isna() | (df['age'] > 0)).all()
        # ASSERTION_END
        log_age = np.log(df["age"])
        ```
    - *Not too tight*: Do not add constraints the code never uses. For example, asserting a high correlation between MonthlyIncome and SalarySlab is too strict if the downstream code never depends on the correlation. It will lead to making the safe error be distinguished from an erroneous error. For example,
        ```
        # ASSERTION_START
        # BAD: The code does not depend on correlation, so this is too strict
        assert df['MonthlyIncome'].corr(df['SalarySlab'].map({{'Upto 5k': 1, '5k-10k': 2, '10k-15k': 3, '15k+': 4}})) > 0.7
        # ASSERTION_END
        # Code does not depend on correlation, so this is unnecessary
        df['IncomeLevel'] = pd.cut(df['MonthlyIncome'], bins=[0, 5000, 10000, 15000, np.inf], labels=['Upto 5k', '5k-10k', '10k-15k', '15k+'])
        ```

### Code IO
We provide input_path and output_path as command-line arguments. The input data is a CSV file located at `input_path/previous_data.csv`. Read the input CSV from `input_path/previous_data.csv` and write any outputs to `output_path/`.
For example, you can include the following code snippet to parse the arguments:
```python
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()
```

### Output format
Return strictly valid JSON (no markdown fences, no commentary).
```
{{
  "result": {{
    "code": "<full code as a single string>",
    "language": "python",
  }}
}}
```
Note: The `source` field must be either `data statistics` or `domain knowledge`.

### Inputs
Task description:
{task_description}

Table name: `{table_name}`.
The table profile:
{table_profile}

Example rows:
{example_rows}

Assumptions:
{assumptions}
"""
