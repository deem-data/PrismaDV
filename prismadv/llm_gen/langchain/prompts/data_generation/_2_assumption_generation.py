ASSUMPTION_GENERATION_PROMPT = """You are step 2 in a pipeline. Given a table from step 1, you must generate task-level assumptions for one or more target column groups.

### Requirements
- Infer a realistic downstream application in the dataset’s industry. Describe the task in one practical sentence.
- Choose some column groups that contains domain-specific knowledge and are meaningful for downstream tasks. A column group can be a single column or multiple columns.
- List 3–8 plausible, atomic assumptions/constraints that an engineer would impose on these columns in a real-world application. Each assumption should be:
  * Assumptions should be testable. When I say testable, I mean we could inject errors that violate the assumption and then the code which holds the assumption would crash or misbehave on the corrupted data.
  * They can come from observed data statistics or from domain knowledge.
  * Each assumption could focus on one column only, or involve correlations between multiple columns
    * For assumption which only on one column. Don't mention the correlation with other columns. You can do that in the multi-column assumption.
    * For assumption which involve multiple columns, the constraint must be a correlation between the columns.
- The set of assumptions should be complete enough that, if code is later generated, all ground-truth assumptions are already represented here. This ensures recall and precision can be computed against them.

### Output format
Return strictly valid JSON (no comments, no extra text):
{{
  "task_description": "<real-world application description>",
  "assumptions_on_column_groups": [
    {{
      "target_column_group": "<e.g., 'age' or 'height,weight'>",
      "assumptions": [
        {{ "assumption": "<assumption about the column group>", "source": "data statistics|domain knowledge" }}
      ]
    }}
  ]
}}
NOTE: The `source` field must be either `data statistics` or `domain knowledge`.

### Example 1 (single-column group)
{{
  "task_description": "Predicting customer churn for a telecom company based on demographics and usage.",
  "assumptions_on_column_groups": [
    {{
      "target_column_group": "age",
      "assumptions": [
        {{ "assumption": "Age should be between 18 and 100.", "source": "domain knowledge" }},
        {{ "assumption": "At least 90% of ages fall within 21–70.", "source": "data statistics" }}
      ]
    }}
  ]
}}

### Example 2 (multi-column group)
{{
  "task_description": "Compute lifetime value from billing data.",
  "assumptions_on_column_groups": [
    {{
      "target_column_group": "monthly_charges,total_charges,tenure",
      "assumptions": [
        {{ "assumption": "total_charges >= monthly_charges * tenure.", "source": "domain knowledge" }},
        {{ "assumption": "Correlation(monthly_charges,total_charges) > 0.7.", "source": "data statistics" }}
      ]
    }}
  ]
}}

The table name is `{table_name}`.
The table profile is:
{table_profile}
The example rows are:
{example_rows}

Please generate the JSON object only.
"""
# 1. an eos token is needed at the end to make sure the model stops generating
# 2. single column assumptions should be complete to make sure the fine-tuned model can also generate all single-column assumptions
