from inspect import cleandoc

COLUMN_CORRELATION_DISCOVERY_PROMPT = cleandoc("""You are part of the task-aware data validation system. The system is designed to generate data quality constraints tailored to specific downstream tasks by analyzing both the dataset and the code that processes it. You serve as the *Column Correlation Discovery* component.
Given a dataset and the downstream code, your goal is to identify sets of columns that are correlated and require joint constraints on them.

You should discover correlations using the following categories (i.e., cases where the validity of one column depends on another and thus requires a joint constraint):

Consistency Constraint — Two columns must agree under a deterministic rule, e.g., age + birth_year ≈ current_year.
Order/Range Dependency — One column bounds or orders another, e.g., start_date < end_date; discount <= price.
Functional Dependency — One column functionally determines another, e.g., country_code → country_name.
Conditional Completeness / Exclusivity — Presence or values in one column require or forbid another, e.g., if country='US' then state IS NOT NULL; payment_card_xor_token is exclusive.
Task-Driven Dependency — Downstream code or model logic couples columns, e.g., churn_label uses last_login_date with account_status; features used jointly in conditions.
Temporal Consistency — Time fields must follow domain timelines, e.g., signup_date ≤ activation_date ≤ cancel_date.
Others - If you identify other types of correlations not covered above, please use "Others" as the correlation_type.

Previous work has identified the following columns as accessed columns:
{columns_to_consider}

The dataset is a table with the following columns:
{considered_columns_desc}

The user writes the code snippet below:
{code_script}

The above code snippet is used for the following downstream task:
{downstream_task_description}

Your response should be a JSON array with entries in this format:
```json
[
  {{
    "correlated_columns": ["column_name_1", "column_name_2"],
    "correlation_type": "Functional dependencies"
  }},
  {{
    "correlated_columns": ["column_name_3", "column_name_4", "column_name_5"],
    "correlation_type": "Range/order constraints"
  }}
]
```
Remember to quote your answers in ````json``` format, and ensure the JSON is valid and well-structured. Do not include any additional text outside the JSON response. Please discover as complete correlations as possible.

Your answer is:
""")
