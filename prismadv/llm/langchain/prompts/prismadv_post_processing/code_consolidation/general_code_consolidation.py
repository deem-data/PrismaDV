from inspect import cleandoc

CODE_CONSOLIDATION_PROMPT = cleandoc("""
You are part of a task-aware data validation system. The system generates data quality constraints tailored to specific downstream tasks by analyzing both the dataset and the code that processes it. You serve as the *Code Consolidation* component.

Previous work has generated constraint code using PyDeequ on the dataset to reflect code requirements. However, there may be redundancy between some PyDeequ constraints. Your task is to remove redundant constraints.

There are two redundancy scenarios:

1. **Function-level equivalence:**  
   `.satisfies("Age is not NULL")` is functionally equivalent to `.isComplete("age")`.  
   In such cases, always keep the `.satisfies()` constraint and remove the `.isComplete()` style one.

2. **Logical subsumption:**  
   If two `.satisfies()` constraints express the same logic (i.e., if constraint A passes, then constraint B must also pass, and if A fails, B must also fail), remove the less strict or less clear constraint. Keep the stricter and clearer one.

Input: a list of constraints, each acting on a single column or a group of columns.  
Output: a JSON object in the following format:

```json
{{
    "results": [
        {{
            "idx_to_remove": <int>,
            "redundant_with": [<int>, ...]
        }},
        ...
    ]
}}

For example:
Given code on column: Age
And the codes are:

0: .isComplete("age")
1: .satisfies("age IS NOT NULL", lambda x: "Age is not NULL")
2: .satisfies("age BETWEEN 0 AND 120", lambda x: "Age within [0,120]")

You answer should be 
```
{{
  "results": [
    {{
      "idx_to_remove": 0,
      "redundant_with": [1]
    }}
  ]
}}
```

if there is no redundent code. return
```json
{{
    "results": []
}}
```

Let's start, the code is on column (group):
{column_names}

The relevant column statstics are:
{related_column_desc}

The codes are:
{valid_code_entries_str}

Please show your response in the format above.
```
""")
# scenarios:
#   1. Multicolumn constraints but not included multi columns, match column_names, if only one column name appears, remove it. No need to invoke llm.
#   2. single/multi-column same constraints in different expression. lead to maintaining problem.
#       1) completeness using both iscomplete and satisfies.
#       2) SQL in different grammar.
