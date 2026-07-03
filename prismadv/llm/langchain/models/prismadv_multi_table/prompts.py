from inspect import cleandoc


TABLE_COLUMN_ACCESS_DETECTION_PROMPT = cleandoc("""
You are part of a task-aware data validation system for real ETL pipelines.
The pipeline reads multiple input tables and is implemented across multiple code files.

Your task is to identify which raw input table columns are accessed or required by the provided code.
Return only columns from the declared input tables. For derived columns, map them back to the raw input table columns used to derive them.

Input table profiles:
{tables_desc}

The user writes the multi-file code below:
{code_context}

Downstream task description:
{downstream_task_description}

Return only valid JSON in this shape:
```json
{{
  "columns": [
    {{"table": "table_name", "column": "column_name"}}
  ]
}}
```
""")


TABLE_COLUMN_DATA_FLOW_PROMPT = cleandoc("""
You are an expert in data-flow analysis for multi-file ETL code.
Given a target input table column, list every code line that reads, transforms, filters, joins, aggregates, writes, or passes along data derived from that target column.

The target table is: {target_table}
The target column is: {target_column}
The script reads these tables: {script_reads}

The multi-file code is below. Each file has independent 1-based line numbers.
{code_context}

Return only valid JSON in this shape:
```json
{{
  "sources": [
    {{"file": "relative/path.py", "start_line": 1, "end_line": 3}}
  ]
}}
```
If there are no relevant lines, return {{"sources": []}}.
Use plain decimal integers without leading zeros for line numbers (e.g., 22 not 0022).
""")


TABLE_COLUMN_ASSUMPTION_PROMPT = cleandoc("""
You are part of a task-aware data validation system for real ETL pipelines.
Your task is to infer assumptions about one raw input table column based on the table profile and focused multi-file code.

The generated assumptions must be table-local: they should be constraints that can be checked on the target table alone. Do not generate cross-table constraints or foreign-key constraints.

Target table: {target_table}
Target column: {target_column}

Target table profile:
{target_table_desc}

Target column profile:
{target_column_desc}

Focused multi-file code:
{focused_code}

Downstream task description:
{downstream_task_description}

Return only valid JSON in this shape:
```json
{{
  "assumptions": [
    {{
      "text": "The target table column should ...",
      "sources": [
        {{"file": "relative/path.py", "start_line": 1, "end_line": 3}}
      ]
    }}
  ]
}}
```
Use integers without leading zeros for line numbers.
""")


TABLE_COLUMN_DIRECT_CODE_PROMPT = cleandoc("""
You are part of a task-aware data validation system for real ETL pipelines.
Directly produce executable PyDeequ constraint code for one raw input table column, without an intermediate assumption stage. Base your code on the table profile, focused multi-file code, and downstream task description.

The generated code will run against only the target table dataframe, not against a joined table set. Do not generate cross-table constraints or foreign-key checks.

PyDeequ row-level functions:
{row_level_functions}

PyDeequ aggregate-level functions:
{aggregate_level_functions}

Target table: {target_table}
Target column: {target_column}

Target table profile:
{target_table_desc}

Target column profile:
{target_column_desc}

Focused multi-file code:
{focused_code}

Downstream task description:
{downstream_task_description}

Levels: use "error" only when violation would crash or seriously degrade the downstream task; otherwise use "warning".

Return only valid JSON in this shape:
```json
{{
  "constraint_code": [
    {{
      "suggestion": ".isComplete(\\"column_name\\")",
      "level": "error"
    }}
  ]
}}
```
""")


TABLE_COLUMN_GROUP_DIRECT_CODE_PROMPT = cleandoc("""
You are part of a task-aware data validation system for real ETL pipelines.
Directly produce executable PyDeequ constraint code for a same-table column group, without an intermediate assumption stage. Base your code on the table profile, focused multi-file code, and downstream task description.

The generated code will run against only the target table dataframe, not against a joined table set. Do not generate cross-table constraints or foreign-key checks. All referenced columns MUST belong to the target table.

PyDeequ functions that can be used with multiple columns of a single table:
{multi_column_functions}

PyDeequ provides a `satisfies` function that accepts a SQL boolean expression for row-level joint validation. Use it whenever a same-table multi-column constraint is best expressed as a per-row predicate.

Target table: {target_table}
Target columns: {target_columns}

Target columns profile:
{target_columns_desc}

Focused multi-file code:
{focused_code}

Downstream task description:
{downstream_task_description}

Levels: use "error" only when violation would crash or seriously degrade the downstream task; otherwise use "warning".

Return only valid JSON in this shape:
```json
{{
  "constraint_code": [
    {{
      "suggestion": ".satisfies(\\"`col_a` < `col_b`\\", \\"col_a_lt_col_b\\", lambda x: x == 1)",
      "level": "warning"
    }}
  ]
}}
```
""")


TABLE_COLUMN_GROUP_DISCOVERY_PROMPT = cleandoc("""
You are part of a task-aware data validation system for real ETL pipelines.
You serve as the *Same-Table Column Group Discovery* component.

Given a list of accessed columns from raw input tables and the downstream multi-file code, identify groups of columns within the SAME table whose validity is coupled and that therefore require a joint table-local constraint.

Use these categories (any case where the validity of one column depends on another and a joint constraint is meaningful):
- Consistency Constraint  - two columns must agree under a deterministic rule (e.g., age + birth_year approx current_year).
- Order / Range Dependency  - one column bounds or orders another (e.g., start_date < end_date).
- Functional Dependency  - one column functionally determines another within the same table (e.g., code -> description).
- Conditional Completeness / Exclusivity  - presence or values in one column require or forbid another.
- Task-Driven Dependency  - downstream code or model logic couples columns explicitly.
- Temporal Consistency  - time fields must follow a domain timeline.
- Others  - any other same-table coupling.

Hard constraints on your output:
- Every column in a group MUST belong to the same `table`.
- DO NOT propose cross-table groups, foreign-key checks, or referential-integrity constraints.
- Drop singleton groups; only return groups with 2 or more columns.
- Only use tables and columns that appear in the accessed-columns list.

Accessed columns (table-qualified):
{accessed_columns}

Input table profiles:
{tables_desc}

The user writes the multi-file code below:
{code_context}

Downstream task description:
{downstream_task_description}

Return only valid JSON in this shape:
```json
{{
  "groups": [
    {{
      "table": "table_name",
      "correlated_columns": ["column_name_1", "column_name_2"],
      "correlation_type": "Functional dependency"
    }}
  ]
}}
```
If you cannot find any same-table coupled group, return {{"groups": []}}.
""")


TABLE_COLUMN_GROUP_DATA_FLOW_PROMPT = cleandoc("""
You are an expert in data-flow analysis for multi-file ETL code.
Given a target same-table set of columns, list every code line that jointly reads, transforms, filters, joins, aggregates, writes, or passes along data derived from any of those target columns.

The target table is: {target_table}
The target columns are: {target_columns}
The script reads these tables: {script_reads}

Focus on regions where the listed target columns interact (used together, compared, joined, or combined). Solo references to a single column without coupling can be skipped.

The multi-file code is below. Each file has independent 1-based line numbers.
{code_context}

Return only valid JSON in this shape:
```json
{{
  "sources": [
    {{"file": "relative/path.py", "start_line": 1, "end_line": 3}}
  ]
}}
```
If there are no relevant lines, return {{"sources": []}}.
Use plain decimal integers without leading zeros for line numbers (e.g., 22 not 0022).
""")


TABLE_COLUMN_GROUP_ASSUMPTION_PROMPT = cleandoc("""
You are part of a task-aware data validation system for real ETL pipelines.
You serve as the *Same-Table Column Group Assumption Generation* component.
When people write code, they often make joint assumptions about multiple columns from the same table  - relationships such as ordering, conditional presence, or functional dependencies.

Your goal is to infer such joint assumptions for the target column group, based on the table profile and focused multi-file code. The assumptions must be table-local: checkable on the target table alone, no cross-table or foreign-key statements.

Target table: {target_table}
Target columns: {target_columns}

Target columns profile:
{target_columns_desc}

Focused multi-file code:
{focused_code}

Downstream task description:
{downstream_task_description}

Return only valid JSON in this shape:
```json
{{
  "assumptions": [
    {{
      "text": "Within the target table, columns X and Y should satisfy ...",
      "sources": [
        {{"file": "relative/path.py", "start_line": 1, "end_line": 3}}
      ]
    }}
  ]
}}
```
Use integers without leading zeros for line numbers.
""")


TABLE_COLUMN_GROUP_CODE_PROMPT = cleandoc("""
You are part of a task-aware data validation system for real ETL pipelines.
Translate same-table joint assumptions into executable PyDeequ constraint code.

The generated code will run against only the target table dataframe, not against a joined table set. Do not generate cross-table constraints or foreign-key checks. All referenced columns MUST belong to the target table.

PyDeequ functions that can be used with multiple columns of a single table:
{multi_column_functions}

PyDeequ provides a `satisfies` function that accepts a SQL boolean expression for row-level joint validation. Use it whenever a same-table multi-column assumption is best expressed as a per-row predicate (the expression goes after WHERE in SQL, and a row is valid when it evaluates to true).

A single piece of generated code may be linked to multiple assumptions. If an assumption cannot be translated into a same-table check, skip it.

Levels: use "error" only when violation would crash or seriously degrade the downstream task; otherwise use "warning".

Target table: {target_table}
Target columns: {target_columns}

Target columns profile:
{target_columns_desc}

Multi-file code:
{code_context}

Downstream task description:
{downstream_task_description}

Assumptions:
{assumptions}

Return only valid JSON in this shape:
```json
{{
  "constraint_code": [
    {{
      "suggestion": ".satisfies(\\"`col_a` < `col_b`\\", \\"col_a_lt_col_b\\", lambda x: x == 1)",
      "level": "warning",
      "linked assumptions": [0]
    }}
  ]
}}
```
The linked assumption indices are zero-based.
""")


TABLE_COLUMN_CODE_PROMPT = cleandoc("""
You are part of a task-aware data validation system for real ETL pipelines.
Translate table-local assumptions into executable PyDeequ constraint code.

The generated code will run against only the target table dataframe, not against a joined table set. Do not generate cross-table constraints or foreign-key checks.

PyDeequ row-level functions:
{row_level_functions}

PyDeequ aggregate-level functions:
{aggregate_level_functions}

Target table: {target_table}
Target column: {target_column}

Target table profile:
{target_table_desc}

Target column profile:
{target_column_desc}

Multi-file code:
{code_context}

Downstream task description:
{downstream_task_description}

Assumptions:
{assumptions}

Return only valid JSON in this shape:
```json
{{
  "constraint_code": [
    {{
      "suggestion": ".isComplete(\\"column_name\\")",
      "level": "error",
      "linked assumptions": [0]
    }}
  ]
}}
```
The linked assumption indices are zero-based.
""")
