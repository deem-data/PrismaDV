from inspect import cleandoc

CodeDataFlowInspectorPrompt = cleandoc(
    """You are an expert in data-flow analysis and code inspection.
Given a code snippet, a column name, and (optionally) a sink variable, list every line that touches the column—reads it, writes to it, filters on it, transforms it, merges it, or passes it along.
If a sink variable is supplied, include only the lines that ultimately affect that sink. If no sink is given, include every reference through to the end of the snippet.

Output (1-based line numbers)
Return only a JSON object whose "sources" field contains the contiguous line-number ranges that reference or manipulate the column:
```json
{{
“sources”: [
    {{ “start_line”: 2, “end_line”: 2 }},
    {{ “start_line”: 5, “end_line”: 7 }}
    ]
}}
```

Rules
1.	Use 1-based line numbers exactly as they appear in the snippet.
2.	Be exhaustive: any line that interacts with the column must appear in at least one range.
3.	Adjacent lines go in the same range; non-adjacent lines go in separate ranges.
4.	The order inside "sources" does not matter.
5.	If {sink_variable} is empty or omitted, consider the entire snippet.
6.	Output only the JSON—no extra text or formatting.

Mini-example

1:  df = pd.read_csv(“employees.csv”)
2:  df[“age”] = df[“age”].fillna(0)
3:  df[“age_group”] = pd.cut(df[“age”], bins=[0,17,64,120], labels=[“child”,“adult”,“senior”])
4:  df[“salary_eur”] = df[“salary_usd”] * 0.9
5:  engineering = df[df[“department”] == “Engineering”]
6:  high_earners = engineering[engineering[“salary_eur”] > 80000]
7:  result = high_earners[[“name”,“age”,“salary_eur”,“age_group”]].copy()
8:  result.reset_index(drop=True, inplace=True)
9:  final_df = result



-> Column: age and Sink: result_df
```json
{{
    "sources": [
        {{"start_line": 1, "end_line": 3}},
        {{"start_line": 7, "end_line": 9}},
    ]
}}
```

We will use a JSON parser to parse your output, so please ensure that your output is valid JSON. the line number should be valid 1-based line numbers.
For example, 0073 is not a valid line number, but 73 is.

The user writes the code script below:
```
{code_script}
```
The Target Column: {target_column}
Sink variable (optional): {sink_variable}
""")
