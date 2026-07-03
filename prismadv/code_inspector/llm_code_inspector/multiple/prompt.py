from inspect import cleandoc

CodeDataFlowInspectorPrompt = cleandoc("""
You are an expert in data-flow analysis and code inspection. Your task is to analyze the provided data-centric code snippet and identify the data flow paths related to specific target columns.
You will be given a code snippet, target columns, and a sink variable derived from that column. Your goal is to identify all the data flow paths that lead to the sink variable, including any transformations or operations applied to the data.
If a sink variable is supplied, include only the lines that ultimately affect that sink. If no sink is given, include every reference through to the end of the snippet.

You should return the code line numbers of the identified data flow paths in a json format.

After getting the relevant code snippet, it will be used to form the complete data flow between the target column and the sink variable. Thus the data flow must be complete and accurate. Actually, the completeness is even more important than the accuracy, since the data flow is used to form the complete data flow between the target column and the sink variable. This step is more about making the next task more focused.

```json
{{
“sources”: [
    {{ “start_line”: 2, “end_line”: 2 }},
    {{ “start_line”: 5, “end_line”: 7 }}
    ]
}}
```

We will use a JSON parser to parse your output, so please ensure that your output is valid JSON. the line number should be valid 1-based line numbers.
For example, 0073 is not a valid line number, but 73 is.

The user writes the code script below:
```
{code_script}
```
The target columns are: {target_columns}
The sink variable is: {sink_variable}
""")
