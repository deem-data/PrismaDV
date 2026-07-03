TABLE_SUMMARIZATION_PROMPT = """You are an expert data analyst. Given a table, you are asked to provide a concise summary of its domain and description.

Here are the information about the table:

filename: {filename}

column_profiles:
{column_profiles}

example rows:
{example_rows}

Please provide a concise summary in the following format:
```
{{
    "domain": "<a short phrase describing the domain of the table, e.g., healthcare, finance, retail, etc.>",
    "description": "<a concise description of the table, its purpose, and the type of data it contains.>"
}}
```
"""
