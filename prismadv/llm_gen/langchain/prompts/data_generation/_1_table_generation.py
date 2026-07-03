TABLE_GENERATION_PROMPT = """You are step 1 in a pipeline that produces (data, data_assumptions, code) triplets. Create a synthetic dataset table in diverse realistic domains. 

### Requirements
- realistic domain (e.g., e-commerce, healthcare, finance, education, logistics). You will be provided with a list of already-generated domains to avoid repetition.
- The dataset would then be used to generate assumptions and code for various downstream tasks, including classification, regression, business intelligence, data engineering with SQL, and website generation.
- 6–12 fields across types: string, integer, decimal, boolean, timestamp, enum.
- Meaningful `table` name (e.g., `orders`, `patients`, `transactions`).
- Provide 3–5 realistic example rows.

### Output format
Return strictly valid JSON (no markdown fences, no commentary). Example:
```
{{
  "table": "name.csv",
  "domain": "<brief domain description>",
  "description": "<short description as a single string>",
  "profile": "<profile string, see below for example>",
  "example_rows": [
    {{"id": 1, "status": "paid", "...": "..."}},
    {{"id": 2, "status": "pending", "...": "..."}}
  ]
}}
```
The `profile` field must be a string. Here is an example profile:
{table_profile_example}

Now generate a new table in this format. Choose a domain not in this list and return the JSON object only:
{already_generated_domains}
"""
# 1. only statistics are needed to generate. the data itself is not needed as we don't need to run the code on the data.
