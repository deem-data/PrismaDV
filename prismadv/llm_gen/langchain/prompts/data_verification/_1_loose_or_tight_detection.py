# TODO: detect loose or tight assertions
LOOSE_OR_TIGHT_DETECTION_PROMPT = """
Given Python code that contains assertion blocks delimited by:
# ASSERTION_START
...
# ASSERTION_END

Goal: Identify blocks that violate the tight/loose rule.

Indexing:
- The first non-empty line after # ASSERTION_START must be # Assertion <idx> where <idx> is a 0-based integer.
- Use that <idx> as "bad_assertion_id".

Definitions:
- "loose": The block allows values that cause later code to fail or misbehave without first failing the assertion.
- "tight": The block enforces constraints not required by any code that executes after the block.

Heuristics:
- If no later code relies on the asserted property → "tight".
- If later code assumes a stronger property than asserted → "loose".
- Consider only effects after the block. Helper code inside the block is validation-only.

Balance tightness and looseness. Assertions must reflect exactly what the code needs for proper execution and valid outputs.

Examples:
*Loose (too permissive)*
```
# ASSERTION_START
# Assertion <idx>
# BAD: Allows None values, which propagate silently and cause issues downstream
assert (df['age'].isna() | (df['age'] > 0)).all()
# ASSERTION_END
log_age = np.log(df["age"])
```

*Tight (too strict)*
```
# ASSERTION_START
# Assertion <idx>
# BAD: The code does not depend on correlation, so this is too strict
assert df['MonthlyIncome'].corr(df['SalarySlab'].map({{'Upto 5k': 1, '5k-10k': 2, '10k-15k': 3, '15k+': 4}})) > 0.7
# ASSERTION_END
# Code does not depend on correlation, so this is unnecessary
df['IncomeLevel'] = pd.cut(df['MonthlyIncome'], bins=[0, 5000, 10000, 15000, np.inf],
                           labels=['Upto 5k', '5k-10k', '10k-15k', '15k+'])
```

Decision policy:
- decided_to_keep:
  - true if the block is salvageable with a tighter/looser assertion.
  - false if the block should be removed entirely because it encodes an unused constraint or cannot be fixed without guessing new business logic.
- suggested_new_assertion:
  - Required iff decided_to_keep is true.
  - Must be Python code only (no backticks, no prose).
  - Self-contained inside an assertion block. It must not introduce variables used outside the block and must not change program behavior beyond validation.

Output:
Return strictly valid JSON (no markdown fences):
{{
  "result": [
    {{
      "bad_assertion_id": <int>,
      "type": "loose" | "tight",
      "short_explanation": "<=200 chars>",
      "decided_to_keep": true | false,
      "suggested_new_assertion": "<python code if decided_to_keep=true, else empty string>"
    }}
  ]
}}
If all blocks are acceptable, return ```{{"result": []}}```.

Input code:
{code_with_indexed_assertions}
"""
