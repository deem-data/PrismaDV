BAD_ASSERTIONS_REMOVING_PROMPT = """
Given a code with assertions. Please rewrite the code to remove the assertion that labeled as bad by remove the assertion it self and the code depends on the assertion.

Bad assertion will be labeled in the following block.
```
# START_OF_BAD_ASSERTION
<assertion_body>
# END_OF_BAD_ASSERTION
```

A regular assertion will be labeled in the following block.
```
# START_OF_ASSERTION
<assertion_body>
# END_OF_ASSERTION
```

When you removing the bad assertions, please make sure to keep the regular assertion and the code depends on them unchanged.

The code with assertions is:
{code_with_assertions}
"""
