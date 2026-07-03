from sifta.proposer.reflective_mutation.base import Signature


class InstructionProposalSignature(Signature):
    prompt_template = """I provided an assistant with the following instructions to perform a task for me:
```
<curr_instructions>
```

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
```
<inputs_outputs_feedback>
```

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.

Provide the new instructions within ``` blocks."""

    input_keys = ["current_instruction_doc", "dataset_with_feedback"]
    output_keys = ["new_instruction"]

    @classmethod
    def prompt_renderer(cls, input_dict: dict[str, str]) -> str:
        def format_samples(samples):
            def render_value(value, level=3):
                # level controls markdown header depth (###, ####, etc.)
                if isinstance(value, dict):
                    s = ""
                    for k, v in value.items():
                        s += f"{'#' * level} {k}\n"
                        s += render_value(v, min(level + 1, 6))
                    if not value:
                        s += "\n"
                    return s
                elif isinstance(value, (list, tuple)):
                    s = ""
                    for i, item in enumerate(value):
                        s += f"{'#' * level} Item {i + 1}\n"
                        s += render_value(item, min(level + 1, 6))
                    if not value:
                        s += "\n"
                    return s
                else:
                    return f"{str(value).strip()}\n\n"

            def convert_sample_to_markdown(sample, examplenum):
                s = f"# Example {examplenum}\n"
                for key, val in sample.items():
                    s += f"## {key}\n"
                    s += render_value(val, level=3)
                return s

            return "\n\n".join(convert_sample_to_markdown(sample, i + 1) for i, sample in enumerate(samples))

        prompt = cls.prompt_template
        prompt = prompt.replace("<curr_instructions>", input_dict["current_instruction_doc"])
        prompt = prompt.replace("<inputs_outputs_feedback>", format_samples(input_dict["dataset_with_feedback"]))
        return prompt

    @classmethod
    def output_extractor(cls, lm_out: str) -> dict[str, str]:
        # Extract ``` blocks
        new_instruction = None
        if lm_out.count("```") >= 2:
            start = lm_out.find("```")
            end = lm_out.rfind("```")
            if start >= end:
                new_instruction = lm_out
            if start == -1 or end == -1:
                new_instruction = lm_out
            else:
                new_instruction = lm_out[start+3:end].strip()
        else:
            lm_out = lm_out.strip()
            if lm_out.startswith("```"):
                lm_out = lm_out[3:]
            if lm_out.endswith("```"):
                lm_out = lm_out[:-3]
            new_instruction = lm_out

        return {"new_instruction": new_instruction}


class GlobalInstructionProposalSignature(Signature):
    """
    A global-view instruction proposer that sees all modules and their feedback,
    then decides which module(s) to update and proposes new instructions.
    """
    prompt_template = """## System Overview: Data Validation Constraint Generator

You are optimizing an AI system that generates **data validation constraints** for DataFrame columns. The system analyzes code scripts to understand how columns are used, then generates executable assertions that can detect data quality issues.

### Pipeline Architecture

The system has three modules that work sequentially:

1. **dataflow_inspector**: Analyzes the code script to identify which lines are relevant to the target column (where it's read, transformed, or used).

2. **assumption_generation**: Based on the relevant code context, generates assumptions about what should be true for valid data in this column.

3. **ir_generation**: Converts assumptions into executable constraint code (PyDeequ constraints). Each constraint is linked back to its source assumption(s).

### Constraint Validation and Filtering Workflow

**IMPORTANT**: The system employs a two-phase approach that naturally filters out invalid or overly strict constraints:

1. **Validation on Observed Partition (Training Phase)**:
   - All generated constraints are first validated on the **observed/clean data partition**
   - This phase checks:
     - **Grammar/Syntax**: Does the PyDeequ constraint code compile and execute without errors?
     - **Validity on Clean Data**: Does the constraint pass on known-good data? (Constraints that fail on clean data are too strict and are discarded)
   - Only constraints that pass both checks are retained for the next phase

2. **Application to New Partitions (Evaluation Phase)**:
   - The filtered, validated constraints are then applied to **new data partitions**
   - **Constraint Suite Evaluation Rule**: All constraints for a column form a validation suite. **If any single constraint fails, the entire data partition is predicted as erroneous.**
     - One constraint fails → Partition predicted as erroneous
     - All constraints pass → Partition predicted as safe
   - This AND logic means:
     - **TN**: One constraint fails on corrupted partition → Correctly detected error
     - **FN**: One constraint fails on clean partition → False alarm (too strict)
     - **FP**: All constraints pass on corrupted partition → Missed error (too loose)
     - **TP**: All constraints pass on clean partition → Correctly accepted

**Key Implication**: Be **comprehensive and thorough** when generating assumptions and constraints. You don't need to worry about:
- Generating too many constraints (invalid ones will be filtered out during validation)
- Being overly cautious about constraint strictness (too-strict constraints will fail on observed data and be discarded)
- Minor syntax errors (non-compilable constraints are automatically filtered)

However, understand that each constraint in the final suite affects partition-level predictions—one overly strict constraint can cause false alarms, while missing important constraints can lead to missed errors.

### Optimization Metric: Fail Precision

The system is evaluated using **fail precision**, which measures constraint quality:

- **True Negative (TN)**: Constraint fails on corrupted/unsafe data → Correct detection (caught error)
- **False Negative (FN)**: Constraint fails on clean/safe data → False alarm (too strict)
- **False Positive (FP)**: Constraint passes on corrupted/unsafe data → Missed violation (too loose)
- **True Positive (TP)**: Constraint passes on clean/safe data → Correct acceptance

**fail_precision = TN / (TN + FN)**

- Score = 1.0: Perfect - all failures are on actual bad data
- Score < 1.0: Some false alarms - constraints incorrectly fail on clean data
- Score = NaN: No failures detected - constraints may be too general

### Common Issues to Address

- **False negatives (FN)**: Constraints are too strict and fail on clean data. Relax overly narrow assumptions to avoid false alarms
- **False positives (FP)**: Constraints are too loose and miss corrupted data. Add more specific checks to catch edge cases
- **Invalid constraints**: Code doesn't compile or fails on training data. Fix syntax/logic.
- **Missing assumptions**: Important data properties not captured. Enhance assumption generation. This is the most common root cause of low precision.
- **Assumption-constraint mismatch**: Constraints don't properly implement assumptions. Improve IR generation.

---

## Current Module Configurations

<all_modules>

## Execution Traces and Feedback

The modules work sequentially on the same data: dataflow_inspector → assumption_generation → ir_generation. The detailed execution traces below show the full pipeline output including all three modules. Placeholder scores are shown for upstream modules, while the final module (ir_generation) contains the complete trace with code context, assumptions, and constraint performance.

<all_feedback>

## Your Task

1. Analyze the feedback across ALL modules holistically
2. Identify root causes: Is the issue in dataflow detection, assumption quality, or constraint generation?
3. Consider module interactions - bad dataflow context leads to bad assumptions leads to bad constraints
4. Decide which module(s) need instruction updates to improve fail precision

Guidelines:
- **Prioritize comprehensiveness**: Generate thorough, complete assumptions covering all relevant data quality aspects. The validation workflow (see above) naturally filters invalid constraints, so be comprehensive rather than overly conservative
- Ensure assumptions are grounded in actual code behavior, not just column names
- Constraints should be specific enough to catch real issues but general enough to pass on valid data
- Include domain-specific patterns you observe in the feedback
- **Capture dataset-specific characteristics**: The optimized instructions will be applied to other scripts operating on the same dataset and used to validate new data partitions. Extract and include dataset-specific patterns, typical value ranges, common data quality issues, and domain conventions observed in the feedback so future runs benefit from these insights
- **Preserve output formats**: Do NOT modify the instruction about JSON output format specifications for any module. Each module has a hardcoded parser expecting specific field names and structures (e.g., "sources", "assumptions", "constraint_code"). Only refine the semantic instructions about what to analyze or generate, never the output schema.
- **Keep PyDeequ grammar in ir_generation**: When updating the ir_generation module, ensure that any new instructions preserve the correct PyDeequ constraint syntax and API usage. The constraints must remain valid PyDeequ code that can be executed by the validation engine
- **Propose incremental refinements**: Build on the current instructions rather than rewriting from scratch. Preserve elements that work well and make targeted adjustments to address specific issues identified in the feedback. Small, focused changes are often more effective than wholesale rewrites
- **Generate comprehensive assumptions**: Encourage generating complete, thorough assumptions that cover diverse data quality aspects. As explained in the "Constraint Validation and Filtering Workflow" section above, all constraints are validated on observed/clean data partitions before being applied to new partitions. This two-phase validation approach means you should be comprehensive rather than conservative—cast a wide net to capture all potentially relevant data quality rules, knowing that invalid or overly strict constraints will be naturally filtered out during the validation phase on observed data
- **Focus on error-level constraints**: Warning level constraints will be ignored during evaluation. Ensure that the instructions emphasize generating error-level constraints to impact fail precision scores.


## Output Format

For each module you want to update, use this exact format:

[MODULE_NAME]
```
new instruction here
```

You can include multiple [MODULE_NAME] blocks if updating multiple modules.
Only include modules that need changes - do not repeat unchanged instructions."""

    input_keys = ["all_modules", "all_feedback"]
    output_keys = ["updated_instructions"]  # dict of module_name -> new_instruction

    @classmethod
    def prompt_renderer(cls, input_dict: dict) -> str:
        def format_module_info(modules_info: dict[str, dict]) -> str:
            """Format all modules with their signatures and current instructions."""
            s = ""
            for name, info in modules_info.items():
                s += f"### Module: `{name}`\n\n"
                s += f"**Signature:**\n"
                s += f"- Inputs: {info.get('inputs', 'N/A')}\n"
                s += f"- Outputs: {info.get('outputs', 'N/A')}\n\n"
                s += f"**Current Instruction:**\n```\n{info.get('instruction', '')}\n```\n\n"
            return s

        def format_all_feedback(feedback_by_module: dict[str, list]) -> str:
            """Format feedback for all modules."""
            def render_value(value, level=4):
                if isinstance(value, dict):
                    s = ""
                    for k, v in value.items():
                        s += f"{'#' * level} {k}\n"
                        s += render_value(v, min(level + 1, 6))
                    if not value:
                        s += "\n"
                    return s
                elif isinstance(value, (list, tuple)):
                    s = ""
                    for i, item in enumerate(value):
                        s += f"{'#' * level} Item {i + 1}\n"
                        s += render_value(item, min(level + 1, 6))
                    if not value:
                        s += "\n"
                    return s
                else:
                    return f"{str(value).strip()}\n\n"

            s = ""
            for module_name, examples in feedback_by_module.items():
                s += f"### Module: `{module_name}`\n\n"
                if not examples:
                    s += "_No feedback examples for this module._\n\n"
                    continue
                for i, example in enumerate(examples):
                    s += f"#### Example {i + 1}\n"
                    for key, val in example.items():
                        s += f"##### {key}\n"
                        s += render_value(val, level=6)
                s += "\n"
            return s

        prompt = cls.prompt_template
        prompt = prompt.replace("<all_modules>", format_module_info(input_dict["all_modules"]))
        prompt = prompt.replace("<all_feedback>", format_all_feedback(input_dict["all_feedback"]))
        return prompt

    @classmethod
    def output_extractor(cls, lm_out: str) -> dict[str, dict[str, str]]:
        """
        Extract module updates from LM output.
        Expected format:
        [MODULE_NAME]
        ```
        instruction content
        ```

        Returns: {"updated_instructions": {module_name: new_instruction, ...}}
        """
        import re

        updated_instructions = {}

        # Try multiple patterns to handle different LM output formats
        # Note: Use [ \t]* for horizontal whitespace only (not \s* which includes newlines)
        patterns = [
            # Pattern 1: [MODULE_NAME] followed by ``` block
            r'\[([^\]]+)\][ \t]*\n```(?:\w*\n)?(.*?)```',
            # Pattern 2: `MODULE_NAME` followed by ``` block (backticks)
            r'`([^`]+)`[ \t]*\n```(?:\w*\n)?(.*?)```',
            # Pattern 3: **MODULE_NAME** followed by ``` block (bold)
            r'\*\*([^*]+)\*\*[ \t]*\n```(?:\w*\n)?(.*?)```',
            # Pattern 4: MODULE_NAME: followed by ``` block
            r'^([a-zA-Z_][a-zA-Z0-9_.]*)[ \t]*:[ \t]*\n```(?:\w*\n)?(.*?)```',
            # Pattern 5: Plain module.name on its own line followed by ``` block (no markers)
            r'(?:^|\n)([a-zA-Z_][a-zA-Z0-9_.]*)[ \t]*\n```(?:\w*\n)?(.*?)```',
        ]

        matches = []
        for pattern in patterns:
            matches = re.findall(pattern, lm_out, re.DOTALL | re.MULTILINE)
            if matches:
                break

        for module_name, instruction in matches:
            module_name = module_name.strip()
            instruction = instruction.strip()
            if module_name and instruction:
                updated_instructions[module_name] = instruction
                # Print first 1000 chars of proposed prompt
                preview = instruction[:500] + "..." if len(instruction) > 500 else instruction
                print(f"[Proposed Prompt] {module_name} ({len(instruction)} chars):\n{preview}\n")

        return {"updated_instructions": updated_instructions}
