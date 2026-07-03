from typing import List, Tuple, Dict, Any

from prismadv.data_models import SourceLocation


class CodeContainer(str):
    def __new__(cls, code: str):
        if not isinstance(code, str):
            raise TypeError("Code must be a string.")
        return super().__new__(cls, code)

    def with_line_numbers(self) -> str:
        lines = self.strip().split('\n')
        if len(lines) > 10000:
            raise ValueError("Code snippet has more than 10000 lines.")
        return "\n".join(f"{i:04}: {line}" for i, line in enumerate(lines, start=1))

    def with_pygments_highlighting(self) -> str:
        """
        Adds Pygments highlighting to the code snippet.
        Args:
            code_snippet (str): The code snippet to add Pygments highlighting to.
        Returns:
            str: The code snippet with Pygments highlighting added.
        """
        from pygments import highlight
        from pygments.lexers import guess_lexer
        from pygments.formatters import TerminalFormatter
        from pygments.util import ClassNotFound
        try:
            lexer = guess_lexer(self)
        except ClassNotFound:
            raise ValueError("Could not guess lexer for the code snippet.")
        formatter = TerminalFormatter(linenos=True)
        highlighted_code = highlight(self, lexer, formatter)
        return highlighted_code

    def add_highlighted_line_numbers(self, source_locations: List[SourceLocation]) -> str:
        lines = self.strip().split('\n')
        if len(lines) > 10000:
            raise ValueError("Code snippet has more than 10000 lines.")
        highlights = set()
        for src in source_locations:
            highlights.update(line for line in range(src.start_line, src.end_line + 1))

        return "\n".join(
            f"-**-> {i:04}: {line}" if i in highlights else f"      {i:04}: {line}"
            for i, line in enumerate(lines, start=1)
        )

    def focused_code(self, source_locations: List[SourceLocation]) -> str:
        # return code snippet with only the lines in source_locations. with ... in the middle
        lines = self.strip().split('\n')
        if len(lines) > 10000:
            raise ValueError("Code snippet has more than 10000 lines.")
        highlights = set()
        for src in source_locations:
            highlights.update(line for line in range(src.start_line, src.end_line + 1))
        focused_lines = []
        if not any(i == 1 for i in highlights):
            focused_lines.append("...")
        for i, line in enumerate(lines, start=1):
            if i in highlights:
                focused_lines.append(f"{i:04}: {line}")
            elif focused_lines and focused_lines[-1].endswith("..."):
                continue
            elif focused_lines:
                focused_lines.append("...")
        return "\n".join(focused_lines) if focused_lines else ""

    def extract_assertions(self) -> Tuple['CodeContainer', List[Dict[str, Any]]]:
        """
        Returns:
            Tuple[str, List[Dict[str, Any]]]:
                - code_without_assertions: the code with all ASSERT blocks removed.
                  A single blank line immediately before the block and immediately after the block is also removed if present.
                - assertions: list of dicts, each with:
                      {
                          "code": "<contents of the assertion block (without markers)>",
                          "start_line": <1-based line number of the first line inside the block in the ORIGINAL code>,
                          "insert_before_line": 1-based line number of the first line after the block in the **stripped code** (code without assertions, skipping at most one blank line), or None if the block was at EOF,
                          "indent": the exact leading whitespace of the `# ASSERTION_START` line.
                      }
        """
        lines = self.strip().split('\n')
        if len(lines) > 10000:
            raise ValueError("Code snippet has more than 10000 lines.")

        out: List[str] = []
        assertions: List[Dict[str, Any]] = []
        i = 0
        n = len(lines)

        while i < n:
            stripped = lines[i].strip()
            if stripped == "# ASSERTION_START":
                # Capture indent of the # ASSERTION_START line
                line_before_strip = lines[i]
                indent = line_before_strip[:len(line_before_strip) - len(line_before_strip.lstrip())]

                # Remove a single blank line immediately before the block, if present
                if out and out[-1].strip() == "":
                    out.pop()

                # Record the original start line of the block's first content line (1-based)
                block_first_content_line_num = i + 2 if (i + 1) < n else i + 1

                # Collect assertion block lines (exclude markers)
                block_lines: List[str] = []
                i += 1
                while i < n and lines[i].strip() != "# ASSERTION_END":
                    block_lines.append(lines[i])
                    i += 1

                # Skip the "# ASSERTION_END" itself
                if i < n and lines[i].strip() == "# ASSERTION_END":
                    i += 1

                # Compute anchor for reinsertion
                after_idx = i  # current pointer is first line after the block
                if after_idx < n and lines[after_idx].strip() == "":
                    after_idx += 1
                # Anchor to the next emitted line position in the stripped output
                insert_before_line = (len(out) + 1) if after_idx < n else None

                # Remove a single blank line immediately after the block, if present
                if i < n and lines[i].strip() == "":
                    i += 1

                assertions.append({
                    "code": "\n".join(block_lines),
                    "start_line": block_first_content_line_num,
                    "insert_before_line": insert_before_line,
                    "indent": indent,
                })
                continue
            else:
                out.append(lines[i])
                i += 1

        code_without_assertions = "\n".join(out) + "\n"
        assertions_sorted = sorted(
            assertions,
            key=lambda x: (x.get("insert_before_line") is None, x.get("insert_before_line") or 0)
        )
        return CodeContainer(code_without_assertions), assertions_sorted

    def insert_assertions(self, assertions: List[Dict[str, Any]], with_index: bool = False) -> 'CodeContainer':
        lines = self.strip().split('\n')
        if len(lines) > 10000:
            raise ValueError("Code snippet has more than 10000 lines.")

        assertions_sorted = sorted(
            assertions, key=lambda x: (
                x.get("insert_before_line") is None, x.get("insert_before_line") or 0
            )
        )
        if assertions_sorted != assertions:
            raise AssertionError("Assertions are not pre-sorted to the required order.")

        out: List[str] = []
        i = 0
        n = len(lines)
        a_idx = 0

        def insert_block(block_code: str, indent: str, idx: int):
            # Remove a single blank line immediately before the block, if present
            if out and out[-1].strip() == "":
                out.pop()
            out.append(f"{indent}# ASSERTION_START")
            if with_index:
                out.append(f"{indent}# Assertion {idx}")
            if block_code:
                out.extend(block_code.split('\n'))
            out.append(f"{indent}# ASSERTION_END")

        while i < n:
            while a_idx < len(assertions_sorted) and assertions_sorted[a_idx].get("insert_before_line") == (i + 1):
                insert_block(assertions_sorted[a_idx]["code"], assertions_sorted[a_idx].get("indent", ""), a_idx)
                a_idx += 1
            out.append(lines[i])
            i += 1

        while a_idx < len(assertions_sorted):
            insert_block(assertions_sorted[a_idx]["code"], assertions_sorted[a_idx].get("indent", ""), a_idx)
            a_idx += 1

        # Always end with a newline for stable diffs and test parity
        code_with_assertions = CodeContainer("\n".join(out) + "\n")
        return code_with_assertions

    def without_blank_lines(self) -> 'CodeContainer':
        lines = self.strip().split('\n')
        if len(lines) > 10000:
            raise ValueError("Code snippet has more than 10000 lines.")
        non_blank_lines = [line for line in lines if line.strip() != '']
        return CodeContainer("\n".join(non_blank_lines) + "\n")
