import asyncio
import logging

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from prismadv.code_inspector.llm_code_inspector.multiple.prompt import (
    CodeDataFlowInspectorPrompt,
)
from prismadv.llm_backend.entry import get_langchain_model


def _is_source_list(value) -> bool:
    return (isinstance(value, list) and len(value) > 0
            and all(isinstance(v, dict) and "start_line" in v and "end_line" in v for v in value))


def _require_sources(output):
    """The parsed reply with its line ranges under `sources`, or a ValueError the retry loop can act on.
    """
    if isinstance(output, dict) and "sources" not in output:
        aliased = [k for k, v in output.items() if _is_source_list(v)]
        if len(aliased) == 1:
            print(f"\tdata-flow reply used key '{aliased[0]}' for 'sources'; accepted")
            output = {**output, "sources": output[aliased[0]]}
    if not isinstance(output, dict) or "sources" not in output:
        keys = sorted(output) if isinstance(output, dict) else type(output).__name__
        raise ValueError(f"data-flow reply has no 'sources' key (got {keys})")
    return output


class ColumnDataFlowInspector():
    def __init__(self, model_name: str = None,
                 logger: logging.Logger = None):
        if model_name is None:
            raise ValueError("Model name is required.")
        else:
            self.model = self._get_langchain_model(model_name)
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
        self.single_chain = self._build_single_chain()

    @staticmethod
    def _get_langchain_model(model_name: str):
        return get_langchain_model(model_name)

    @staticmethod
    def _build_prompt() -> ChatPromptTemplate:
        return ChatPromptTemplate(
            [
                ("human", CodeDataFlowInspectorPrompt)
            ]
        )

    def _build_single_chain(self):
        prompt = self._build_prompt()
        parser = JsonOutputParser()
        single_chain = prompt | self.model | parser
        return single_chain

    def show_prompts(self, input_variables: dict):
        """
        Displays the prompts used in the model.
        Args:
            input_variables (dict): The input variables for the prompt.
        Returns:
            str: The prompt string.
        """
        prompt = self._build_prompt()
        return prompt.invoke(
            input={
                "code_script": input_variables["code_script"],
                "target_columns": input_variables["target_columns"],
                "sink_variable": input_variables["sink_variable"]
            }
        )

    def invoke(self, input_variables: dict):
        """
        Invokes the model with the given input variables.
        Args:
            input_variables (dict): The input variables for the model.
        Returns:
            dict: The output from the model.
        """
        input_variables["code_script"] = input_variables["code_script"]
        return _require_sources(self.single_chain.invoke(input_variables))

    def invoke_with_retries(self, input_variables: dict, max_retries: int = 3):
        """
        Invokes the model with retries in case of failure.
        Args:
            input_variables (dict): The input variables for the model.
            max_retries (int): Maximum number of retry attempts.
        Returns:
            dict: The output from the model.
        """
        for attempt in range(max_retries):
            try:
                return self.invoke(input_variables)
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if self.logger:
                    self.logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise e

    async def ainvoke(self, input_variables: dict):
        """
        Async variant of invoke using the runnable's ainvoke.
        Ensures code_script has line numbers before calling.
        """
        input_variables["code_script"] = input_variables["code_script"]
        return _require_sources(await self.single_chain.ainvoke(input_variables))

    async def ainvoke_with_retries(self, input_variables: dict, max_retries: int = 3, backoff_s: float = 0.5):
        """
        Async: Invokes the model with retries and exponential backoff.
        Suitable for `await inspector.ainvoke_with_retries(...)`.
        Rate limit errors don't count against retry attempts.
        """
        attempt = 0
        while attempt < max_retries:
            try:
                return await self.ainvoke(input_variables)
            except Exception as e:
                # Check if this is a rate limit/ResourceExhausted error (common with Gemini API)
                error_str = str(e).lower()
                is_rate_limit = (
                    "resourceexhausted" in error_str or
                    "resource exhausted" in error_str or
                    "quota exceeded" in error_str or
                    "rate limit" in error_str
                )
                
                print(f"Attempt {attempt + 1} failed: {e}")
                if self.logger:
                    self.logger.error(f"Attempt {attempt + 1} failed: {e}")
                
                # Rate limit errors don't count against retry attempts - retry indefinitely
                if is_rate_limit:
                    wait_time = 60.0
                    if self.logger:
                        self.logger.info(f"Rate limit error detected. Waiting {wait_time} seconds before retry (not counting against retry limit)...")
                    print(f"Rate limit error detected. Waiting {wait_time} seconds before retry (not counting against retry limit)...")
                    await asyncio.sleep(wait_time)
                    # Don't increment attempt counter for rate limit errors
                    continue
                
                # For non-rate-limit errors, increment attempt counter
                attempt += 1
                if attempt >= max_retries:
                    raise
                
                await asyncio.sleep(backoff_s * (2 ** (attempt - 1)))
