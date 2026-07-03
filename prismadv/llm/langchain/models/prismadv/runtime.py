import asyncio
from typing import Union, Dict

from prismadv.llm.tasks import PrismaDVTasks


class Runtime:
    def __init__(self, chains: Dict[PrismaDVTasks, object], logger=None):
        self._chains = chains
        self._logger = logger

    def run_task(self, task: PrismaDVTasks, input_vars: dict,
                 num_retries: int = 10) -> Union[list, dict]:
        for attempt in range(num_retries):
            try:
                return self._chains[task].invoke(input_vars)
            except Exception as e:
                if self._logger:
                    self._logger.error(f"Error during task {task.name} (attempt {attempt + 1}): {e}")
                print(f"Error during task {task.name} (attempt {attempt + 1}): {e}")
                if attempt == num_retries - 1:
                    raise

    async def arun_task(self, task: PrismaDVTasks, input_vars: dict,
                        num_retries: int = 10, backoff_s: float = 0.5) -> Union[list, dict]:
        attempt = 0
        while attempt < num_retries:
            try:
                chain = self._chains[task]
                if hasattr(chain, "ainvoke"):
                    return await chain.ainvoke(input_vars)
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, chain.invoke, input_vars)
            except Exception as e:
                # Check if this is a rate limit/ResourceExhausted error (common with Gemini API)
                error_str = str(e).lower()
                is_rate_limit = (
                    "resourceexhausted" in error_str or
                    "resource exhausted" in error_str or
                    "quota exceeded" in error_str or
                    "rate limit" in error_str
                )
                
                if self._logger:
                    self._logger.error(f"Error during task {task.name} (attempt {attempt + 1}): {e}")
                print(f"Error during task {task.name} (attempt {attempt + 1}): {e}")
                
                # Rate limit errors don't count against retry attempts - retry indefinitely
                if is_rate_limit:
                    wait_time = 60.0
                    if self._logger:
                        self._logger.info(f"Rate limit error detected. Waiting {wait_time} seconds before retry (not counting against retry limit)...")
                    print(f"Rate limit error detected. Waiting {wait_time} seconds before retry (not counting against retry limit)...")
                    await asyncio.sleep(wait_time)
                    # Don't increment attempt counter for rate limit errors
                    continue
                
                # For non-rate-limit errors, increment attempt counter
                attempt += 1
                if attempt >= num_retries:
                    raise
                
                await asyncio.sleep(backoff_s * (2 ** (attempt - 1)))
