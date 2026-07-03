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
                    return {}

    async def arun_task(self, task: PrismaDVTasks, input_vars: dict,
                        num_retries: int = 10, backoff_s: float = 0.5) -> Union[list, dict]:
        for attempt in range(num_retries):
            try:
                chain = self._chains[task]
                if hasattr(chain, "ainvoke"):
                    return await chain.ainvoke(input_vars)
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, chain.invoke, input_vars)
            except Exception as e:
                if self._logger:
                    self._logger.error(f"Error during task {task.name} (attempt {attempt + 1}): {e}")
                print(f"Error during task {task.name} (attempt {attempt + 1}): {e}")
                if attempt == num_retries - 1:
                    return {}
                await asyncio.sleep(backoff_s * (2 ** attempt))
