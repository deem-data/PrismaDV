import asyncio
from typing import List, Awaitable, TypeVar

T = TypeVar('T')


async def batched_gather(
    coros: List[Awaitable[T]],
    max_concurrent: int = None
) -> List[T]:
    """
    Execute coroutines in batches with a concurrency limit.
    If max_concurrent is None, executes all coroutines concurrently.
    
    Args:
        coros: List of coroutines to execute
        max_concurrent: Maximum number of concurrent executions (None = unlimited)
    
    Returns:
        List of results in the same order as input coroutines
    """
    if max_concurrent is None or len(coros) <= max_concurrent:
        return await asyncio.gather(*coros)
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def bounded_coro(coro: Awaitable[T]) -> T:
        async with semaphore:
            return await coro
    
    return await asyncio.gather(*[bounded_coro(coro) for coro in coros])
