# steps/column_access.py
from typing import List

from prismadv.llm.langchain.models.prismadv.runtime import Runtime
from workflow_prismadv.eid_bench_ablation_studies.pipelines.prismadv_wo_assumption_generation.tasks import PrismaDVTasks


def basic_column_access_detection(runtime: Runtime, vars: dict, num_retries: int = 3) -> List[str]:
    # Honor pre-supplied columns
    if "columns_to_consider" in vars and vars["columns_to_consider"]:
        print("\tUsing provided columns to consider... No need to determine accessed columns.")
        cols = vars["columns_to_consider"]
        return cols if isinstance(cols, list) else [cols]

    print("\tDetermining accessed columns...")
    out = runtime.run_task(PrismaDVTasks.COLUMN_ACCESS_DETECTION, vars, num_retries)
    return out
