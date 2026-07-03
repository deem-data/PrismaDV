# steps/correlation.py
from typing import Dict, List, Any

import oyaml as yaml

from prismadv.llm.langchain.models.prismadv.runtime import Runtime
from prismadv.llm.tasks import PrismaDVTasks


def discovery(runtime: Runtime,
              column_desc_dict: Dict[str, dict],
              columns_to_consider: List[str],
              vars: dict,
              num_retries: int = 10) -> List[Dict[str, Any]]:
    print("\tDetecting correlated columns...")
    considered_columns_desc_dict = {
        col: column_desc_dict[col] for col in column_desc_dict if col in columns_to_consider
    }
    considered_columns_desc = yaml.dump(considered_columns_desc_dict, default_flow_style=False, sort_keys=False)
    input_vars = {
        "columns_to_consider": ", ".join(columns_to_consider),
        "considered_columns_desc": considered_columns_desc,
        "code_script": vars["code_script"],
        "downstream_task_description": vars["downstream_task_description"],
    }
    return runtime.run_task(PrismaDVTasks.COLUMN_CORRELATION_DISCOVERY, input_vars, num_retries)
