import oyaml as yaml

from prismadv.data_models import ConstraintsWithSources
from prismadv.data_models.config import PrismaDVConfig
from prismadv.data_models.constraints_v2 import de_column_group_key
from prismadv.llm.langchain.models.prismadv_post_processing import PrismaLangChainDVPostProcessing


def apply_fixings_on_constraints(
        dataset_name,
        subtask_name,
        processed_data_label,
        post_processed_file):
    constraint_file_wo_post_processing = post_processed_file.with_name(
        post_processed_file.name.replace("post_processed_prismadv", "prismadv")
    )
    assert constraint_file_wo_post_processing.exists()

    output_file = post_processed_file.with_name(
        post_processed_file.name.replace("post_processed_prismadv", "prismadv_wo_code_consolidation")
    )

    with open(f"{constraint_file_wo_post_processing}", "r") as f:
        constraint_file_wo_post_processing_dict = yaml.load(f, Loader=yaml.FullLoader)
    with open(f"{post_processed_file}", "r") as f:
        post_processed_file_dict = yaml.load(f, Loader=yaml.FullLoader)
    prismadv_config = PrismaDVConfig.from_dict(constraint_file_wo_post_processing_dict["prismadv_config"])
    prismadv_post_processing = PrismaLangChainDVPostProcessing.from_config(prismadv_config)
    existing_constraints = ConstraintsWithSources.from_dict(
        {"constraints": constraint_file_wo_post_processing_dict["constraints"]})
    column_data_flow_locations = constraint_file_wo_post_processing_dict["column_data_flow_locations"]

    cost_summary = post_processed_file_dict["cost_summary"]
    if "consolidate_constraints" in cost_summary:
        cost_summary.pop("consolidate_constraints")
    fixing_results_ser = post_processed_file_dict['fixing_results']
    fixing_results = {
        de_column_group_key(k): v for k, v in fixing_results_ser.items()
    }
    constraints_after_fixing = prismadv_post_processing.merge_fixed_code(existing_constraints, fixing_results)
    res_dict = {
        "prismadv_config": prismadv_config.to_dict(),
        "constraints": constraints_after_fixing.to_dict()["constraints"],
        "column_data_flow_locations": column_data_flow_locations,
        "fixing_results": fixing_results_ser,
        "cost_summary": cost_summary,
    }
    with open(output_file, "w") as f:
        yaml.dump(res_dict, f, default_flow_style=False, sort_keys=False)
