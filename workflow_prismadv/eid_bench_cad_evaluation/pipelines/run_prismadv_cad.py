import oyaml as yaml
import pandas as pd

from prismadv.data_models import ConstraintsWithSources
from prismadv.data_models.config import PrismaDVConfig
from prismadv.data_models.constraints_v2 import de_column_group_key
from prismadv.project_manager.manager.base import ProjectManager
from prismadv.utils import get_project_root
from workflow.cad_evaluation.metrics.cad import f1_accessed_columns, recall_correlated_columns


def run_prismadv_cad(
        dataset_name,
        subtask_name,
        constraint_file
):
    script_name = constraint_file.parent.stem
    project_manager = ProjectManager(project_root=get_project_root(), dataset_name=dataset_name)
    with open(f"{constraint_file}", "r") as f:
        raw_constraint_dict = yaml.load(f, Loader=yaml.FullLoader)
        existing_constraints = ConstraintsWithSources.from_dict(
            {"constraints": raw_constraint_dict["constraints"]})
        existing_cost_summary = raw_constraint_dict["cost_summary"]
        column_data_flow_locations_prediction = {de_column_group_key(k): v for k, v in
                                                 raw_constraint_dict["column_data_flow_locations"].items()}
    try:
        prismadv_config = PrismaDVConfig.from_dict(raw_constraint_dict["prismadv_config"])
    except Exception:
        print('Not a valid config')
        return
    annotation_file_path = project_manager.get_annotation_file_path_from_subtask(subtask_name, script_name)
    with open(annotation_file_path, "r") as f:
        annotations = yaml.load(f, Loader=yaml.Loader)['annotations']
    accessed_columns_ground_truth = annotations['accessed_columns']
    correlated_columns_ground_truth = [frozenset(i) for i in annotations['correlated_columns']]
    column_data_flow_locations_ground_truth = annotations['column_data_flow']
    single_column_data_flow_ground_truth = {
        k: v for k, v in column_data_flow_locations_ground_truth.items() if
        type(k) == str or (type(k) == frozenset and len(k) == 1)
    } if column_data_flow_locations_ground_truth is not None else None
    multi_column_data_flow_ground_truth = {
        k: v for k, v in column_data_flow_locations_ground_truth.items() if
        type(k) == frozenset and len(k) > 1
    } if column_data_flow_locations_ground_truth is not None else None

    accessed_columns_prediction = [k for k in existing_constraints.data_map.keys() if type(k) == str]
    correlated_columns_prediction = [k for k in existing_constraints.data_map.keys() if type(k) == frozenset]
    single_column_data_flow_prediction = {
        k: v for k, v in column_data_flow_locations_prediction.items() if
        type(k) == str or (type(k) == frozenset and len(k) == 1)
    }
    multi_column_data_flow_prediction = {
        k: v for k, v in column_data_flow_locations_prediction.items() if
        type(k) == frozenset and len(k) > 1
    }

    accessed_columns_f1 = f1_accessed_columns(accessed_columns_ground_truth, accessed_columns_prediction)
    correlated_columns_recall = recall_correlated_columns(
        correlated_columns_ground_truth, correlated_columns_prediction)
    return pd.DataFrame([{
        "dataset_name": dataset_name,
        "model_name": prismadv_config.llm.model_name,
        "temperature": prismadv_config.llm.temperature,
        "script_name": script_name,
        "len(accessed_columns_prediction)": len(accessed_columns_prediction or []),
        "len(accessed_columns_ground_truth)": len(accessed_columns_ground_truth or []),
        "accessed_columns_f1": accessed_columns_f1,
        "correlated_columns_recall": correlated_columns_recall,
    }])
