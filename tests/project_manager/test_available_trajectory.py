from prismadv.project_manager.manager.base import ProjectManager


def test_get_dspy_trajectory():
    project_manager = ProjectManager(dataset_name="students",
                                     downstream_task_type="general")
    subtask_name = "general_task"
    processed_data_label = "1"
    script_name = "general_task_11"
    dspy_trajectories = project_manager.get_dspy_trajectory(
        subtask_name,
        processed_data_label,
        script_name
    )
    assert dspy_trajectories[0].script_path.name == "general_task_11.py"


def test_get_more_dspy_trajectories():
    project_manager = ProjectManager(dataset_name="students",
                                     downstream_task_type="general")
    subtask_name_list = ["general_task"]
    processed_data_label_list = [str(i) for i in range(1, 4)]
    all_trajectories = []
    for subtask_name in subtask_name_list:
        available_script_path = project_manager.get_available_script_path_list_for_subtask(
            subtask_name
        )
        available_script_name_list = [path.stem for path in available_script_path]

        dspy_trajectories = project_manager.get_dspy_trajectories(
            subtask_name,
            processed_data_label_list,
            script_name_list=available_script_name_list
        )
        all_trajectories.extend(dspy_trajectories)
    assert len(all_trajectories) > 1


def test_trajectory_to_dspy_example():
    project_manager = ProjectManager(dataset_name="students",
                                     downstream_task_type="general")
    subtask_name = "general_task"
    processed_data_label = "1"
    script_name = "general_task_11"
    dspy_trajectories = project_manager.get_dspy_trajectory(
        subtask_name,
        processed_data_label,
        script_name
    )
    trajectory_sample = dspy_trajectories[0]
