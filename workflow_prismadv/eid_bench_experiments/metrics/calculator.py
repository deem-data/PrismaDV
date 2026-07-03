import pandas as pd
from sklearn.metrics import roc_auc_score

from prismadv.loader import FileLoader
from workflow.e2e_evaluation.metrics.abstract_calculator import AbstractMetricsCalculation


class MetricsCalculation(AbstractMetricsCalculation):
    def __init__(self):
        super().__init__()

    def calculate(self, sub_task, script_output_dir):
        if sub_task == "classification":
            result = self.calculate_classification_metrics(
                script_output_dir)
        elif sub_task == "regression":
            result = self.calculate_regression_metrics(
                script_output_dir)
        elif sub_task == "bi" or sub_task == "dev" or sub_task == "feature_engineering":
            result = self.calculate_sql_metrics(script_output_dir)
        elif sub_task == "info":
            result = self.calculate_webpage_metrics(
                script_output_dir)
        elif sub_task == "general_task":
            result = self.calculate_general_task(
                script_output_dir)
        else:
            raise ValueError(f"sub_task {sub_task} is not supported")
        return result

    def calculate_classification_metrics(self, script_output_dir):
        corrupted_new_data_path = script_output_dir / "results_on_corrupted_new_data"
        clean_new_data_path = script_output_dir / "results_on_clean_new_data"
        ground_truth_csv = FileLoader.load_csv(
            script_output_dir.parent.parent / "files_with_clean_new_data" / "ground_truth.csv")
        submission_on_corrupted_new_data = self._load_output_file_or_error(corrupted_new_data_path / "submission.csv")
        submission_on_clean_new_data = self._load_output_file_or_error(clean_new_data_path / "submission.csv")
        if isinstance(submission_on_corrupted_new_data, str) and submission_on_corrupted_new_data == "error":
            result_on_corrupted_new_data = "error"
        else:
            result_on_corrupted_new_data = self._calculate_auc(submission_on_corrupted_new_data, ground_truth_csv)

        if isinstance(submission_on_clean_new_data, str) and submission_on_clean_new_data == "error":
            result_on_clean_new_data = "error"
        else:
            result_on_clean_new_data = self._calculate_auc(submission_on_clean_new_data, ground_truth_csv)

        clean_data_is_safe = result_on_clean_new_data != "error" and result_on_clean_new_data >= 0.6
        if result_on_corrupted_new_data != "error" and result_on_clean_new_data != "error":
            corrupted_data_performance_drop = (
                                                      result_on_clean_new_data - result_on_corrupted_new_data) / result_on_clean_new_data
            # auc should be not worse than 5% of the clean data
            if corrupted_data_performance_drop <= 0.05:
                corrupted_data_is_safe = True
            else:
                corrupted_data_is_safe = False
        else:
            corrupted_data_is_safe = False

        return {
            "result_on_corrupted_new_data": result_on_corrupted_new_data,
            "result_on_clean_new_data": result_on_clean_new_data,
            "clean_data_is_safe": clean_data_is_safe,
            "corrupted_data_is_safe": corrupted_data_is_safe
        }

    def calculate_regression_metrics(self, script_output_dir):
        corrupted_new_data_path = script_output_dir / "results_on_corrupted_new_data"
        clean_new_data_path = script_output_dir / "results_on_clean_new_data"
        ground_truth_csv = FileLoader.load_csv(
            script_output_dir.parent.parent / "files_with_clean_new_data" / "ground_truth.csv")
        submission_on_corrupted_new_data = self._load_output_file_or_error(corrupted_new_data_path / "submission.csv")
        submission_on_clean_new_data = self._load_output_file_or_error(clean_new_data_path / "submission.csv")
        if isinstance(submission_on_corrupted_new_data, str) and submission_on_corrupted_new_data == "error":
            result_on_corrupted_new_data = "error"
        else:
            result_on_corrupted_new_data = self._calculate_mse(submission_on_corrupted_new_data, ground_truth_csv)

        if isinstance(submission_on_clean_new_data, str) and submission_on_clean_new_data == "error":
            result_on_clean_new_data = "error"
        else:
            result_on_clean_new_data = self._calculate_mse(submission_on_clean_new_data, ground_truth_csv)

        clean_data_is_safe = (result_on_clean_new_data != "error")
        if result_on_corrupted_new_data != "error" and result_on_clean_new_data != "error":
            corrupted_data_performance_increase = (
                                                          result_on_corrupted_new_data - result_on_clean_new_data) / result_on_clean_new_data
            # mse should be not worse than 5% of the clean data
            if corrupted_data_performance_increase <= 0.05:
                corrupted_data_is_safe = True
            else:
                corrupted_data_is_safe = False
        else:
            corrupted_data_is_safe = False

        return {
            "result_on_corrupted_new_data": result_on_corrupted_new_data,
            "result_on_clean_new_data": result_on_clean_new_data,
            "clean_data_is_safe": clean_data_is_safe,
            "corrupted_data_is_safe": corrupted_data_is_safe
        }

    def calculate_sql_metrics(self, script_output_dir):
        # TODO: make it more representative
        corrupted_new_data_path = script_output_dir / "results_on_corrupted_new_data"
        clean_new_data_path = script_output_dir / "results_on_clean_new_data"
        output_on_corrupted_new_data = self._load_output_file_or_error(corrupted_new_data_path / "output.csv")
        output_on_clean_new_data = self._load_output_file_or_error(clean_new_data_path / "output.csv")
        if isinstance(output_on_corrupted_new_data, str) and output_on_corrupted_new_data == "error":
            result_on_corrupted_new_data = "error"
        else:
            result_on_corrupted_new_data = "success"

        if isinstance(output_on_clean_new_data, str) and output_on_clean_new_data == "error":
            result_on_clean_new_data = "error"
        else:
            result_on_clean_new_data = "success"

        clean_data_is_safe = result_on_clean_new_data == "success"
        corrupted_data_is_safe = result_on_corrupted_new_data == "success"

        return {
            "result_on_corrupted_new_data": result_on_corrupted_new_data,
            "result_on_clean_new_data": result_on_clean_new_data,
            "clean_data_is_safe": clean_data_is_safe,
            "corrupted_data_is_safe": corrupted_data_is_safe
        }

    def calculate_webpage_metrics(self, script_output_dir):
        # TODO: make it more representative
        corrupted_new_data_path = script_output_dir / "results_on_corrupted_new_data"
        clean_new_data_path = script_output_dir / "results_on_clean_new_data"
        file_suffix_set_on_corrupted_new_data = set(
            [file.suffix for file in corrupted_new_data_path.iterdir() if file.is_file()])
        file_suffix_set_on_clean_new_data = set(
            [file.suffix for file in clean_new_data_path.iterdir() if file.is_file()])
        if ".html" in file_suffix_set_on_corrupted_new_data:
            result_on_corrupted_new_data = "success"
        else:
            result_on_corrupted_new_data = "error"

        if ".html" in file_suffix_set_on_clean_new_data:
            result_on_clean_new_data = "success"
        else:
            result_on_clean_new_data = "error"

        clean_data_is_safe = result_on_clean_new_data == "success"
        corrupted_data_is_safe = result_on_corrupted_new_data == "success"

        return {
            "result_on_corrupted_new_data": result_on_corrupted_new_data,
            "result_on_clean_new_data": result_on_clean_new_data,
            "clean_data_is_safe": clean_data_is_safe,
            "corrupted_data_is_safe": corrupted_data_is_safe,
        }

    def calculate_general_task(self, script_output_dir):
        corrupted_new_data_path = script_output_dir / "results_on_corrupted_new_data"
        clean_new_data_path = script_output_dir / "results_on_clean_new_data"
        if (corrupted_new_data_path / "error.txt").exists():
            result_on_corrupted_new_data = "error"
        else:
            result_on_corrupted_new_data = "success"
        if (clean_new_data_path / "error").exists():
            result_on_clean_new_data = "error"
        else:
            result_on_clean_new_data = "success"
        clean_data_is_safe = result_on_clean_new_data == "success"
        corrupted_data_is_safe = result_on_corrupted_new_data == "success"
        return {
            "result_on_corrupted_new_data": result_on_corrupted_new_data,
            "result_on_clean_new_data": result_on_clean_new_data,
            "clean_data_is_safe": clean_data_is_safe,
            "corrupted_data_is_safe": corrupted_data_is_safe
        }

    @staticmethod
    def _load_output_file_or_error(file_path):
        if file_path.exists():
            return FileLoader.load_csv(file_path)
        else:
            return "error"

    @staticmethod
    def _calculate_auc(submission_on_corrupted_new_data, ground_truth_csv):
        # Extract unique class labels
        classes = ground_truth_csv.iloc[:, -1].unique()
        str_int_mapping = {cls: i for i, cls in enumerate(classes)}

        # Map classes to integers
        y_true = ground_truth_csv.iloc[:, -1].map(str_int_mapping).copy()
        y_pred = submission_on_corrupted_new_data.iloc[:, -1].map(str_int_mapping).copy()

        # One-hot encode for multi-class
        if len(classes) > 2:
            y_true = pd.get_dummies(y_true).values  # Convert to NumPy array
            y_pred_proba = pd.get_dummies(y_pred).values  # Convert to NumPy array
            auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
        else:
            auc = roc_auc_score(y_true, y_pred)
        auc = float(auc)
        return auc

    @staticmethod
    def _calculate_mse(submission_on_corrupted_new_data, ground_truth_csv):
        y_true = ground_truth_csv.iloc[:, -1]
        y_pred = submission_on_corrupted_new_data.iloc[:, -1]
        mse = float(((y_true - y_pred) ** 2).mean())
        return mse
