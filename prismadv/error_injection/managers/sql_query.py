from pathlib import Path

from prismadv.error_injection.abstract_error_injection_manager import AbstractErrorInjectionManager
from prismadv.loader import FileLoader
from prismadv.loader.dataset.sql_query_dataset_loader import SQLQueryDatasetLoader


class GeneralErrorInjectionManager(AbstractErrorInjectionManager):
    def __init__(self,
                 raw_file_path: Path,
                 target_table_name: str,
                 processed_data_dir: Path,
                 sample_size: float
                 ):
        self.post_corruption_new_data = None
        self.corrupts = None
        self.raw_file_path = raw_file_path
        self.target_table_name = target_table_name
        self.processed_data_dir = processed_data_dir

        self.full_data = self.load_data()
        (
            self.previous_data,
            self.new_data
        ) = self._split_dataset(self.full_data, sample_size=sample_size)

    def load_data(self):
        return FileLoader.load_csv(self.raw_file_path / f"{self.target_table_name}.csv")

    def error_injection(self, corrupts):
        self.post_corruption_new_data = self.apply_corrupts(self.new_data, corrupts)
        self.corrupts = corrupts

    def save_data(self, processed_data_label):
        if self.post_corruption_new_data is None:
            raise RuntimeError("call error_injection() before save_data()")
        processed_data_path = self._create_processed_data_path(self.processed_data_dir, processed_data_label)
        SQLQueryDatasetLoader().save_data(
            processed_data_path,
            self.previous_data,
            self.new_data,
            self.post_corruption_new_data
        )
        self.save_error_injection_config(processed_data_path, self.corrupts)
        return processed_data_path

    def save_clean_data(self, processed_data_label):
        """Write the uncorrupted split only, for the clean baseline label."""
        processed_data_path = self._create_processed_data_path(self.processed_data_dir, processed_data_label)
        clean_path = processed_data_path / "files_with_clean_new_data"
        clean_path.mkdir(parents=True, exist_ok=True)
        self.previous_data.to_csv(clean_path / "previous_data.csv", index=False)
        self.new_data.to_csv(clean_path / "new_data.csv", index=False)
        return processed_data_path

    def _split_dataset(self, full_data, sample_size):
        full_data.drop(columns=["id"], inplace=True) if "id" in full_data.columns else None
        full_data_shuffled = full_data.sample(frac=sample_size, random_state=1).reset_index(drop=True)
        full_data_shuffled.reset_index(drop=False, inplace=True)
        full_data_shuffled.rename(columns={"index": "id"}, inplace=True)
        previous_data_size = int(len(full_data_shuffled) * 0.6)
        previous_data = full_data_shuffled[:previous_data_size].reset_index(drop=True)
        new_data = full_data_shuffled[previous_data_size:].reset_index(drop=True)
        return previous_data, new_data
