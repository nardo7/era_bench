import os
from isaaclab.utils.datasets.hdf5_dataset_file_handler import HDF5DatasetFileHandler


class AppendHDF5DatasetFileHandler(HDF5DatasetFileHandler):
    """HDF5 dataset file handler that appends to an existing dataset file if it exists."""

    def create(self, file_path: str, env_name: str | None = None):
        if not file_path.endswith(".hdf5"):
            file_path += ".hdf5"

        if os.path.exists(file_path):
            self.open(file_path, mode="r+")
        else:
            super().create(file_path, env_name=env_name)  # pyright: ignore[reportArgumentType]

    def get_executed_task_ids_with_pairs(self) -> dict[str, list[int]]:
        """Get the executed task IDs from the dataset file with its executed and stored grid pairs indices."""
        self._raise_if_not_initialized()
        if self._hdf5_data_group is None:
            raise RuntimeError("HDF5 data group is not initialized.")
        executed_task_ids = []
        executed_task_pairs = []
        for demo_id in range(self._demo_count):
            demo_group = self._hdf5_data_group[f"demo_{demo_id}"]  # pyright: ignore[reportIndexIssue]
            task_id_data = demo_group["task_info"]["task_id"][()]
            if task_id_data is not None:
                task_id = bytes(task_id_data[-1].tolist()).decode("ascii")
                executed_task_ids.append(task_id)
            task_pair_data = demo_group["task_info"]["task_pair"][()]
            if task_pair_data is not None:
                task_pair = int(task_pair_data[-1].item())
                executed_task_pairs.append(task_pair)

        executed_task_dict: dict[str, list[int]] = {}
        for task_id, task_pair in zip(executed_task_ids, executed_task_pairs):
            if task_id not in executed_task_dict:
                executed_task_dict[task_id] = []
            executed_task_dict[task_id].append(task_pair)
        return executed_task_dict
