import json
import zipfile
from dataclasses import dataclass
from typing import List

from data.dataset import Dataset
from environments.shell.dataset.terminal_dataset_reader import TerminalDatasetReader
from tokenization.tokenizer import Tokenizer


@dataclass
class TerminalTaskInfo:
    task_id: int
    task_name: str
    task_prompt: str
    difficulty: str
    terminal_recordings: List[TerminalDatasetReader]


class TerminalTaskDataset(Dataset):
    """
    Wraps a zip file containing a dataset of tasks.
    Structure:
        - train/
        - val/
        - meta.json

    Each folder contains a set of tasks:
        - task_1/
        - task_2/
        - ...

    metadata.json contains metadata about the dataset:
    {
        "name": str, # The name of the dataset"
        "terminal": {
            "width": int, # The width of the terminal
            "height": int # The height of the terminal
        }
    }

    Each task contains a set of files:
        - task.json
        - task.md
        - alternatives/
            - recording_*.bin

    The task.json file contains metadata about the task:
    {
        "name": str, # The name of the task
        "difficulty": str # The difficulty of the task (e.g. "easy", "medium", "hard")
    }

    The task.md file contains the prompt of the task in Markdown format.

    The recording.bin file is a terminal recording the format described in terminal_dataset_reader.py
    TerminalDatasetReader wraps a file-like object containing a terminal recording.
    """

    def __init__(self, dataset_path: str, tokenizer: Tokenizer, batch_size: int, split: str = 'train'):
        self.zip_file = zipfile.ZipFile(dataset_path, 'r')
        self.tokenizer = tokenizer
        self.batch_size = batch_size

        self.split = split
        self.tasks = self._get_tasks()

        # parse metadata.json
        metadata_json = json.load(self.zip_file.open('meta.json'))
        self.name = metadata_json['name']
        self.width = metadata_json['terminal']['width']
        self.height = metadata_json['terminal']['height']

    def _get_tasks(self) -> List[TerminalTaskInfo]:
        tasks = []
        for task_folder_info in self.zip_file.infolist():
            task_folder_name = task_folder_info.filename

            # check if is task_* folder
            if not task_folder_name.startswith("train/task_") or 'alternatives/' in task_folder_name or task_folder_name[-1] != '/':
                continue

            if task_folder_name.startswith(self.split + '/'):
                task_id = int(task_folder_name.split('/')[1].split('_')[1])
                task_json_file = self.zip_file.open(task_folder_name + 'task.json')
                task_json = json.load(task_json_file)

                task_name = task_json['name']
                task_prompt = self.zip_file.read(task_folder_name + 'task.md').decode('utf-8')
                task_difficulty = task_json['difficulty']
                terminal_recordings = []

                for recording_file_info in self.zip_file.infolist():
                    recording_file_name = recording_file_info.filename
                    if recording_file_name.startswith(task_folder_name + 'alternatives/recording_') and recording_file_name.endswith('.bin'):
                        recording_file = self.zip_file.open(recording_file_name)
                        terminal_recording = TerminalDatasetReader(recording_file, self.tokenizer)
                        terminal_recordings.append(terminal_recording)

                task_info = TerminalTaskInfo(task_id, task_name, task_prompt, task_difficulty, terminal_recordings)
                tasks.append(task_info)
        return tasks

    def __iter__(self):
        while True:
            for task in self.tasks:
                for recording in task.terminal_recordings:
                    yield iter(recording)
