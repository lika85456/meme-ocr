"""module that contains the dataset class"""
from dataclasses import dataclass
import os
from typing import List


@dataclass
class DatasetEntry:
    """This class contains a single entry of a dataset."""

    image_path: str
    expected_text: str
    entry_id: str


# This should be a class, even though it only contains a single method.
# pylint: disable=too-few-public-methods
class Dataset:
    """This class contains the dataset for a test."""

    def __init__(self, path: str) -> None:
        self.__load_dataset(path)

    def __load_dataset(self, path: str) -> None:
        """This method loads the dataset from a given path."""
        # path contains meme ids with "png" or "jpg" extension

        files = os.listdir(path)

        # find all files with "png" or "jpg" extension
        files = [
            file for file in files if file.endswith(".png") or file.endswith(".jpg")
        ]

        # load images
        dataset: List[DatasetEntry] = []
        for file in files:
            entry_id = file.split(".")[0]

            # if id already in dataset
            if entry_id in [entry.entry_id for entry in dataset]:
                continue

            # id.txt contains expected text
            expected_text = None
            with open(
                os.path.join(path, f"{entry_id}.txt"), "r", encoding="utf-8"
            ) as expected_text_file:
                expected_text = expected_text_file.read()

            # load image
            image_path = os.path.join(path, file)

            # if image or expected text is missing
            if image_path is None or expected_text is None:
                print(f"missing image or expected text for {entry_id}")
                continue

            dataset.append(DatasetEntry(image_path, expected_text, entry_id))

        self.dataset = dataset

    def get(self, count: int) -> list:
        """This method returns a list of dataset entries."""
        return self.dataset[:count]
