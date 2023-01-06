from dataclasses import dataclass
import cv2
import os


@dataclass
class DatasetEntry:
    """This class contains a single entry of a dataset."""

    imagePath: str
    expectedText: str
    id: str


class Dataset:
    """This class contains the dataset for a test."""

    def __init__(self, path: str) -> None:
        """This method loads the dataset from a file."""
        # path contains meme ids with "png" or "jpg" extension

        files = os.listdir(path)

        # find all files with "png" or "jpg" extension
        files = [
            file for file in files if file.endswith(".png") or file.endswith(".jpg")
        ]

        # load images
        dataset = []
        for file in files:
            id = file.split(".")[0]

            # if id already in dataset
            if id in [entry.id for entry in dataset]:
                continue

            # id.txt contains expected text
            expectedText = None
            with open(os.path.join(path, f"{id}.txt"), "r", encoding="utf-8") as f:
                expectedText = f.read()

            # load image
            imagePath = os.path.join(path, file)

            # if image or expected text is missing
            if imagePath is None or expectedText is None:
                print(f"missing image or expected text for {id}")
                continue

            dataset.append(DatasetEntry(imagePath, expectedText, id))

        self.dataset = dataset

    def get(self, count: int) -> list:
        """This method returns a list of dataset entries."""
        return self.dataset[:count]
