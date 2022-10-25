import os
import cv2
import json
from dataclasses import dataclass
from typing import List
from reader import OCRReader
from filter import Filter


@dataclass
class TestSettings:
    reader: OCRReader
    filters: List[Filter]  # list of filters to apply to image in order


@dataclass
class SingleTestResult(TestSettings):
    time: float
    success: float
    resultText: str
    expectedText: str

    def toJSON(self):
        copy = self.__dict__.copy()
        copy["reader"] = str(copy["reader"])
        copy["filters"] = [str(filter) for filter in copy["filters"]]

        return json.dumps(copy)


class TestCase:
    __test__ = False

    def __init__(self, testSettings: TestSettings, imagePath: str, expectedText: str):
        self.testSettings = testSettings
        self.imagePath = imagePath
        self.expectedText = expectedText

    def test(self) -> SingleTestResult:
        """This method will test the OCR engine with filters applied."""

        # load image
        image = cv2.imread(self.imagePath)

        # start measuring time in milliseconds
        time = cv2.getTickCount()

        # apply filters
        for filter in self.testSettings.filters:
            image = filter.filter(image)

        # read text
        text = self.testSettings.reader.read(image)

        # stop measuring time
        time = (cv2.getTickCount() - time) / cv2.getTickFrequency() * 1000

        # compare text
        success = self.getSuccessRate(text, self.expectedText)

        # return result
        return SingleTestResult(
            self.testSettings.reader,
            self.testSettings.filters,
            time,
            success,
            text,
            self.expectedText,
        )

    # compares result to expected text
    # resulting float number is a percentage of how many words both strings have in common
    def getSuccessRate(self, text: list, expected: list) -> float:
        # make a union of lowercased words and calculate how many of them are in the expected text
        words = set([word.lower() for line in text for word in line.split()])
        expectedWords = set(
            [word.lower() for line in expected for word in line.split()]
        )
        return len(words.intersection(expectedWords)) / len(expectedWords)

    def __str__(self):
        return f"TestCase({self.testSettings}, {self.imagePath})"


class TesterUtil:
    __test__ = False
    """
    Tester utility for OCR engines with filters and cache their results for better performance.
    """

    def __init__(self, cachePath: str):
        self.cachePath = cachePath
        self.cache = {}
        self.loadCache()

    def loadCache(self):
        """
        Loads the cache from the cache file.
        """

        # Check if the cache file exists
        if not os.path.exists(self.cachePath):
            return

        # Load the cache
        with open(self.cachePath, "r") as file:
            self.cache = json.load(file)

    def saveCache(self):
        """
        This method will save the cache to the cache file.
        """

        # Save the cache
        with open(self.cachePath, "w") as file:
            json.dump(self.cache, file)

    def test(self, testCase: TestCase, id: str):
        """
        This method will test a specific test case and return the result.
        """

        # Check if we already have a cached result
        if id in self.cache:
            return self.cache[id]

        # Test the OCR engine
        result = testCase.test()

        # Cache the result
        self.cache[id] = result.toJSON()
        self.saveCache()

        return result
