import os
from dataclasses import dataclass
from pickle import dump, load
import cv2
from typing import List
from filter import Filter
from dataset import Dataset
from reader import OCRReader


@dataclass
class TestSettings:
    """This class contains all the settings for a test."""

    reader: OCRReader
    filters: List[Filter]  # list of filters to apply to image in order

    def __str__(self) -> str:
        filtersString = ", ".join([str(filter) for filter in self.filters])
        return f"{self.reader}: {filtersString}"


@dataclass
class SingleTestResult(TestSettings):
    """This class contains the result of a single test."""

    time: float
    success: float
    resultText: str
    expectedText: str


@dataclass
class MultiTestResult:
    results: List[SingleTestResult]
    testSettings: TestSettings

    totalTime: float = 0
    totalSuccess: float = 0

    averageTime: float = 0
    averageSuccess: float = 0

    def __post_init__(self):
        # calculate average time and success
        self.totalTime = 0
        self.totalSuccess = 0

        for result in self.results:
            self.totalTime += result.time
            self.totalSuccess += result.success

        self.averageTime = self.totalTime / len(self.results)
        self.averageSuccess = self.totalSuccess / len(self.results)

    def __str__(self) -> str:
        return self.testSettings.__str__()


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
        for currentFilter in self.testSettings.filters:
            image = currentFilter.filter(image)

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
    def getSuccessRate(self, text: list, expected: str) -> float:
        # make a union of lowercased words and calculate how many of them are in the expected text
        words = set([word.lower() for line in text for word in line.split()])
        expectedWords = set([word.lower() for word in expected.split()])

        return len(words.intersection(expectedWords)) / len(expectedWords)

    def __str__(self):
        return f"TestCase({self.testSettings}, {self.imagePath})"


class TesterUtil:
    __test__ = False
    """
    Tester utility for OCR engines with filters and cache their results for better performance.
    """

    def __init__(self, cachePath: str, dataset: Dataset):
        self.cachePath = cachePath
        self.cache = {}
        self.loadCache()
        self.dataset = dataset

    def loadCache(self):
        """
        Loads the cache from the cache file.
        """

        # check if the cache file exists
        if not os.path.exists(self.cachePath):
            return

        # load the cache
        with open(self.cachePath, "rb") as file:
            self.cache = load(file)

    def saveCache(self):
        """
        This method will save the cache to the cache file.
        """

        # save the cache
        with open(self.cachePath, "wb") as file:
            dump(self.cache, file)

    def test(
        self, testSettings: TestSettings, memeCount=10, ignoreCache=False
    ) -> SingleTestResult:
        """
        This method will test a specific test case and return the result.
        """

        id = str(testSettings)

        # check if we already have a cached result
        if not ignoreCache and id in self.cache:
            return self.cache[id]

        # load first memeCount memes
        memesDataset = self.dataset.get(memeCount)

        # create a list of test cases
        testCases = [
            TestCase(testSettings, meme.imagePath, meme.expectedText)
            for meme in memesDataset
        ]

        # run all test cases
        results = [testCase.test() for testCase in testCases]

        # cache the result
        self.cache[id] = MultiTestResult(
            results, testSettings, testSettings.reader, testSettings.filters
        )

        self.saveCache()

        return self.cache[id]
