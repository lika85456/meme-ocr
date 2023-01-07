"""Util for testing OCR engines with filters and cache their results"""
import os
from dataclasses import dataclass
from pickle import dump, load
from typing import List

# ???
# pylint: disable=no-name-in-module
from cv2 import imread, getTickCount, getTickFrequency
from src.filter import Filter
from src.dataset import Dataset
from src.reader import OCRReader


@dataclass
class TestSettings:
    """This class contains all the settings for a test."""

    reader: OCRReader
    filters: List[Filter]  # list of filters to apply to image in order

    def __str__(self) -> str:
        filters_string = ", ".join([str(filter) for filter in self.filters])
        return f"{self.reader}: {filters_string}"


@dataclass
class SingleTestResult(TestSettings):
    """This class contains the result of a single test."""

    time: float
    success: float
    result_text: str
    expected_text: str


@dataclass
class MultiTestResult:
    """This class contains the result of multiple tests."""

    results: List[SingleTestResult]
    test_settings: TestSettings

    total_time: float = 0
    total_success: float = 0

    average_time: float = 0
    average_success: float = 0

    def __post_init__(self):
        # calculate average time and success
        self.total_time = 0
        self.total_success = 0

        for result in self.results:
            self.total_time += result.time
            self.total_success += result.success

        self.average_time = self.total_time / len(self.results)
        self.average_success = self.total_success / len(self.results)

    def __str__(self) -> str:
        return self.test_settings.__str__()


class TestCase:
    """This class contains a single test case."""

    __test__ = False

    def __init__(
        self, test_settings: TestSettings, image_path: str, expected_text: str
    ):
        self.test_settings = test_settings
        self.image_path = image_path
        self.expected_text = expected_text

    def test(self) -> SingleTestResult:
        """This method will test the OCR engine with filters applied."""

        # load image
        image = imread(self.image_path)

        # start measuring time in milliseconds
        time = getTickCount()

        # apply filters
        for current_filter in self.test_settings.filters:
            image = current_filter.filter(image)

        # read text
        text = self.test_settings.reader.read(image)

        # stop measuring time
        time = (getTickCount() - time) / getTickFrequency() * 1000

        # compare text
        success = self.get_success_rate(text, self.expected_text)

        # return result
        return SingleTestResult(
            self.test_settings.reader,
            self.test_settings.filters,
            time,
            success,
            text,
            self.expected_text,
        )

    def get_success_rate(self, text: list, expected: str) -> float:
        """
        Compares result to expected text. Resulting float number is
        a percentage of how many words both strings have in common.
        """

        # make a union of lowercased words and calculate how many of them are in the expected text

        # Dear pylint, I did consider it, but I don't want to use a set comprehension here.
        # pylint: disable=consider-using-set-comprehension
        words = set([word.lower() for line in text for word in line.split()])
        expected_words = set([word.lower() for word in expected.split()])

        return len(words.intersection(expected_words)) / len(expected_words)

    def __str__(self):
        return f"TestCase({self.test_settings}, {self.image_path})"


class TesterUtil:
    """
    Tester utility for OCR engines with filters and cache their results for better performance.
    """

    __test__ = False

    def __init__(self, cache_path: str, dataset: Dataset):
        self.cache_path = cache_path
        self.cache = {}
        self.load_cache()
        self.dataset = dataset

    def load_cache(self):
        """
        Loads the cache from the cache file.
        """

        # check if the cache file exists
        if not os.path.exists(self.cache_path):
            return

        # load the cache
        with open(self.cache_path, "rb") as file:
            try:
                self.cache = load(file)
            except ModuleNotFoundError:
                self.cache = {}

    def save_cache(self):
        """
        This method will save the cache to the cache file.
        """

        # save the cache
        with open(self.cache_path, "wb") as file:
            dump(self.cache, file)

    def test(
        self, test_settings: TestSettings, meme_count=10, ignore_cache=False
    ) -> SingleTestResult:
        """
        This method will test a specific test case and return the result.
        """

        test_settings_id = str(test_settings)

        # check if we already have a cached result
        if not ignore_cache and test_settings_id in self.cache:
            return self.cache[test_settings_id]

        # load first memeCount memes
        meme_dataset = self.dataset.get(meme_count)

        # create a list of test cases
        test_cases = [
            TestCase(test_settings, meme.image_path, meme.expected_text)
            for meme in meme_dataset
        ]

        # run all test cases
        results = [testCase.test() for testCase in test_cases]

        # cache the result
        self.cache[test_settings_id] = MultiTestResult(
            results, test_settings, test_settings.reader, test_settings.filters
        )

        self.save_cache()

        return self.cache[test_settings_id]
