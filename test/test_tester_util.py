"""This module contains tests for the tester util."""

from src.reader import TesseractReader
from src.tester_util import TestCase, TestSettings, TesterUtil
from src.dataset import Dataset


def test_tester_util():
    """This test tests the tester util."""

    cache_path = "/tmp/testCache.pyc"
    dataset_path = "./test/testDataset"

    # create test case
    test_settings = TestSettings(TesseractReader(), [])

    dataset = Dataset(path=dataset_path)

    # create tester util
    tester_util = TesterUtil(cache_path, dataset)

    # test the test case
    result = tester_util.test(test_settings, meme_count=1)

    # check if the result is correct
    assert str(result.test_settings.reader) == str(TesseractReader())
    assert result.test_settings.filters == []
    assert result.total_success == 1.0


def test_test_case():
    """This test tests the test case."""

    image_path = "./test/testDataset/testImage.png"
    expected_text = "\n".join(
        [
            "It was the best of",
            "times, it was the worst",
            "of times, it was the age",
            "of wisdom, it was the",
            "age of foolishness...",
        ]
    )

    # create test case
    test_case = TestCase(TestSettings(TesseractReader(), []), image_path, expected_text)

    # test the test case
    result = test_case.test()

    # check if the result is correct
    assert result.success
    assert result.expected_text == expected_text
    assert "\n".join(result.result_text) == expected_text
