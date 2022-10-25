import cv2
from reader import TesseractReader
from testerUtil import TestCase, TestSettings, TesterUtil
import os


def test_itTestsTheOCR():
    cachePath = "./src/testCache.json"
    testerUtil = TesterUtil(cachePath)

    testCase = TestCase(
        TestSettings(TesseractReader(), []),
        "./src/testImage.png",
        [
            "It was the best of",
            "times, it was the worst",
            "of times, it was the age",
            "of wisdom, it was the",
            "age of foolishness...",
        ],
    )

    testerUtil.test(testCase, "test_itTestsTheOCR")

    # remove cache file
    os.remove(cachePath)


def test_itReusesTheCache():
    cachePath = "./src/testCache.json"
    testerUtil = TesterUtil(cachePath)

    testCase = TestCase(
        TestSettings(TesseractReader(), []),
        "./src/testImage.png",
        [
            "It was the best of",
            "times, it was the worst",
            "of times, it was the age",
            "of wisdom, it was the",
            "age of foolishness...",
        ],
    )

    # measure time for first test
    time1 = cv2.getTickCount()
    testerUtil.test(testCase, "test_itReusesTheCache")
    time1 = (cv2.getTickCount() - time1) / cv2.getTickFrequency() * 1000

    # measure time for second test
    time2 = cv2.getTickCount()
    testerUtil.test(testCase, "test_itReusesTheCache")
    time2 = (cv2.getTickCount() - time2) / cv2.getTickFrequency() * 1000

    # assert that the second test is faster atleast twice
    assert time2 < time1 / 2

    # remove cache file
    os.remove(cachePath)
