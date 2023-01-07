"""This module contains tests for the reader module."""

# pylint: disable=no-name-in-module
from cv2 import imread
from src.reader import TesseractReader, EasyOCRReader


TEST_IMAGE_PATH = "./test/testDataset/testImage.png"


def test_reads_text_from_image_tesseract():
    """This test tests the tesseract reader."""

    reader = TesseractReader()
    image = imread(TEST_IMAGE_PATH)
    text = reader.read(image)
    assert text == [
        "It was the best of",
        "times, it was the worst",
        "of times, it was the age",
        "of wisdom, it was the",
        "age of foolishness...",
    ]


def test_reads_text_from_image_easyocr():
    """This test tests the easy ocr reader."""

    reader = EasyOCRReader()
    image = imread(TEST_IMAGE_PATH)
    text = reader.read(image)
    assert text == [
        "It was the best of",
        "times, it was the worst",
        "of times, it was the age",
        "of wisdom; it was the",
        "age of foolishness .",
    ]
