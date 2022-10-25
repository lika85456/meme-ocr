from reader import TesseractReader, EasyOCRReader
import cv2

testImagePath = "./src/testImage.png"


def test_readsTextFromImageTesseract():
    reader = TesseractReader()
    image = cv2.imread(testImagePath)
    text = reader.read(image)
    assert text == [
        "It was the best of",
        "times, it was the worst",
        "of times, it was the age",
        "of wisdom, it was the",
        "age of foolishness...",
    ]


def test_readsTextFromImageEasyOCR():
    reader = EasyOCRReader()
    image = cv2.imread(testImagePath)
    text = reader.read(image)
    assert text == [
        "It was the best of",
        "times, it was the worst",
        "of times, it was the age",
        "of wisdom; it was the",
        "age of foolishness .",
    ]
