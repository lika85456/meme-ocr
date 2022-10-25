from abc import abstractmethod
from dataclasses import dataclass
from typing import List
import pytesseract
import easyocr
from cv2 import Mat

# abstract class for readers
class OCRReader:
    def read(self, image: Mat) -> List[str]:
        raise NotImplementedError

    # helper method for cleaning text
    # @see https://pyimagesearch.com/2020/09/14/getting-started-with-easyocr-for-optical-character-recognition/
    def cleanupText(self, text: str):
        return "".join([c if ord(c) < 128 else "" for c in text]).strip()

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    def __json__(self):
        return str(self)


class TesseractReader(OCRReader):
    def read(self, image: Mat) -> List[str]:
        """reads text from image and returns the result as a list of strings separated by line"""
        return self.cleanupText(pytesseract.image_to_string(image)).splitlines()

    def __str__(self):
        return "tesseract"


class EasyOCRReader(OCRReader):
    def read(self, image: Mat) -> List[str]:
        """reads text from image and returns the result as a list of strings separated by line"""

        reader = easyocr.Reader(["en"])
        result = reader.readtext(image)
        return [self.cleanupText(text) for (bbox, text, prob) in result]

    def __str__(self):
        return "easyocr"
