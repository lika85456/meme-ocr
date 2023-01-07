"""This module contains the abstract class for readers and
the implementations for tesseract and easyocr"""

from abc import abstractmethod
from typing import List
import pytesseract
import easyocr

# pylint: disable=no-name-in-module
from cv2 import Mat


class OCRReader:
    """This class is the abstract class for readers."""

    @abstractmethod
    def read(self, image: Mat) -> List[str]:
        """reads text from image and returns the result as a list of strings separated by line"""

    # pylint: disable=line-too-long
    # @see https://pyimagesearch.com/2020/09/14/getting-started-with-easyocr-for-optical-character-recognition/
    def cleanup_text(self, text: str):
        """cleans up the text by removing non-ascii characters"""
        return "".join([c if ord(c) < 128 else "" for c in text]).strip()

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    def __json__(self):
        return str(self)


class TesseractReader(OCRReader):
    """This class is the implementation for tesseract"""

    def read(self, image: Mat) -> List[str]:
        """reads text from image and returns the result as a list of strings separated by line"""
        return self.cleanup_text(pytesseract.image_to_string(image)).splitlines()

    def __str__(self):
        return "tesseract"


class EasyOCRReader(OCRReader):
    """This class is the implementation for easyocr"""

    def read(self, image: Mat) -> List[str]:
        """reads text from image and returns the result as a list of strings separated by line"""

        reader = easyocr.Reader(["en"])
        result = reader.readtext(image)
        return [self.cleanup_text(text) for (bbox, text, prob) in result]

    def __str__(self):
        return "easyocr"
