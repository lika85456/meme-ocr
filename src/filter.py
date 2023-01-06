from abc import abstractclassmethod
from cv2 import cv2
import numpy as np

# abstract class for filters
class Filter:
    @abstractclassmethod
    def filter(self, image: cv2.Mat) -> cv2.Mat:
        """applies a filter on image"""
        pass

    @abstractclassmethod
    def __str__(self):
        raise NotImplementedError

    def __json__(self):
        return str(self)


class NoFilter(Filter):
    def filter(self, image: cv2.Mat) -> cv2.Mat:
        """returns image without applying any filter"""
        return NormalizeFilter().filter(image)

    def __str__(self) -> str:
        return "no filter"


class GrayscaleFilter(Filter):
    def filter(self, image: cv2.Mat) -> cv2.Mat:
        """applies a grayscale filter on image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return NormalizeFilter().filter(gray)

    def __str__(self) -> str:
        return "grayscale"


class CannyEdgeFilter(Filter):
    def filter(self, image: cv2.Mat) -> cv2.Mat:
        """applies a canny edge filter on image"""
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        canny = cv2.Canny(gray, 100, 200)

        return NormalizeFilter().filter(canny)

    def __str__(self) -> str:
        return "canny edge"


class CannyEdgeWithFilledShapesFilter(Filter):
    def filter(self, image: cv2.Mat) -> cv2.Mat:
        """applies a canny edge filter with filled shapes on image"""
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        canny = cv2.Canny(gray, 100, 200)
        contours, hierarchy = cv2.findContours(
            canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canny, contours, -1, (255, 255, 255), 3)
        return NormalizeFilter().filter(canny)

    def __str__(self) -> str:
        return "canny edge with filled shapes"


class SharpenFilter(Filter):
    def filter(self, image: cv2.Mat) -> cv2.Mat:
        """applies a sharpen filter on image"""
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

        sharpened = cv2.filter2D(gray, -1, kernel)

        return NormalizeFilter().filter(sharpened)

    def __str__(self) -> str:
        return "sharpen"


class OneBitColorFilter(Filter):
    def filter(self, image: cv2.Mat) -> cv2.Mat:
        """applies a one bit color filter on image"""
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        result = cv2.threshold(gray, 20, 235, cv2.THRESH_BINARY)[1]

        return NormalizeFilter().filter(result)

    def __str__(self) -> str:
        return "one bit color"


class GaussianBlurFilter(Filter):
    def filter(self, image: cv2.Mat) -> cv2.Mat:
        """applies a gaussian blur filter on image"""
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        return NormalizeFilter().filter(blurred)

    def __str__(self) -> str:
        return "gaussian blur"


def detectTextColor(image: cv2.Mat) -> str:
    """detects the color of the text in image"""
    # convert image to grayscale if not already
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    light = np.sum(gray < 10)

    dark = np.sum(gray > 245)

    # return color with more pixels
    if light > dark:
        return "light"
    else:
        return "dark"


class NormalizeFilter(Filter):
    def filter(self, image: cv2.Mat) -> cv2.Mat:
        """normalizes the image to have black text (it's not very good at it though) and also resizes the image up to 600 pixels in width or height"""
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # detect text color
        color = detectTextColor(gray)

        # invert image if text is light
        if color == "light":
            gray = cv2.bitwise_not(gray)

        if gray.shape[0] > gray.shape[1]:
            scale = 600 / gray.shape[0]
        else:
            scale = 600 / gray.shape[1]

        gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale)

        return gray

    def __str__(self) -> str:
        return "normalize"


def removeIrelevantContent(image: cv2.Mat) -> cv2.Mat:

    # create a blurred canny edge mask
    irelevantMask = CannyEdgeWithFilledShapesFilter().filter(image)

    # blur the mask so we do not lose any important content near the text
    # irelevantMask = GaussianBlurFilter().filter(irelevantMask)

    # invert
    irelevantMask = cv2.bitwise_not(irelevantMask)

    # create a new mask, that will have pixel as white if they are 5 pixels near full white, else black
    irelevantMask = cv2.dilate(irelevantMask, np.ones((10, 10), np.uint8))

    # treshold to get a binary mask
    irelevantMask = cv2.threshold(irelevantMask, 250, 255, cv2.THRESH_BINARY)[1]

    # invert back
    irelevantMask = cv2.bitwise_not(irelevantMask)

    # invert blurredCanny (white important, black irelevant)
    blurredCanny = cv2.bitwise_not(irelevantMask)

    # invert image
    gray = cv2.bitwise_not(image)

    # remove irelevant content from image
    gray = cv2.bitwise_and(gray, gray, mask=blurredCanny)

    # invert gray
    gray = cv2.bitwise_not(gray)

    return gray


class Custom(Filter):
    def filter(self, image: cv2.Mat) -> cv2.Mat:
        """
        First step is to remove irelevant image content by masking out everything that is not near edges
        """

        gray = GrayscaleFilter().filter(image)

        gray = removeIrelevantContent(gray)

        # remove colors over 245
        gray = cv2.threshold(gray, 254, 255, cv2.THRESH_TRUNC)[1]

        # sharpen
        sharpened = SharpenFilter().filter(gray)

        # enhance by 50% in sharpened mask using addWeighted
        gray = cv2.addWeighted(sharpened, 0.3, gray, 0.7, 0)

        return gray

    def __str__(self) -> str:
        return "Custom"
