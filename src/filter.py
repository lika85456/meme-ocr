"""module that contains the filters"""
from abc import abstractclassmethod

# pylint: disable=no-name-in-module
from cv2 import (
    Mat,
    cvtColor,
    COLOR_BGR2GRAY,
    Canny,
    RETR_TREE,
    CHAIN_APPROX_SIMPLE,
    drawContours,
    filter2D,
    threshold,
    THRESH_BINARY,
    GaussianBlur,
    THRESH_TRUNC,
    bitwise_not,
    bitwise_and,
    addWeighted,
    resize,
    findContours,
    dilate,
)
import numpy as np

# abstract class for filters
class Filter:
    """abstract class for filters"""

    # pylint: disable=deprecated-decorator
    @abstractclassmethod
    def filter(cls, image: Mat) -> Mat:
        """applies a filter on image"""

    # pylint: disable=deprecated-decorator
    @abstractclassmethod
    def __str__(cls):
        """returns the name of the filter"""

    def __json__(self):
        return str(self)


class NoFilter(Filter):
    """filter that does not apply any filter on image;"""

    # pylint: disable=arguments-differ
    def filter(self, image: Mat) -> Mat:
        """returns image without applying any filter"""
        return NormalizeFilter().filter(image)

    def __str__(self) -> str:
        return "no filter"


class GrayscaleFilter(Filter):
    """filter that applies a grayscale filter on image"""

    # pylint: disable=arguments-differ
    def filter(self, image: Mat) -> Mat:
        """applies a grayscale filter on image"""
        gray = cvtColor(image, COLOR_BGR2GRAY)
        return NormalizeFilter().filter(gray)

    def __str__(self) -> str:
        return "grayscale"


class CannyEdgeFilter(Filter):
    """filter that applies a canny edge filter on image"""

    # pylint: disable=arguments-differ
    def filter(self, image: Mat) -> Mat:
        """applies a canny edge filter on image"""
        if len(image.shape) > 2:
            gray = cvtColor(image, COLOR_BGR2GRAY)
        else:
            gray = image

        canny = Canny(gray, 100, 200)

        return NormalizeFilter().filter(canny)

    def __str__(self) -> str:
        return "canny edge"


class CannyEdgeWithFilledShapesFilter(Filter):
    """filter that applies a canny edge filter and fills the resulting shapes on image"""

    # pylint: disable=arguments-differ
    def filter(self, image: Mat) -> Mat:
        """applies a canny edge filter with filled shapes on image"""
        if len(image.shape) > 2:
            gray = cvtColor(image, COLOR_BGR2GRAY)
        else:
            gray = image

        canny = Canny(gray, 100, 200)
        contours, _ = findContours(canny, RETR_TREE, CHAIN_APPROX_SIMPLE)
        drawContours(canny, contours, -1, (255, 255, 255), 3)

        return NormalizeFilter().filter(canny)

    def __str__(self) -> str:
        return "canny edge with filled shapes"


class SharpenFilter(Filter):
    """filter that applies a sharpen filter on image"""

    # pylint: disable=arguments-differ
    def filter(self, image: Mat) -> Mat:
        """applies a sharpen filter on image"""
        if len(image.shape) > 2:
            gray = cvtColor(image, COLOR_BGR2GRAY)
        else:
            gray = image

        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

        sharpened = filter2D(gray, -1, kernel)

        return NormalizeFilter().filter(sharpened)

    def __str__(self) -> str:
        return "sharpen"


class OneBitColorFilter(Filter):
    """filter that applies a one bit color filter on image (binarization)"""

    # pylint: disable=arguments-differ
    def filter(self, image: Mat) -> Mat:
        """applies a one bit color filter on image"""
        if len(image.shape) > 2:
            gray = cvtColor(image, COLOR_BGR2GRAY)
        else:
            gray = image

        result = threshold(gray, 20, 235, THRESH_BINARY)[1]

        return NormalizeFilter().filter(result)

    def __str__(self) -> str:
        return "one bit color"


class GaussianBlurFilter(Filter):
    """filter that applies a gaussian blur filter on image"""

    # pylint: disable=arguments-differ
    def filter(self, image: Mat) -> Mat:
        """applies a gaussian blur filter on image"""
        if len(image.shape) > 2:
            gray = cvtColor(image, COLOR_BGR2GRAY)
        else:
            gray = image

        blurred = GaussianBlur(gray, (5, 5), 0)

        return NormalizeFilter().filter(blurred)

    def __str__(self) -> str:
        return "gaussian blur"


def detect_text_color(image: Mat) -> str:
    """detects the color of the text in image"""
    # convert image to grayscale if not already
    if len(image.shape) > 2:
        gray = cvtColor(image, COLOR_BGR2GRAY)
    else:
        gray = image

    light = np.sum(gray < 10)

    dark = np.sum(gray > 245)

    # return color with more pixels
    if light > dark:
        return "light"

    return "dark"


class NormalizeFilter(Filter):
    """filter that normalizes the image to have black text
    (it's not very good at it though) and also resizes the image
    up to 600 pixels in width or height"""

    # pylint: disable=arguments-differ
    def filter(self, image: Mat) -> Mat:
        """normalizes the image to have black text (it's not very good at it though)
        and also resizes the image up to 600 pixels in width or height"""
        if len(image.shape) > 2:
            gray = cvtColor(image, COLOR_BGR2GRAY)
        else:
            gray = image

        # detect text color
        color = detect_text_color(gray)

        # invert image if text is light
        if color == "light":
            gray = bitwise_not(gray)

        if gray.shape[0] > gray.shape[1]:
            scale = 600 / gray.shape[0]
        else:
            scale = 600 / gray.shape[1]

        gray = resize(gray, (0, 0), fx=scale, fy=scale)

        return gray

    def __str__(self) -> str:
        return "normalize"


def remove_irelevant_content(image: Mat) -> Mat:
    """removes irelevant content from image"""

    # create a blurred canny edge mask
    irelevant_mask = CannyEdgeWithFilledShapesFilter().filter(image)

    irelevant_mask = bitwise_not(irelevant_mask)
    irelevant_mask = dilate(irelevant_mask, np.ones((10, 10), np.uint8))
    irelevant_mask = threshold(irelevant_mask, 250, 255, THRESH_BINARY)[1]

    gray = bitwise_not(image)
    gray = bitwise_and(gray, gray, mask=irelevant_mask)
    gray = bitwise_not(gray)

    return gray


class Custom(Filter):
    """custom filter that removes irelevant content and then applies
    a sharpen filter and then enhances the sharpened image in the original image.
    Sadly it doesn't produce very good results
    """

    # pylint: disable=arguments-differ
    def filter(self, image: Mat) -> Mat:
        """
        First step is to remove irelevant image content
        by masking out everything that is not near edges
        """

        gray = GrayscaleFilter().filter(image)

        gray = remove_irelevant_content(gray)

        # remove colors over 245
        gray = threshold(gray, 254, 255, THRESH_TRUNC)[1]

        # sharpen
        sharpened = SharpenFilter().filter(gray)

        # enhance by 50% in sharpened mask using addWeighted
        gray = addWeighted(sharpened, 0.3, gray, 0.7, 0)

        return gray

    def __str__(self) -> str:
        return "Custom"
