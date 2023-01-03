from abc import abstractclassmethod
import cv2
import numpy as np
import pytesseract

# abstract class for filters
class Filter:
    @abstractclassmethod
    def filter(self, image: cv2.Mat) -> cv2.Mat:
        """applies a filter on image"""
        pass


class GrayscaleFilter(Filter):
    def filter(self, image: cv2.Mat) -> cv2.Mat:
        """applies a grayscale filter on image"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def __str__(self) -> str:
        return "grayscale"

class CannyEdgeFilter(Filter):
    def filter(self, image: cv2.Mat) -> cv2.Mat:
        """applies a canny edge filter on image"""
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        return cv2.Canny(gray, 100, 200)

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
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canny, contours, -1, (255, 255, 255), 3)
        return canny

    def __str__(self) -> str:
        return "canny edge with filled shapes"

class SharpenFilter(Filter):
    def filter(self, image: cv2.Mat) -> cv2.Mat:
        """applies a sharpen filter on image"""
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

        return cv2.filter2D(gray, -1, kernel)

    def __str__(self) -> str:
        return "sharpen"

class OneBitColorFilter(Filter):
    def filter(self, image: cv2.Mat) -> cv2.Mat:
        """applies a one bit color filter on image"""
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        return cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]

    def __str__(self) -> str:
        return "one bit color"

class GaussianBlurFilter(Filter):
    def filter(self, image: cv2.Mat) -> cv2.Mat:
        """applies a gaussian blur filter on image"""
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        return cv2.GaussianBlur(gray, (5, 5), 0)

    def __str__(self) -> str:
        return "gaussian blur"

def detectTextColor(image: cv2.Mat) -> str:
    """detects the color of the text in image"""
    # convert image to grayscale if not already
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # sum pixels lighter than 128
    light = np.sum(gray < 20)

    # sum pixels darker than 128
    dark = np.sum(gray > 235)

    # return color with more pixels
    if light > dark:
        return "light"
    else:
        return "dark"

# define filter for normalizing images to black text on white background
class NormalizeFilter(Filter):
    def filter(self, image: cv2.Mat) -> cv2.Mat:
        """applies a normalize filter on image"""
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # detect text color
        color = detectTextColor(gray)

        # invert image if text is light
        if color == "light":
            return cv2.bitwise_not(gray)

        return gray

    def __str__(self) -> str:
        return "normalize"

class SharpenOneBit(Filter):
    def filter(self, image: cv2.Mat) -> cv2.Mat:
        sharpened = SharpenFilter().filter(image)
        return OneBitColorFilter().filter(sharpened)

    def __str__(self) -> str:
        return "Sharpen one bit"

def levels(image: cv2.Mat, black_point, white_point, midtone) -> cv2.Mat:
    image = image.astype(np.float32)
    scale = 255 / (white_point - black_point)

    image = (image - black_point) * scale
    image = np.clip(image, 0, 255)

    image = (image - 127.5) * midtone + 127.5
    image = np.clip(image, 0, 255)
    
    image = image.astype(np.uint8)
    return image

class Custom(Filter):
    def filter(self, image: cv2.Mat) -> cv2.Mat:
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # high pass
        

        return gray


    def __str__(self) -> str:
        return "Custom"