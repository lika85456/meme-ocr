from abc import abstractclassmethod
from cv2 import Mat

# abstract class for filters
class Filter:
    @abstractclassmethod
    def filter(self, image: Mat) -> Mat:
        """applies a filter on image creating a new one and returning its new path"""
        pass


# class TresholdFilter(Filter):
#     def __init__(self, treshold: int):
#         self.treshold = treshold

#     def filter(self, imagePath: str) -> str:
#         """applies a filter on image creating a new one and returning its new path"""
#         image = self._loadImage(imagePath)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         _, image = cv2.threshold(image, self.treshold, 255, cv2.THRESH_BINARY)
#         imagePath = imagePath.replace(".jpg", "_treshold.jpg")
#         self._saveImage(image, imagePath)
#         return imagePath
