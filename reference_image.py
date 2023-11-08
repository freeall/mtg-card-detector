import cv2
import imagehash
from PIL import Image as PILImage
import numpy as np

class ReferenceImage:
    """
    Container for a card image and the associated recoginition data.
    """

    def __init__(self, name, original_image, clahe, phash=None, set=None, number=None, id=None):
        self.name = name
        self.set = set
        self.number = number
        self.id = id
        self.original = original_image
        self.clahe = clahe
        self.adjusted = None
        self.phash = phash

        if self.original is not None:
            self.histogram_adjust()
            self.calculate_phash()

    def calculate_phash(self):
        """
        Calculates the perceptive hash for the image
        """
        self.phash = imagehash.phash(
            PILImage.fromarray(np.uint8(255 * cv2.cvtColor(
                self.adjusted, cv2.COLOR_BGR2RGB))),
            hash_size=32)

    def histogram_adjust(self):
        """
        Adjusts the image by contrast limited histogram adjustmend (clahe)
        """
        lab = cv2.cvtColor(self.original, cv2.COLOR_BGR2LAB)
        lightness, redness, yellowness = cv2.split(lab)
        corrected_lightness = self.clahe.apply(lightness)
        limg = cv2.merge((corrected_lightness, redness, yellowness))
        self.adjusted = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
