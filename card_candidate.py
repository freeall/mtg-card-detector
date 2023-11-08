from dataclasses import dataclass
import numpy as np
from shapely.geometry.polygon import Polygon

@dataclass
class CardCandidate:
    """
    Class representing a segment of the image that may be a recognizable card.
    """
    image: np.ndarray
    bounding_quad: Polygon
    image_area_fraction: float
    is_recognized: bool = False
    recognition_score: float = 0.
    is_fragment: bool = False
    name: str = 'unknown'
    set: str = 'setsetsets'
    id: str = 'ididid'
    number: str = 'numnumnum'

    # def __init__(self, im_seg, bquad, fraction):
    #    self.image = im_seg
    #    self.bounding_quad = bquad
    #    self.is_recognized = False
    #    self.recognition_score = 0.
    #    self.is_fragment = False
    #    self.image_area_fraction = fraction
    #    self.name = 'unknown'

    def contains(self, other):
        """
        Returns whether the bounding polygon of the card candidate
        contains the bounding polygon of the other candidate.
        """
        return bool(other.bounding_quad.within(self.bounding_quad) and
                    other.name == self.name)

