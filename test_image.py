import cv2
from itertools import product
from copy import deepcopy

class TestImage:
    """
    Container for a card image and the associated recoginition data.
    """

    def __init__(self, name, original_image, clahe):
        self.name = name
        self.original = original_image
        self.clahe = clahe
        self.adjusted = None
        self.phash = None
        self.histogram_adjust()
        # self.calculate_phash()

        self.candidate_list = []

    def histogram_adjust(self):
        """
        Adjusts the image by contrast limited histogram adjustmend (clahe)
        """
        lab = cv2.cvtColor(self.original, cv2.COLOR_BGR2LAB)
        lightness, redness, yellowness = cv2.split(lab)
        corrected_lightness = self.clahe.apply(lightness)
        limg = cv2.merge((corrected_lightness, redness, yellowness))
        self.adjusted = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def mark_fragments(self):
        """
        Finds doubly (or multiply) segmented cards and marks all but one
        as a fragment (that is, an unnecessary duplicate)
        """
        for (candidate, other_candidate) in product(self.candidate_list,
                                                    repeat=2):
            if candidate.is_fragment or other_candidate.is_fragment:
                continue
            if ((candidate.is_recognized or other_candidate.is_recognized) and
                    candidate is not other_candidate):
                i_area = candidate.bounding_quad.intersection(
                    other_candidate.bounding_quad).area
                min_area = min(candidate.bounding_quad.area,
                               other_candidate.bounding_quad.area)
                if i_area > 0.5 * min_area:
                    if (candidate.is_recognized and
                            other_candidate.is_recognized):
                        if (candidate.recognition_score <
                                other_candidate.recognition_score):
                            candidate.is_fragment = True
                        else:
                            other_candidate.is_fragment = True
                    else:
                        if candidate.is_recognized:
                            other_candidate.is_fragment = True
                        else:
                            candidate.is_fragment = True


    def return_recognized(self):
        """
        Returns a list of recognized and non-fragment card candidates.
        """
        recognized_list = []
        for candidate in self.candidate_list:
            if candidate.is_recognized and not candidate.is_fragment:
                recognized_list.append(candidate)
        return recognized_list

    def discard_unrecognized_candidates(self):
        """
        Trims the candidate list to keep only the recognized ones
        """
        recognized_list = deepcopy(self.return_recognized())
        self.candidate_list.clear()
        self.candidate_list = recognized_list

    def may_contain_more_cards(self):
        """
        Simple area-based test to see if using a different segmentation
        algorithm may lead to finding more cards in the image.
        """
        recognized_list = self.return_recognized()
        if not recognized_list:
            return True
        tot_area = 0.
        min_area = 1.
        for card in recognized_list:
            tot_area += card.image_area_fraction
            if card.image_area_fraction < min_area:
                min_area = card.image_area_fraction
        return bool(tot_area + 1.5 * min_area < 1.)
