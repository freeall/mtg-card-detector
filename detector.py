#!/usr/bin/env python

import argparse
import magic_card_detector as mcg

def main():
  """
  Python MTG Card Detector.
  Can be used also purely through the defined classes.
  """

  # Add command line parser
  parser = argparse.ArgumentParser(
    description=
      'Recognize Magic: the Gathering cards from an image.\n' +
      'Authors: Timo Ikonen <timo.ikonen(at)iki.fi>\n' +
      '         Tobias Baunbæk Christensen <freeall(at)gmail.com>'
  )

  parser.add_argument(
    'input_path',
    help='path containing the images to be analyzed'
  )
  parser.add_argument(
    '-p',
    '--phash',
    required=True,
    help='pre-calculated phash reference file'
  )

  args = parser.parse_args()

  # Instantiate the detector
  card_detector = mcg.MagicCardDetector()

  # Read the reference and test data sets
  card_detector.read_prehashed_reference_data(args.phash)
  card_detector.read_and_adjust_test_images(args.input_path)

  # Run the card detection and recognition.
  card_detector.run_recognition()

if __name__ == '__main__':
  main()
