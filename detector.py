#!/usr/bin/env python

import argparse
import magic_card_detector as mcg
import sys

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
    '--phash',
    '-p',
    required=True,
    help='pre-calculated phash reference file'
  )
  parser.add_argument(
    '--continuous',
    '-c',
    action='store_true',
    help='do not exit, but continuously read filenames from stdin'
  )
  parser.add_argument(
    '--input_path',
    '-i',
    help='folder containing images to be analyzed'
  )

  args = parser.parse_args()

  # Instantiate the detector
  card_detector = mcg.MagicCardDetector()

  # Read the reference and test data sets
  card_detector.read_prehashed_reference_data(args.phash)
  if (args.input_path):
    card_detector.read_and_adjust_test_images(args.input_path)
    card_detector.run_recognition()
  if (args.continuous):

    for line in sys.stdin:
      filename = line.strip()
      card_detector.read_and_adjust_single_image(filename)
      card_detector.run_recognition()
      sys.stdout.flush()

if __name__ == '__main__':
  main()
