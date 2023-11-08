#!/usr/bin/env python

import pickle
import magic_card_detector as mcg
import argparse
import os

def main():
  parser = argparse.ArgumentParser(
    description=
      'Generate phashes for mtg-card-detector. \n' +
      '\n'+
      'IMPORTANT: Expect files to be named COLLECTOR-NUMBER_SOME-ID_CARD-NAME.jpg\n' +
      '           e.g. 232_b0faa7f2-b547-42c4-a810-839da50dadfe_Black Lotus.jpg\n'+
      '           Uses the underscores (_) and extension (.jpg) to identify name\n'+
      '\n'+
      'If you want to train with different versions of the same card, e.g. some versions are a bit crooked, you can easily do that just by adding the different versions.'
  )
  parser.add_argument(
    '--folder',
    '-f',
    type=str,
    required=True,
    help='Folder containing the images to be analyzed and added to the phash file.\nIf --sets-in-subfolders is provided then it is assumes that all sets are in sub-folders of this, e.g. /sets/lea, /sets/ktk, etc'
  )
  parser.add_argument(
    '--output',
    '-o',
    type=str,
    required=True,
    help='The file to store the phashes in, e.g. all_phashes.dat'
  )
  parser.add_argument(
    '--set',
    '-s',
    type=str,
    help='The set code of this set, e.g. "lea" for "Limited Edition Alpha", or "ktk" for "Khans of Tarkir"'
  )
  parser.add_argument(
    '--sets-in-subfolders',
    required=False,
    action='store_true',
    help='If all your sets are in subfolder, e.g. /sets/lea, /sets/ktk, etc, then use this. Then you do not need to use --set.'
  )
  parser.add_argument(
    '--append',
    '-a',
    required=False,
    action='store_true',
    help='Append to phash file, e.g. --append all_phashes.dat'
  )

  args = parser.parse_args()

  if (args.sets_in_subfolders):
    add_all_subfolders(args.folder, args.output, args.append)
  else:
    add(args.folder, args.set, args.output, args.append)


def add_all_subfolders(folder, output_file, append):
  if (not append):
    try:
      os.remove(output_file)
    except:
      pass
  if (not os.path.exists(output_file)):
    f = open(output_file, 'wb')
    pickle.dump([], f)
    f.close()

  sets = os.listdir(folder)
  for i in range(len(sets)):
    set = sets[i]
    print('[' + str(i + 1) + '/' + str(len(sets)) + '] Generating phash from images from the "' + set + '" set')
    add(folder + '/' + set, set, output_file, True)


def add(folder, set, output_file, append):
  if (folder[-1] == '/'):
    folder = folder[:-1]
  hlist = []
  if (append):
    f = open(output_file, 'rb')
    hlist = pickle.load(f)
  card_detector = mcg.MagicCardDetector()
  card_detector.read_and_adjust_reference_images(folder + '/')

  for image in card_detector.reference_images:
    image.original = None
    image.clahe = None
    image.adjusted = None
    # image.name contains the filename
    card_parts = image.name.split('_', 2)
    image.set = set
    image.number = card_parts[0]
    image.id = card_parts[1]
    image.name = card_parts.pop()[:-4]
    hlist.append(image)

  f = open(output_file, 'wb')
  pickle.dump(hlist, f)

if __name__ == '__main__':
  main()
