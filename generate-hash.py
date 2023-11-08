import pickle
import magic_card_detector as mcg
import argparse

def main():
  parser = argparse.ArgumentParser(
    description=
      'Generate hashes for mtg-card-detector. \n' +
      '\n'+
      'IMPORTANT: Expect files to be named COLLECTOR-NUMBER_SOME-ID_CARD-NAME.jpg\n' +
      '           e.g. 232_b0faa7f2-b547-42c4-a810-839da50dadfe_Black Lotus.jpg\n'+
      '           Uses the underscores (_) and extension (.jpg) to identify name\n'+
      '\n'+
      'If you want to train with different versions of the same card, e.g. some versions are a bit crooked, you can easily do that just by adding the different versions.'
  )

  parser.add_argument(
    '-s',
    '--set',
    type=str,
    required=True,
    help='The set code of this set, e.g. "lea" for "Limited Edition Alpha", or "ktk" for "Khans of Tarkir"'
  )
  parser.add_argument(
    '-f',
    '--folder',
    type=str,
    required=True,
    help='Folder containing the images to be analyzed and added to the hash file'
  )
  parser.add_argument(
    '-o',
    '--output',
    type=str,
    required=True,
    help='The file to store the hashes in, e.g. all_hashes.dat'
  )
  parser.add_argument(
    '-a',
    '--append',
    required=False,
    action='store_true',
    help='Append to hash file, e.g. --append all_hashes.dat'
  )

  args = parser.parse_args()
  add(args.folder, args.set, args.output, args.append)

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
