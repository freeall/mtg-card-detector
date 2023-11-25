# mtg-card-detector

Detects Magic: The Gathering cards in images and outputs JSON as
``` json
{
  "name": "Counterspell",
  "set": "lea", // The unique set code. E.g. lea = Limited Edition Alpha
  "number": "54", // The collectors number in the set
  "id": "0df55e3f-14de-46ef-b6b1-616618724d9e" // The id from the Scryfall API
}
```

or if it cannot detect any card it will output
``` json
{
  "error": "COULD_NOT_DETECT_ANY_CARDS"
}
```

There are two tools:
* `phash-generator.py`: Generates phashes from a set of card scans/pictures. This is needed for the dector to work.
* `detector.py`: Detects cards in an image (this is your end goal).


## Installation

```
pip install -r requirements.txt
```


## Usage

### Detecting cards in an image

**Note**: you first need to generate phashes of pictures/scans of the cards you want to detect. Read the next usage sections first.

`usage: detector.py [--help] --phash PHASH [--continuous] [--input_path PATH]`

You will either need to use `--continuous` or `--input_path`

- `--continuous`: The program will keep running and you will need to pass paths to file throught stdin
- `--input_path PATH`: The program will run only on those files in that folder

**Example 1:**

In a long-running setup you will most likely not want to keep starting the detector, but keep it running and send paths to it over stdin when there is a card you need to detect.

```
$ echo "test/limited_edition_alpha/3_6f9ea46a-411f-40ce-a873-a905180093f4_Balance.jpg" | ./detector.py --phash test/lea_phashes.dat --continuous
{"name": "Balance", "set": "lea", "number": "3", "id": "6f9ea46a-411f-40ce-a873-a905180093f4", "recognition_score": 1.066549935262097}
```

**Example 2:**

This example can run out of the box, if you have generated the `.dat` file first running the above.

```
$ ./detector.py --phash test/lea_phashes.dat --input_path test/test-pictures
ready
{"name": "Counterspell", "set": "lea", "id": "0df55e3f-14de-46ef-b6b1-616618724d9e", "number": "54"}
{"name": "Instill Energy", "set": "lea", "id": "5bd38716-874c-4e3c-a315-837839a6258c", "number": "202"}
{"name": "Island", "set": "lea", "id": "90a57c0e-fa61-45ef-955d-d296403967d5", "number": "288"}
...
```

### Generating phashes for one set

You need to generate a `.dat` file with phashes of pictures before you can detect cards.

It is important that the files are structured in a correct way for this to work correctly. You can see `/test/limited_edition_alpha` to see how files are correctly named.
Generally they should be named `COLLECTOR-NUMBER_SCRYFALL-ID_NAME.jpg`. The underscores are important. An example for `Black Lotus` from `Alpha` is `232_b0faa7f2-b547-42c4-a810-839da50dadfe_Black Lotus.jpg`. This ensures that the returned JSON will look sound.

`usage: phash-generator.py [-h] --set SET --folder FOLDER --output OUTPUT [--append]`

This example can run out of the box, with the alpha cards provided in this repo.

```
$ ./phash-generator.py --set lea --folder test/lea --output lea_phashes.dat
```

### Generating phashes for all sets

You can generate a phash file which contains all of your cards if you use `--sets-in-subfolders`.

To do this, you need to have sorted your card scans into subfolders like this:

```
/sets
     /lea
         /232_b0faa7f2-b547-42c4-a810-839da50dadfe_Black Lotus.jpg
         ...
     /ktk
         /98_c57534fb-2591-4003-aeec-6452faa4a759_Arrow Storm.jpg
         ...
     ...
```

Then you can run

```
$ ./phash-generator.py --folder sets --output all_phashes.dat --sets-in-subfolders
[1/869] Generating phash from images from the "pmh2" set
[2/869] Generating phash from images from the "iko" set
...
```


## Speed

### Generating phashes

I've tested locally where I created a database of all images of all ~80k Magic card, and it took roughly 15 minutes to generate an `all_phashes.dat` that was 70mb large

### Detecting cards in an image

With an `all_phashes.dat` of 70mb it still only took ~1 second to find a card in an image when using `--input_path`. So it's pretty fast.

For a long-running or production setup, you will probably want to use `--continuous`.

## Test

```
$ python test/test.py
```

Some test cases to test both `phash-generator.py` and `detector.py`


## Fork

This has been forked from https://github.com/tmikonen/magic_card_detector.

I've changed the way it work quite a bit, but the nitty gritty difficult part of actually gnerates phashes and detecting images was made by the @tmikonen.
