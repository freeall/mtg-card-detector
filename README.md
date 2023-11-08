# mtg-card-detector

Detects Magic: The Gathering cards in images and returns JSON as
``` json
{
  "name": "Counterspell",
  "set": "lea", // The unique set code. E.g. lea = Limited Edition Alpha
  "number": "54", // The collectors number in the set
  "id": "0df55e3f-14de-46ef-b6b1-616618724d9e" // The id from the Scryfall API
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

### Generating phashes

You need to generate a `.dat` file with phashes of pictures before you can detect cards.

It is important that the files are structured in a correct way for this to work correctly. You can see `/test/limited_edition_alpha` to see how files are correctly named.
Generally they should be named `COLLECTOR-NUMBER_SCRYFALL-ID_NAME.jpg`. The underscores are important. An example for `Black Lotus` from `Alpha` is `232_b0faa7f2-b547-42c4-a810-839da50dadfe_Black Lotus.jpg`. This ensures that the returned JSON will look sound.

`usage: phash-generator.py [-h] -s SET -f FOLDER -o OUTPUT [-a]`

This example can run out of the box, with the alpha cards provided in this repo.

```
$ python phash-generator.py --set lea --folder test/lea --output lea_phashes.dat
```

### Detecting cards in an image

Note that you first need to have a `.dat` file with phashes of pictures/scans of the cards you want to detect.

`usage: detector.py [-h] [--phash PHASH] input_path`

This example can run out of the box, if you have generated the `.dat` file first running the above.

```
$ python detector.py --phash lea_phashes.dat test/test-pictures
{"name": "Counterspell", "set": "lea", "id": "0df55e3f-14de-46ef-b6b1-616618724d9e", "number": "54"}
{"name": "Instill Energy", "set": "lea", "id": "5bd38716-874c-4e3c-a315-837839a6258c", "number": "202"}
{"name": "Island", "set": "lea", "id": "90a57c0e-fa61-45ef-955d-d296403967d5", "number": "288"}
...
```


## Speed

### Generating phashes

I've tested locally where I created a database of all images of all ~80k Magic card, and it took roughly 15 minutes to generate an `all_phashes.dat` that was 70mb large

### Detecting cards in an image

With an `all_phashes.dat` of 70mb it still only took ~1 second to find a card in an image. So it's pretty fast.


## Test

```
$ python test/test.py
```

Some test cases to test both `phash-generator.py` and `detector.py`


## Fork

This has been forked from https://github.com/tmikonen/magic_card_detector.

I've changed the way it work quite a bit, but the nitty gritty difficult part of actually gnerates phashes and detecting images was made by the @tmikonen.
