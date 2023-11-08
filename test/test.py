import subprocess
import os
import json

# First test the hash generator
print('ℹ️  Generating hashes from pictures in the "./lea" folder')
result = subprocess.run(
  [
    'python', '../generate-hash.py',
    '--set', 'lea',
    '--folder', 'lea/',
    '--output', 'lea_hashes.dat'
  ],
  capture_output = True,
  text = True
)
assert result.stdout == ''
assert result.stderr == ''
print('✅ Hashes were generated and stored in lea_hashes.dat')

# Then use the fil{"name": "Counterspell", "set": "lea", "id": "0df55e3f-14de-46ef-b6b1-616618724d9e", "number": "54"}e from the hash generator to test the mtg-card-identifier
print('ℹ️  Detecting images from "./test-pictures" and check with the previously generated hashes, "lea_hashes.dat"')
result = subprocess.run(
  [
    'python', '../magic_card_detector.py',
    '--phash', 'lea_hashes.dat',
    'test-pictures'
  ],
  capture_output = True,
  text = True
)
cards_found = result.stdout.strip().split('\n')
assert 52 == len(cards_found)

counterspell = json.loads(cards_found[0])
assert 'Counterspell' == counterspell['name']
assert 'lea' == counterspell['set']
assert '54' == counterspell['number']
assert '0df55e3f-14de-46ef-b6b1-616618724d9e' == counterspell['id']
print('✅ All images were detected as suspected')

# Clean up
print('ℹ️  Cleaning up')
os.remove('lea_hashes.dat')

print('✅ All tests were ok!')
