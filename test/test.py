import subprocess
import os
import json

# First test the phash generator
print('ℹ️  Generating phashes from pictures in the "./limited_edition_alpha" folder')
result = subprocess.run(
  [
    'python', '../phash-generator.py',
    '--set', 'lea',
    '--folder', 'limited_edition_alpha',
    '--output', 'test.py-generated.dat'
  ],
  capture_output = True,
  text = True
)
assert result.stdout == ''
assert result.stderr == ''
print('✅ Phashes were generated and stored in test.py-generated.dat')

# Then use the file from the phash generator to test the mtg-card-identifier
print('ℹ️  Detecting images from "./test-pictures" and check with the previously generated phashes, "test.py-generated.dat"')
result = subprocess.run(
  [
    'python', '../detector.py',
    '--phash', 'test.py-generated.dat',
    '--input_path', 'test-pictures'
  ],
  capture_output = True,
  text = True
)
cards_found = result.stdout.strip().split('\n')
assert 55 == len(cards_found)

counterspell = json.loads(cards_found[0])
assert 'Counterspell' == counterspell['name']
assert 'lea' == counterspell['set']
assert '54' == counterspell['number']
assert '0df55e3f-14de-46ef-b6b1-616618724d9e' == counterspell['id']
print('✅ All images were detected as suspected')

# Clean up
print('ℹ️  Cleaning up')
os.remove('test.py-generated.dat')

print('✅ All tests were ok!')
