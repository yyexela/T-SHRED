from pathlib import Path
import re

top_dir = Path(__file__).parent.parent

pickle_dir = top_dir / 'pickles'

# Find all pickle files
for pickle_file in pickle_dir.glob('*.pkl'):
    # Check if filename matches pattern of having p1 twice
    if re.match(r'.*p1.*p1.*', pickle_file.name):
        # Get everything after first p1
        new_name = re.sub(r'^.*?p1', '', pickle_file.name)
        #print("Renaming", pickle_file.name, "to", new_name)
        # Rename the file
        pickle_file.rename(pickle_dir / new_name)
