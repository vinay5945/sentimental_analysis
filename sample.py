import pickle
import os
from pathlib import Path

path = os.path.join('art','sam.txt')

os.makedirs(path, exist_ok=True)

with open(path, "wb") as file_obj:
    pickle.dump("str", file_obj)

print("hello")