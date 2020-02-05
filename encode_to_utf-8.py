import os
import glob

# reverses encode_to_cp1252.py

EVAL_DATA_DIR = os.path.join("data", "eval", "FinSemEvl", "FinSemEvl", "intrusion")

for eval_file in glob.glob(os.path.join(EVAL_DATA_DIR, "*.txt")):
    with open(eval_file, "r", encoding="cp1252") as f:
        content = f.read()
    with open(eval_file, "w", encoding="utf-8") as f:
        try:
            f.write(content)
        except UnicodeEncodeError as err:
            print(err, eval_file)