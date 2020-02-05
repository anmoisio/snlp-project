import os
import glob

# change encoding from utf-8 to cp1252 (aka 'windows 1252')
# try this if you have problems with the letters дце
# if you get an UnicodeEncodeError, the file will be incomplete
# (so best to take a copy of the .txt files somewhere before running this)

EVAL_DATA_DIR = os.path.join("data", "eval", "FinSemEvl", "FinSemEvl", "intrusion")

for eval_file in glob.glob(os.path.join(EVAL_DATA_DIR, "*.txt")):
    with open(eval_file, "r", encoding="utf-8") as f:
        content = f.read()
    with open(eval_file, "w", encoding="cp1252") as f:
        try:
            f.write(content)
        except UnicodeEncodeError as err:
            print(err, eval_file)