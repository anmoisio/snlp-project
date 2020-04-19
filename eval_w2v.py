from gensim.models import KeyedVectors
import numpy as np
import os
import glob
from random import sample, shuffle, seed
import config
import csv

# change integer to get different random samples
seed(12345)

def intrusion(w2v_model):
    """
    Evaluate word embeddings with the intrusion task.
    """
        
    # dict of task samples
    # {'intrusion/animals.txt': [([cat, dog, basketball], basketball), ([horse, Sweden, cow], Sweden)], etc}
    all_samples = {} 

    for eval_file in glob.glob(os.path.join(config.EVAL_DATA_DIR, "intrusion", "*.txt")): # for all files in the dir
        with open(eval_file, "r", encoding="utf-8") as f:
            # split file into lines and each line into words to create lists nested inside a list
            data = [line for line in f.read().splitlines()]

        samples = []
        for i in range(config.n_samples):
            word_sample = sample(data, 5) # 5 words from this category
            while True:
                # sample one of the other categories to get the intrusion word
                file_intrusion = sample(glob.glob(os.path.join(config.EVAL_DATA_DIR, "intrusion", "*.txt")), 1)
                # make sure it's a different category
                if file_intrusion != eval_file: break 

            with open(file_intrusion[0], "r", encoding="Utf-8") as f:
                intrusion_words = [line for line in f.read().splitlines()]

            intrusion = sample(intrusion_words, 1)[0] # one word from the intrusion category
            word_sample.append(intrusion)
            shuffle(word_sample)

            samples.append((word_sample, intrusion))

        all_samples[eval_file] = samples

    correct_total = 0
    incorrect_total = 0
    OOV_line_total = 0
    n_total = 0
    score_strings = ""
    for category, word_samples in all_samples.items():
        score_string = "Intrusion task using the data in:" + category
        print(score_string)
        score_string += "\n"

        correct = 0
        incorrect = 0
        OOV_line = 0
        for word_sample in word_samples:
            score_string += str(word_sample) + "\n"
            # print(word_sample)
            try:
                result = w2v_model.doesnt_match(word_sample[0])
                # print("output: ", result)
                score_string += "output: " + result + "\n"
                if result == word_sample[1]:
                    correct += 1
                else:
                    incorrect += 1
            except ValueError as err:
                OOV_line += 1
                # print(err)

        n_total += len(word_samples)
        correct_total += correct
        incorrect_total += incorrect
        OOV_line_total += OOV_line

        bottom_line = ""
        try:
            bottom_line = "N of tasks: {n}, correct: {correct} ({per:.2f}%), tasks with OOV: {oov}\n".format( \
                n=len(word_samples), correct=correct, per=correct*100/(correct+incorrect), oov=OOV_line)
        except ZeroDivisionError:
            bottom_line = " !division by zero! N of tasks: {n}, correct: {correct}, tasks with OOV: {oov}\n".format( \
                n=len(data), correct=correct, oov=OOV_line)
        print(bottom_line)
        score_string += bottom_line + "\n"

        score_strings += score_string + "\n"

    tot_string = "Total scores in intrusion task:\n"
    try:
        tot_string += "N of tasks: {n}, correct: {correct} ({per:.2f}%), tasks with OOV: {oov}\n".format( \
            n=n_total, correct=correct_total, per=correct_total*100/(correct_total+incorrect_total), oov=OOV_line_total)
    except ZeroDivisionError:
        tot_string += " !division by zero!N of tasks: {n}, correct: {correct}, tasks with OOV: {oov}\n".format( \
            n=n_total, correct=correct_total, oov=OOV_line_total)
    print(tot_string)

    return score_strings + tot_string + "\n"
    
def analogy(w2v_model):
    """
    Evaluate word embeddings with the analogy task.
    """
    correct_total = 0
    incorrect_total = 0
    OOV_line_total = 0 # out of vocabulary
    n_lines = 0
    result_strings = ""
    for eval_file in glob.glob(os.path.join(config.EVAL_DATA_DIR, "analogy", "*.txt")): # for each file in dir
        result_string = "Analogy task using the data in:" + eval_file
        with open(eval_file, "r", encoding="utf-8") as f:
            # split file into lines and each line into words to create list nested inside a list
            data = [line.split() for line in f.read().splitlines()]

        correct = 0
        incorrect = 0
        OOV_line = 0 
        for line in data:
            try:
                # the analogy task:
                # line[0] = line[1] + line[2] - line[3]
                # e.g., leveä = kapea + pitkä - lyhyt
                # or    äiti = isä + tyttö - poika
                # or    bangkok = thaimaa + tallinna - viro
                result = w2v_model.most_similar(positive=[line[1], line[2]], negative=[line[3]], topn=1)
                if result[0][0] == line[0]:
                    correct += 1
                else:
                    incorrect += 1
                # print(line, result)
            except KeyError as err:
                OOV_line += 1
                # print(err)

        try:
            result_string += "\nN of lines: {n}, correct: {correct} ({per:.2f}%), lines with OOV: {oov}\n".format( \
                n=len(data), correct=correct, per=correct*100/(correct+incorrect), oov=OOV_line)
        except ZeroDivisionError:
            result_string += "\n!division by zero! N of lines: {n}, correct: {correct}, lines with OOV: {oov}\n".format( \
                n=len(data), correct=correct, oov=OOV_line)
        print(result_string)
        
        correct_total += correct
        incorrect_total += incorrect
        OOV_line_total += OOV_line
        n_lines += len(data)

        result_strings += result_string + "\n"

    bottom_line = "Total scores in analogy task:\n"
    bottom_line += "N of lines: {n}, correct: {correct} ({per:.2f}%), lines with OOV: {oov} ({oovper:.2f}%)\n".format( \
        n=n_lines, correct=correct_total, per=correct_total*100/(correct_total+incorrect_total), \
            oov=OOV_line_total, oovper=OOV_line_total*100/n_lines)
    print(bottom_line)

    return result_strings + bottom_line + "\n"

def nearest_neighbours(w2v_model):
    """
    Evaluate word embeddings by printing the nearest neighbours of words.
    """
    wordlist_file = os.path.join("data", "eval", "nearest_neighbor_wordlist.csv")

    with open(wordlist_file, 'r', encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        finnish_words = [row[0] for row in csv_reader]

    results = "Nearest neighbours using the data in:" + wordlist_file + "\n"
    print(results)
    OOV_lines = 0
    for word in finnish_words:
        try:
            neighbours = "word:", word, "nearest neighbour:", w2v_model.most_similar(word)
            results += str(neighbours) + "\n"
            # print(neighbours)
        except KeyError as err:
            OOV_lines += 1
            # print(err)

    bottom_line = "lines with OOV:" + str(OOV_lines)
    print(bottom_line)
    results += bottom_line

    return results