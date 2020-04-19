import config
import utils
from train_w2v import train_model
from eval_w2v import *

def main():
    # train model
    # w2v_model = train_model()

    # alternatively load a pretrained model
    print("Using the word2vec model:", config.model_filename)
    print("Loading word2vec model...")
    w2v_model = KeyedVectors.load_word2vec_format(config.model_file, binary=True)
    print("Word2vec model loaded.")

    # evaluate
    result_string = intrusion(w2v_model)
    result_string += analogy(w2v_model)
    result_string += nearest_neighbours(w2v_model)

    with open(config.result_file, 'w', encoding='utf-8') as f:
        f.write(result_string)

if __name__ == "__main__":
    main()