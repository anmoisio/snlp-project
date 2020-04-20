import config
import utils
from train import train_model
from evaluate import *
from gensim.models.keyedvectors import FastTextKeyedVectors

def main():
    # train model
    #model = train_model()

    # alternatively load a pretrained model
    print("Using the "+config.model_type+" model:", config.model_filename)
    print("Loading "+config.model_type+" model...")
    if config.w2vec:
        model = KeyedVectors.load_word2vec_format(config.model_file, binary=True)
    else:
        model = FastTextKeyedVectors.load(config.model_file)
    print(config.model_type+" model loaded.")
    
    # evaluate
    result_string = intrusion(model)
    result_string += analogy(model)
    result_string += nearest_neighbours(model)
    
    with open(config.result_file, 'w', encoding='utf-8') as f:
        f.write(result_string)

if __name__ == "__main__":
    main()