#Import NLTK library for the input text tokenization and
#Libvoikko library (more information about Libvoikko: help(Voikko))
#for lemmatizing.
import libvoikko
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
import time
from tqdm import tqdm

#Define a Voikko class for Finnish
analyzer = libvoikko.Voikko(u"fi")

#Open the text file
filename2read = "a-iltalehti-2020-02-28.txt"
print("Reading the input text file...")
file2read = open(filename2read, "r", encoding="utf-8")
text = file2read.read()
file2read.close()

#Print text
#print("TEXT BEFORE NORMALISATION")
#print(text)

#Remove numbers
#text = ''.join(c for c in text if not c.isdigit())

#Tokenize & remove punctuation
#tokenizer = RegexpTokenizer(r'\w+')
#text = tokenizer.tokenize(text)

#Tokenize
tokenizer = word_tokenize
print("Tokenizing...")
text = word_tokenize(text)
text_length = len(text)

#Lemmatize tokens
pbar = tqdm(total=text_length, ascii=True, desc = 'Lemmatizing...',
            position=0,unit='keys', unit_scale=True)
for idx, word in enumerate(text):
    #Check if word is found from dictionary
    if analyzer.analyze(word):
        #Lemmatize the word. analyze() function returns
        #various info for the word
        text[idx] = analyzer.analyze(word)[0]['BASEFORM']
    pbar.update(1)


#Print normalized text
#print("TEXT AFTER NORMALISATION")    
#print(' '.join(text))

#Write tokenized text to a new text file
filename2write = "a-iltalehti-2020-02-28_normalized.txt"
file2write = open(filename2write,'w', encoding="utf-8")
print("\nWriting the normalized text to a txt file...")
file2write.write(' '.join(text))
file2write.close()