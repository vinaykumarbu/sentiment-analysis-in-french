import pandas as pd
import numpy as np
import glob
import re
#from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
from collections import Counter

fields=['word','polarity']
lex=pd.read_csv('Python programs/ABI/FEEL.csv',delimiter=";",usecols=fields)

train, test = train_test_split(lex,test_size = 0.1)
train_pos = train[ train['polarity'] == 'Positive']
train_pos = train_pos['word']
train_neg = train[ train['polarity'] == 'Negative']
train_neg = train_neg['word']

stopset = set(stopwords.words('french'))

path = 'Python programs/ABI/ftragedy/*.txt'
files=glob.glob(path)
def read_raw_file(path):         
    for file in files:     
        f = open(path,"r") 
        raw = f.read().decode('utf8')
        f.close() 
    return raw
                
def get_tokens(raw,encoding='utf8'): 
    tokens = nltk.word_tokenize(raw) 
    return tokens 
    
def get_nltk_text(raw,encoding='utf8'):
    no_commas = re.sub(r'[.|,|\']',' ', raw) 
    tokens = nltk.word_tokenize(no_commas) 
    text=nltk.Text(tokens,encoding) 
    return text
    
def filter_stopwords(text,stopword_list):
    words=[w.lower() for w in text] 
    filtered_words = [] 
    for word in words: 
        if word not in stopword_list and word.isalpha() and len(word) > 1: 
            filtered_words.append(word) 
    filtered_words.sort() 
    return filtered_words
    
def stem_words(words):
    stemmed_words = [] 
    stemmer = FrenchStemmer() 
    for word in words:
        stemmed_word=stemmer.stem(word) 
        stemmed_words.append(stemmed_word) 
    stemmed_words.sort()  
    return stemmed_words 