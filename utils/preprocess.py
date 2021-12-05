import numpy as np
import pandas as pd

import gensim.downloader as api
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import word_tokenize

def getGloveCorpus(dims=300):
    # Set path and load corpus
    path = './Datasets/'
    filename = f'glove.6B.{dims}d.txt'
    f = open(path+filename, 'r', encoding='latin2')
    vec_txt = f.read()

    vec_data = {}
    words = vec_txt.split('\n')
    for word in words:
        vec = word.split()
        if len(vec) == dims+1:
            vec_data[vec[0]] = np.array([np.float16(x) for x in vec[1:]])
    vec = pd.DataFrame(vec_data, columns=None).transpose()
    return vec


def getWord2VecCorpus():
    vec = api.load('word2vec-google-news-300')
    return vec


def getTFIDFCorpus(MaxvecLen=None, min_df=0.001):
    path = './Datasets/'
    filename = 'Sarcasm_Headlines_Detection.csv'
    df = pd.read_csv(
        path+filename).dropna().reset_index(drop=True)

    tooLong = []
    if MaxvecLen != None:
        for i in range(len(df['headline'])):
            if len(df['headline'][i].split()) > MaxvecLen:
                tooLong.append(i)
        for i in tooLong:
            df = df.drop(i, axis=0).reset_index(drop=True)

    vec = TfidfVectorizer(tokenizer=word_tokenize, min_df=min_df)
    vec.fit(df['headline'])
    return vec
