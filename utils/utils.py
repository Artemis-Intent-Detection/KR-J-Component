import numpy as np
from nltk.tokenize import word_tokenize


def getWord2Vec(word, vec, dims=300):
    vc = np.zeros(dims)
    try:
        vc = np.array(vec[word]).astype(np.float16)
    except:
        vc = np.zeros(dims).astype(np.float16)
    return vc


def getGloveVec(word, vec, dims=300):
    vc = np.zeros(dims)
    try:
        vc = np.array(vec.loc[word])
    except:
        vc = np.zeros(dims)
    return vc


def getTFIDFVec(word, vec):
    vec = np.asarray(vec.transform([word]).toarray()).astype(np.float16)
    return np.reshape(vec, (vec.shape[1],))


def getDocVec(sentence, dims, vec, preprocess, vectype, MaxvecLen=None):
    tokens = word_tokenize(sentence)
    vecs = np.zeros(dims)
    if vectype == 'sum':
        if preprocess == 'glove':
            for word in tokens:
                vecs += getGloveVec(word, vec)
        elif preprocess == 'word2vec':
            for word in tokens:
                vecs += getWord2Vec(word, vec)
        elif preprocess == 'tfidf':
            vecs = getTFIDFVec(sentence, vec)

    elif vectype == 'vector':
        vecs = []
        padWord = None
        if preprocess == 'glove':
            padWord = getGloveVec('', vec)
            for word in tokens:
                vecs.append(getGloveVec(word, vec))
        elif preprocess == 'word2vec':
            padWord = getWord2Vec('', vec)
            for word in tokens:
                vecs.append(getWord2Vec(word, vec))
        elif preprocess == 'tfidf':
            padWord = getTFIDFVec('', vec)
            for word in tokens:
                vecs.append(getTFIDFVec(word, vec))
        # Padding
        while len(vecs) < MaxvecLen:
            vecs.append(padWord)
    return vecs


def getVecForm(X, Y, dims, vec, preprocess='glove', vectype='sum', reshaping=None, MaxvecLen=None):
    '''
    X - Array of sentences.
    Y - Array of output class (numeric).
    dims - Vector dimensions (needed to pad correctly).
    preprocess - Preprocessing method used. Accepts - "glove", "word2vec", "tfidf".
    vectype - Type of vector needed. Accepts - "sum", "vector".
    MaxvecLen - Suggests maximum vector size. Useful only with vectype-"vector".
    '''
    vecList = []
    for i in X:
        vecList.append(getDocVec(i, dims, vec, preprocess, vectype, MaxvecLen))
    X = np.asarray(vecList).astype(np.float16)
    Y = np.asarray(Y).astype(np.float16)
    if reshaping != None:
        X = np.reshape(X, reshaping)
    return X, Y
