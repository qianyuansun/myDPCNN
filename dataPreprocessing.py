import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from string import punctuation
import re
from nltk import FreqDist
from nltk.corpus import stopwords
import bcolz
import numpy as np
import pickle
import csv
from nltk.stem import SnowballStemmer
snow = SnowballStemmer('english')

import warnings
warnings.filterwarnings('ignore')

import nltk
#nltk.download('stopwords')
stop = set(stopwords.words('english'))
excluding = ['against','not','don', "don't",'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
             'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
             'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",'shouldn', "shouldn't", 'wasn',
             "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
stop = [words for words in stop if words not in excluding]


def stripPunc(sent):
    newSent = [w for w in sent if w not in punctuation]
    return newSent


def normalizeString(s):
    s = re.sub(r"\\n", '\n', s)
    s = re.sub(r"\n", ' ', s)
    s = re.sub(r'\\"', '\"', s)
    s = re.sub(r"[^A-Za-z0-9(),!?\'\":\-]", " ", s)
    s = re.sub(r"\'s", " \'s", s)
    s = re.sub(r"\'ve", " \'ve", s)
    s = re.sub(r"n\'t", " n\'t", s)
    s = re.sub(r"\'re", " \'re", s)
    s = re.sub(r"\'d", " \'d", s)
    s = re.sub(r"\'ll", " \'ll", s)
    s = re.sub(r",", " , ", s)
    s = re.sub(r"!", " ! ", s)
    s = re.sub(r"-", " - ", s)
    s = re.sub(r":", " : ", s)
    s = re.sub(r"\(", " ( ", s)
    s = re.sub(r"\)", " ) ", s)
    s = re.sub(r"\?", " ? ", s)
    s = re.sub(r"<br />", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s

def savePreDic():
    words = []
    idx = 0
    word2idx = {}
    glove_path = 'E:/yuqian/pro/glove.6B'
    vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.100.dat', mode='w')

    with open(f'{glove_path}/glove.6B.100d.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    vectors = bcolz.carray(vectors[1:].reshape((400000, 100)), rootdir=f'{glove_path}/6B.100.dat', mode='w')
    vectors.flush()
    pickle.dump(words, open(f'{glove_path}/6B.100_words.pkl', 'wb'))
    pickle.dump(word2idx, open(f'{glove_path}/6B.100_idx.pkl', 'wb'))

def preprocessing(path, outpath):
    data_test = pd.read_csv(path, header=None)
    df_test = pd.DataFrame(data_test)
    df_test["labels"] = data_test[0]
    df_test["tokens"] = data_test[1]  #yelp/ BBC
    #df_test["tokens"] = df_test.apply(lambda row: row[1] + " " + row[2], axis=1)  #AG/ db
    #df_test["tokens"] = df_test.apply(lambda row: str(row[1]) + " " + str(row[2])+ " " + str(row[3]), axis=1)  #yahoo
    df_test["tokens"] = df_test.apply(lambda row: normalizeString(row["tokens"]), axis=1)
    df_test["tokens"] = df_test.apply(lambda row: word_tokenize(row["tokens"]), axis=1)
    df_test = df_test.drop([0,1], axis=1)

    lemmatizer = WordNetLemmatizer()
    #words_list = []
    indexList = []
    for i, row in df_test.iterrows():

        #word_lem = stripPunc(row["tokens"])
        #word_lem = [w.lower() for w in word_lem if w.lower() not in stop]

        word_lem = [lemmatizer.lemmatize(w.lower()) for w in row["tokens"]]

        if len(word_lem) > 2:
            df_test = df_test.set_value(i, "tokens", word_lem)
            #words_list += word_lem
        else:
            indexList.append(i)

    df_test = df_test.drop(df_test.index[indexList])
    df_test = df_test.reset_index(drop=True)
    df_test.to_csv(outpath)

'''
    fdist1 = FreqDist(words_list)
    voc = fdist1.most_common(30001)
    voc = dict(voc)
    voc.pop(None, None)

    w = csv.writer(open("./dictionary/bbc_dic.csv", "w", newline=''))
    wordId = 1
    for key, value in voc.items():
        w.writerow([key, wordId])
        wordId += 1
'''






#def wordToIndex():

'''
if __name__ == '__main__':
    data_test = pd.read_csv('ag_test_pre.csv', header=None)
    data_test[2] = data_test[2].apply(lambda row: row .strip("['").strip("']"))
    data_test[2] = data_test[2].apply(lambda row: row.replace("', '", ","))
    word2idx = dict()
    data_array = list()
    vo_count = 1

    for i in data_test[2]:
        row_array = list()
        for word in i.split(","):
            if word in word2idx.keys():
                row_array.append(word2idx[word])
            else:
                word2idx[word] = float(vo_count)
                row_array.append(word2idx[word])
                vo_count += 1
        data_array.append(row_array)
        #return dict
    print(vo_count)
    print(len(data_array))
'''

#preprocessing('E:/yuqian/pro/dataset/yelp_review_polarity_csv/test.csv', './data/test/yelp_test_pre_4.csv')
#preprocessing('E:/yuqian/pro/dataset/yahoo_answers_csv/train.csv', './data/train/yahoo_train_pre.csv')
#preprocessing('E:/yuqian/pro/dataset/dbpedia_csv/test.csv', './data/test/dbpedia_test_pre.csv')
preprocessing('E:/yuqian/pro/dataset/bbc_csv/bbc_test.csv', './data/test/bbc_test_pre.csv')
#savePreDic()







