# encoding:utf-8
import os
import pandas as pd
import jieba
import jieba.analyse
from gensim import corpora,models,similarities

# Preprocessing: word segmentation to remove stop words
def preprocess(filepath,stopwords):
    data = pd.read_csv(filepath,header=None,names = ['content'],encoding="utf-8")
    data = data.dropna()
    datalist = data['content'].values.tolist()
    datalist = datalist[1:len(datalist)-1]

    article = []
    for line in datalist:
        try:
            segs = jieba.lcut(line)
            segs = filter(lambda x: len(x) > 1, segs)
            segs = filter(lambda x: x not in stopwords, segs)
            article.extend(segs)
        except Exception as e:
            print("error:",line)
            continue
    
    return article

# Create dictionary:word ->id mapping
def load_vocabulary(file_path = "./result/txt",stopwords_path = "./data/stop_words.txt"):
    allfile = list()
    articles = []
    for file in os.listdir(file_path):
        allfile.append(os.path.join(file_path, file))

    stopwords = pd.read_csv(stopwords_path, sep="\t", index_col=False, quoting=3, names=[
                            'stopword'], encoding="utf-8")
    stopwords = stopwords['stopword'].values

    for i in range(len(allfile)):
        articles.append(preprocess(allfile[i],stopwords))

    dictionary = corpora.Dictionary(articles)
    dictionary.update({"UNK": len(dictionary), "PAD": len(dictionary) + 1})
    return dictionary
