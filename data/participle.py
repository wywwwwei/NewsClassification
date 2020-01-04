# encoding:utf-8
import os
import pandas as pd
import jieba
import jieba.analyse
from gensim import corpora, models, similarities



def preprocess(filepath, stopwords):
    """
    Preprocessing: word segmentation to remove stop words
    """
    data = pd.read_csv(filepath, header=None, names=[
                       'content'], encoding="utf-8")
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
            print("error:", line)
            continue

    return article




def load_txt_data(attribute, file_path="./result", stopwords_path="./data/stop_words.txt", label_exist=True):
    """
    Read the text file, divide the words, and store it in a list
    """
    allfile = list()
    articles = []
    label = []
    if label_exist:
        for dir in os.listdir(file_path):
            dir_path = os.path.join(file_path, dir)
            if os.path.isdir(dir_path):
                for file in os.listdir(dir_path):
                    allfile.append(os.path.join(dir_path, file))
                    label.append(attribute.class_list.index(dir))
    else:
        for file in os.listdir(file_path):
            allfile.append(os.path.join(dir_path, file))

    stopwords = pd.read_csv(stopwords_path, sep="\t", index_col=False, quoting=3, names=[
                            'stopword'], encoding="utf-8")
    stopwords = stopwords['stopword'].values

    for i in range(len(allfile)):
        articles.append(preprocess(allfile[i], stopwords))
    return articles, label




def load_vocabulary(attribute, file_path="./result", stopwords_path="./data/stop_words.txt"):
    """
    Create dictionary:word ->id mapping
    """
    articles, label = load_txt_data(
        attribute=attribute, file_path=file_path, stopwords_path=stopwords_path)

    dictionary = corpora.Dictionary(articles)
    print(dictionary.token2id)
    dictionary.add_documents([["<UNK>","<PAD>"]])
    return dictionary



def load_dataset(articles, label, attribute, hyperparameter, vocabulary):
    """
    Convert the data into the form of idx,label, as input to the embedding layer
    """
    pad_size = hyperparameter.pad_size
    contents = []
    for i in range(len(articles)):
        seq_len = len(articles[i])
        if seq_len < pad_size:
            articles[i].extend(["<PAD>"]*(pad_size - seq_len))
        else:
            articles[i] = articles[i][:pad_size]
            seq_len = pad_size
        words_to_id = []
        for word in articles[i]:
            words_to_id.append(vocabulary.get(word, vocabulary.get("<UNK>")))
        contents.append((words_to_id, label[i], seq_len))
    return contents

def load_vocab_file(attribute):
    """
    Load the dictionary file through gensim
    """
    vocab_dir = "./data/vocab.dict"
    if os.path.exists(vocab_dir):
        word_to_id = corpora.Dictionary.load(vocab_dir)
    else:
        word_to_id = load_vocabulary(attribute=attribute)
        word_to_id.save(vocab_dir)
    return word_to_id.token2id