import numpy as np
import re
import itertools
from collections import Counter
from sklearn.utils import shuffle
import jieba
import pandas as pd
from gensim.models import KeyedVectors, Word2Vec
from keras.preprocessing import sequence

def stopwordslist():
    stopwords = [line.strip() for line in open('./data/stop_ch.txt', 'r', encoding='utf-8').readlines()]
    return stopwords


def movestopwords(sentence):
    stopwords = stopwordslist()
    outstr = ''
    for word in sentence:
        if word not in stopwords:
            if word != '\t' and '\n':
                outstr += word
                # outstr += " "
    return outstr


def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels():
    """
    Loads polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    # positive_examples = list(open("./data/rt-polarity.pos", "r", encoding='latin-1').readlines())
    # positive_examples = [s.strip() for s in positive_examples]
    # negative_examples = list(open("./data/rt-polarity.neg", "r", encoding='latin-1').readlines())
    # negative_examples = [s.strip() for s in negative_examples]
    # # Split by words
    # x_text = positive_examples + negative_examples
    # x_text = [clean_str(sent) for sent in x_text]
    # x_text = [s.split(" ") for s in x_text]
    # # Generate labels
    # positive_labels = [[0, 1] for _ in positive_examples]
    # negative_labels = [[1, 0] for _ in negative_examples]
    # y = np.concatenate([positive_labels, negative_labels], 0)
    df = pd.read_csv('./data/spamContent.csv')
    df2 = pd.read_csv('./data/senti_100k3.csv')
    df = df.append(df2)
    df = shuffle(df)
    review_part = df.review
    sentence = [[item for item in jieba.cut(movestopwords(s), cut_all=False)] for s in review_part]
    for item in sentence:
        while True:
            if ' ' in item:
                item.remove(' ')
            else:
                break
    y_label = df.label
    y = []
    for i in y_label:
        if i == '0' or i == 0.0:
            y.append([1, 0])
        else:
            y.append([0, 1])
    x_test = pd.read_csv('./data/eval_dataset.csv')
    x_test_review = x_test.review
    x_test_sentence = [[item for item in jieba.cut(movestopwords(s), cut_all=False)] for s in x_test_review]
    for item in x_test_sentence:
        while True:
            if ' ' in item:
                item.remove(' ')
            else:
                break
    y_test_label = x_test.label
    y_test = []
    for i in y_test_label:
        if i == '0' or i == 0.0:
            y_test.append(0)
        else:
            y_test.append(1)
    return [sentence, np.array(y),x_test_sentence,y_test]


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def tokenizer(texts, word_index):
    data = []
    for sentence in texts:
        new_txt = []
        for word in sentence:
            try:
                new_txt.append(word_index[word])  # 把句子中的 词语转化为index
            except:
                new_txt.append(0)
        data.append(new_txt)

    texts = sequence.pad_sequences(data, maxlen=92)  # 使用kears的内置函数padding对齐句子,好处是输出numpy数组，不用自己转化了
    return texts


def load_data():
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels,x_test,y_test = load_data_and_labels()

    Word2VecModel = KeyedVectors.load_word2vec_format('./data/weibo_data.bin', binary=True)
    vocab_list = [word for word, Vocab in Word2VecModel.wv.vocab.items()]
    word_index = {" ": 0}  # 初始化 `[word : token]` ，后期 tokenize 语料库就是用该词典。
    word_vector = {}  # 初始化`[word : vector]`字典
    # 初始化存储所有向量的大矩阵，留意其中多一位（首行），词向量全为 0，用于 padding补零。
    # 行数 为 所有单词数+1 比如 10000+1 ； 列数为 词向量“维度”比如100。
    embeddings_matrix = np.zeros((len(vocab_list) + 1, Word2VecModel.vector_size))
    ## 填充 上述 的字典 和 大矩阵
    for i in range(len(vocab_list)):
        word = vocab_list[i]  # 每个词语
        word_index[word] = i + 1  # 词语：序号
        word_vector[word] = Word2VecModel.wv[word]  # 词语：词向量
        embeddings_matrix[i + 1] = Word2VecModel.wv[word]

    sentences_padded = tokenizer(sentences, word_index)
    x_test = tokenizer(x_test,word_index)
    # vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    # x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [sentences_padded, labels, embeddings_matrix,x_test,y_test]
