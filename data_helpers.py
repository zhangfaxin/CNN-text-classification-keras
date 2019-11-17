import numpy as np
import re
import itertools
from collections import Counter
from sklearn.utils import shuffle
import jieba
import pandas as pd
from gensim.models import KeyedVectors, Word2Vec
from keras.preprocessing import sequence
from pypinyin import lazy_pinyin
import part_of_speech


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


def load_word_data_and_labels():
    """
    Loads polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    df = pd.read_csv('./data/train_data_1.csv')
    review_part = df.review
    sentence = [[item for item in list(movestopwords(s))] for s in review_part]
    # sentence = [[word2pinyin(i) for i in jieba.cut(movestopwords(s),cut_all=False)] for s in review_part]
    # sentence = [part_of_speech.get_sentence(s) for s in review_part]
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
    x_test = pd.read_csv('./data/test_data_1-pianpang.csv')
    x_test_review = x_test.review
    x_test_sentence = [[item for item in list(movestopwords(s))] for s in x_test_review]
    # x_test_sentence = [[word2pinyin(i) for i in jieba.cut(movestopwords(s),cut_all=False)] for s in x_test_review]
    # x_test_sentence = [part_of_speech.get_sentence(s) for s in x_test_review]
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
    return [sentence, np.array(y),x_test_sentence,y_test,x_test_review]


def load_data_and_labels():
    """
    Loads polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    df = pd.read_csv('./data/train_data_1.csv')
    review_part = df.review
    sentence = [[item for item in jieba.cut(movestopwords(s), cut_all=False)] for s in review_part]
    # sentence = [part_of_speech.get_sentence(s) for s in review_part]
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
    x_test = pd.read_csv('./data/test_data_1-pianpang.csv')
    x_test_review = x_test.review
    x_test_sentence = [[item for item in jieba.cut(movestopwords(s), cut_all=False)] for s in x_test_review]
    # x_test_sentence = [part_of_speech.get_sentence(s) for s in x_test_review]
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
    return [sentence, np.array(y),x_test_sentence,y_test,x_test_review]


def word2pinyin(word):
    pinyin = lazy_pinyin(word)
    return "".join(pinyin)


def load_pinyin_data_and_labels():
    """
    Loads polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    df = pd.read_csv('./data/train_data_1.csv')
    review_part = df.review
    sentence = [[word2pinyin(item) for item in list(movestopwords(s))] for s in review_part]
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
    x_test = pd.read_csv('./data/test_data_1-pianpang.csv')
    x_test_review = x_test.review
    x_test_sentence = [[word2pinyin(item) for item in list(movestopwords(s))] for s in x_test_review]
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


def load_shengmu_data_and_labels():
    """
    Loads polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    df = pd.read_csv('./data/train_data_1.csv')
    review_part = df.review
    ylabel = df.label
    sentence = []
    tone_list = []
    y_label = []
    for x,y in zip(review_part,ylabel):
        shengmu,tone = part_of_speech.get_sentence(x)
        if shengmu:
            sentence.append(shengmu)
            tone_list.append(tone)
            if y == '0' or y == 0.0:
                y_label.append([1,0])
            else:
                y_label.append([0,1])
    x_test = pd.read_csv('./data/test_data_1.csv')
    x_test_review = x_test.review
    y_test = x_test.label
    x_test_sentence = []
    x_test_tone = []
    y_test_label = []
    for x,y in zip(x_test_review,y_test):
        shengmu,tone = part_of_speech.get_sentence(x)
        if shengmu:
            x_test_sentence.append(shengmu)
            x_test_tone.append(tone)
            if y == '0' or y == 0.0:
                y_test_label.append(0)
            else:
                y_test_label.append(1)

    return [sentence, tone_list,np.array(y_label),x_test_sentence,x_test_tone,y_test_label,x_test_review]



def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        sentence = [str(i) for i in sentence]
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


def build_input_data(sentences,vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    return x


def tokenizer(texts, word_index,max_length):
    data = []
    for sentence in texts:
        new_txt = []
        for word in sentence:
            try:
                new_txt.append(word_index[word])  # 把句子中的 词语转化为index
            except:
                new_txt.append(0)
        data.append(new_txt)

    texts = sequence.pad_sequences(data, maxlen=max_length)  # 使用kears的内置函数padding对齐句子,好处是输出numpy数组，不用自己转化了
    return texts


def load_data():
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels,x_test,y_test,sentence_raw = load_data_and_labels()
    length = [len(x) for x in sentences]
    max_sentence = max(length)
    print('index:{}'.format(length.index(max_sentence)))
    print('max_sentence:{}'.format(max_sentence))
    Word2VecModel = KeyedVectors.load_word2vec_format('./data/weibo_data.bin', binary=True)
    vocab_list = [word for word, Vocab in Word2VecModel.wv.vocab.items()]
    word_index = {" ": 0}  # 初始化 `[word : token]` ，后期 tokenize 语料库就是用该词典。
    word_vector = {}  # 初始化`[word : vector]`字典
    # 初始化存储所有向量的大矩阵，留意其中多一位（首行），词向量全为 0，用于 padding补零。
    # 行数 为 所有单词数+1 比如 10000+1 ； 列数为 词向量“维度”比如100。
    embeddings_matrix = np.zeros((len(vocab_list) + 1, Word2VecModel.vector_size))
    # 填充 上述 的字典 和 大矩阵
    for i in range(len(vocab_list)):
        word = vocab_list[i]  # 每个词语
        word_index[word] = i + 1  # 词语：序号
        word_vector[word] = Word2VecModel.wv[word]  # 词语：词向量
        embeddings_matrix[i + 1] = Word2VecModel.wv[word]

    sentences_padded = tokenizer(sentences, word_index,max_length=max_sentence)
    x_test = tokenizer(x_test,word_index,max_length=max_sentence)
    # vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    # x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [sentences_padded, labels, embeddings_matrix,x_test,y_test,sentence_raw]


def load_word_data():
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels,x_test,y_test,sentence_raw = load_word_data_and_labels()
    length = [len(x) for x in sentences]
    max_sentence = max(length)
    print('index:{}'.format(length.index(max_sentence)))
    print('max_sentence:{}'.format(max_sentence))
    Word2VecModel = KeyedVectors.load_word2vec_format('./data/word_sentence.bin', binary=True)
    vocab_list = [word for word, Vocab in Word2VecModel.wv.vocab.items()]
    word_index = {" ": 0}  # 初始化 `[word : token]` ，后期 tokenize 语料库就是用该词典。
    word_vector = {}  # 初始化`[word : vector]`字典
    # 初始化存储所有向量的大矩阵，留意其中多一位（首行），词向量全为 0，用于 padding补零。
    # 行数 为 所有单词数+1 比如 10000+1 ； 列数为 词向量“维度”比如100。
    embeddings_matrix = np.zeros((len(vocab_list) + 1, Word2VecModel.vector_size))
    # 填充 上述 的字典 和 大矩阵
    for i in range(len(vocab_list)):
        word = vocab_list[i]  # 每个词语
        word_index[word] = i + 1  # 词语：序号
        word_vector[word] = Word2VecModel.wv[word]  # 词语：词向量
        embeddings_matrix[i + 1] = Word2VecModel.wv[word]

    sentences_padded = tokenizer(sentences, word_index,max_length=max_sentence)
    x_test = tokenizer(x_test,word_index,max_length=max_sentence)
    # vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    # x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [sentences_padded, labels, embeddings_matrix,x_test,y_test,sentence_raw]


def load_pinyin_data():
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels,x_test,y_test = load_pinyin_data_and_labels()
    length = [len(x) for x in sentences]
    max_sentence = max(length)
    Word2VecModel = KeyedVectors.load_word2vec_format('./data/word_pinyin.bin', binary=True)
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

    sentences_padded = tokenizer(sentences, word_index,max_length=max_sentence)
    x_test = tokenizer(x_test,word_index,max_length=max_sentence)
    # vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    # x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [sentences_padded, labels, embeddings_matrix,x_test,y_test]


def load_shengmu_data():
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences,sentence_tone, labels,x_test,x_test_tone,y_test,sentence_raw = load_shengmu_data_and_labels()
    length = [len(x) for x in sentences]
    max_sentence = max(length)
    print('max_sentence:{}'.format(max_sentence))
    Word2VecModel = KeyedVectors.load_word2vec_format('./data/weibo_data_shengmu_pinyin.bin', binary=True)
    vocab_list = [word for word, Vocab in Word2VecModel.wv.vocab.items()]
    word_index = {" ": 0}  # 初始化 `[word : token]` ，后期 tokenize 语料库就是用该词典。
    word_vector = {}  # 初始化`[word : vector]`字典
    # 初始化存储所有向量的大矩阵，留意其中多一位（首行），词向量全为 0，用于 padding补零。
    # 行数 为 所有单词数+1 比如 10000+1 ； 列数为 词向量“维度”比如100。
    embeddings_matrix = np.zeros((len(vocab_list) + 1, Word2VecModel.vector_size))
    # 填充 上述 的字典 和 大矩阵
    for i in range(len(vocab_list)):
        word = vocab_list[i]  # 每个词语
        word_index[word] = i + 1  # 词语：序号
        word_vector[word] = Word2VecModel.wv[word]  # 词语：词向量
        embeddings_matrix[i + 1] = Word2VecModel.wv[word]

    sentence_tone_padded = pad_sentences(sentence_tone)
    vocabulary, vocabulary_inv = build_vocab(sentence_tone_padded)
    sentence_tone = build_input_data(sentence_tone_padded,vocabulary)

    sentences_padded = tokenizer(sentences, word_index,max_length=max_sentence)
    x_test = tokenizer(x_test,word_index,max_length=max_sentence)
    # vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    # x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [sentences_padded, sentence_tone,labels, embeddings_matrix,x_test,x_test_tone,y_test,sentence_raw,vocabulary, vocabulary_inv]