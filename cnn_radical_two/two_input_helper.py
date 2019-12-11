import numpy as np
import re
import itertools
from collections import Counter
from sklearn.utils import shuffle
import jieba
import pandas as pd
from gensim.models import KeyedVectors, Word2Vec
from keras.preprocessing import sequence
import pkg_resources
from pypinyin import pinyin, lazy_pinyin, Style


def load_pianpang_from_file():
    _dict = {}
    # file = pd.read_csv('../review/pianpang.csv',encoding='utf-8')
    with open('./cnn_radical_two/pianpang.txt', 'r', encoding='utf-8') as dict_file:
        for line in dict_file:
            (key, value) = line.strip().split(',')
            _dict[key] = value

    return _dict


def load_sensitive_from_file():
    _dict = {}
    # file = pd.read_csv('../review/pianpang.csv',encoding='utf-8')
    with open('./cnn_radical_two/dict_file.txt', 'r', encoding='utf-8') as dict_file:
        for line in dict_file:
            (key, value) = line.strip().split(',')
            _dict[key] = value

    return _dict


def load_bushou_from_file():
    _dict = {}
    with open('./bushou.txt', 'r', encoding='utf-8') as dict_file:
        for line in dict_file:
            (key, value) = line.strip().split(',')
            _dict[key] = value

    return _dict


def load_only_bushou_from_file():
    _dict = []
    with open('./cnn_radical_two/only_bushou.txt', 'r', encoding='utf-8') as dict_file:
        for line in dict_file:
            _dict.append(line.strip('\n'))
    return _dict


dict_file = load_pianpang_from_file()

sensitive = load_sensitive_from_file()

only_bushou_list = load_only_bushou_from_file()


def get_word_pianpang(word):
    if word in dict_file:
        return word2pinyin(dict_file[word])
    else:
        return word


def get_pianpang(word):
    if word in dict_file:
        return dict_file[word]
    else:
        return word


def get_word_bushou(word):
    dict = load_bushou_from_file()
    if word in dict:
        return dict[word]
    else:
        return word


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


def word_handle(sentence):
    word_list = list(sentence)
    new_list = []
    for i in word_list:
        if i not in only_bushou_list:
            new_list.append(get_pianpang(i))
    sentence_new = ''.join(new_list)
    for d, x in sensitive.items():
        if x in sentence_new:
            sentence_new = sentence_new.replace(x, d)
    return sentence_new


def load_data_and_labels():
    """
    Loads polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    df = pd.read_csv('./data/train_data_1.csv')
    review_part = df.review
    sentence = [[word2pinyin(item) for item in list(word_handle(movestopwords(s)))] for s in review_part]
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
    # x_test_sentence = [[get_word_pianpang(item) for item in list(movestopwords(s))] for s in x_test_review]
    # # x_test_sentence = [part_of_speech.get_sentence(s) for s in x_test_review]
    # for item in x_test_sentence:
    #     while True:
    #         if ' ' in item:
    #             item.remove(' ')
    #         else:
    #             break
    # y_test_label = x_test.label
    # y_test = []
    # for i in y_test_label:
    #     if i == '0' or i == 0.0:
    #         y_test.append(0)
    #     else:
    #         y_test.append(1)
    return [sentence, np.array(y), x_test_review]


def load_eval_data_and_labels():
    """
    Loads polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    x_test = pd.read_csv('./data/test_data_1-pianpang.csv')
    x_test_review = x_test.review
    x_test_sentence = [[word2pinyin(item) for item in word_handle(list(movestopwords(s)))] for s in x_test_review]
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
    return [x_test_sentence, np.array(y_test), x_test_review]


def word2pinyin(word):
    pinyin = lazy_pinyin(word)
    return "".join(pinyin)


def load_pinyin_data_and_labels():
    """
    Loads polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    df = pd.read_csv('../data/train_data_1.csv')
    review_part = df.review
    sentence = [[get_word_bushou(item) for item in list(movestopwords(s))] for s in review_part]
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
    x_test = pd.read_csv('../data/test_data_1-pianpang.csv')
    x_test_review = x_test.review
    x_test_sentence = [[get_word_bushou(item) for item in list(movestopwords(s))] for s in x_test_review]
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
    return [sentence, np.array(y), x_test_sentence, y_test]


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


def build_input_data(sentences, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    return x


def tokenizer(texts, word_index, max_length):
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


def load_pianpang_data():
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels, sentence_raw = load_data_and_labels()
    length = [len(x) for x in sentences]
    max_sentence = max(length)
    print('index:{}'.format(length.index(max_sentence)))
    print('max_sentence:{}'.format(max_sentence))
    Word2VecModel = KeyedVectors.load_word2vec_format('./data/word_pinyin.bin', binary=True)
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

    sentences_padded = tokenizer(sentences, word_index, max_length=max_sentence)
    # x_test = tokenizer(x_test,word_index,max_length=max_sentence)
    # vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    # x, y = build_input_data(sentences_padded, labels, vocabulary)
    print('Loaded word pianpang')
    return [sentences_padded, labels, embeddings_matrix]


def load_pianpang_eval_data():
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels, sentence_raw = load_eval_data_and_labels()
    # length = [len(x) for x in sentences]
    max_sentence = 481
    # print('index:{}'.format(length.index(max_sentence)))
    # print('max_sentence:{}'.format(max_sentence))
    Word2VecModel = KeyedVectors.load_word2vec_format('./data/word_pinyin.bin', binary=True)
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

    sentences_padded = tokenizer(sentences, word_index, max_length=max_sentence)
    # x_test = tokenizer(x_test,word_index,max_length=max_sentence)
    # vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    # x, y = build_input_data(sentences_padded, labels, vocabulary)
    print('Loaded word pianpang')
    return [sentences_padded, labels]


def load_bushou_data():
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels, x_test, y_test = load_pinyin_data_and_labels()
    length = [len(x) for x in sentences]
    max_sentence = max(length)
    Word2VecModel = KeyedVectors.load_word2vec_format('./word_bushou.bin', binary=True)
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

    sentences_padded = tokenizer(sentences, word_index, max_length=max_sentence)
    x_test = tokenizer(x_test, word_index, max_length=max_sentence)
    # vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    # x, y = build_input_data(sentences_padded, labels, vocabulary)
    print('Loaded word bushou')
    return [sentences_padded, labels, embeddings_matrix, x_test, y_test]


if __name__ == '__main__':

    test = '一堆句畐事'
    sentence = [get_word_pianpang(item) for item in list(word_handle(movestopwords(test)))]
    Word2VecModel = KeyedVectors.load_word2vec_format('../data/word_pinyin.bin', binary=True)
    for i in sentence:
        if i not in Word2VecModel:
            print(i)
    # print(sentence)
