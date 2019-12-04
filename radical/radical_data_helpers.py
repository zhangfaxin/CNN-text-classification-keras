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
    stopwords = [line.strip() for line in open('../data/stop_ch.txt', 'r', encoding='utf-8').readlines()]
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


def load_dict_from_file():
    _dict = {}
    with open('./wubi21003+.txt', 'r',encoding='utf-8') as dict_file:
        for line in dict_file:
            (key, value) = line.strip().split('	')
            _dict[key] = value

    return _dict


def load_data_and_labels():
    """
    Loads polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    df = pd.read_csv('../data/train_data_1.csv')
    review_part = df.review
    dict_file = load_dict_from_file()
    total_sentence = []
    sentence = [[item for item in list(movestopwords(s))] for s in review_part]
    # sentence = [part_of_speech.get_sentence(s) for s in review_part]
    for item in sentence:
        while True:
            if ' ' in item:
                item.remove(' ')
            else:
                break
    for i in sentence:
        radical_sentence = []
        for word in i:
            if word in dict_file:
                radical_sentence.append(dict_file[word])
            else:
                radical_sentence.append(word)
        total_sentence.append(radical_sentence)
    y_label = df.label
    y = []
    for i in y_label:
        if i == '0' or i == 0.0:
            y.append([1, 0])
        else:
            y.append([0, 1])
    x_test = pd.read_csv('../data/test_data_1-pianpang.csv')
    x_test_review = x_test.review
    x_test_sentence = [[item for item in list(movestopwords(s))] for s in x_test_review]
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
    return [total_sentence, np.array(y),x_test_sentence,y_test,x_test_review]


def word2pinyin(word):
    pinyin = lazy_pinyin(word)
    return "".join(pinyin)


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


def get_one_hot_label(word):
    word2 = word.lower()
    alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789'
    alphabet_list = list(alphabet)
    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    # integer encode input data
    integer_encoded = [char_to_int[char] for char in word2 if char in alphabet_list]
    # print(integer_encoded)'
    # one hot encode
    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        onehot_encoded.append(letter)
    result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in onehot_encoded:
        result = np.add(result,i)
    return list(result)
    # print(onehot_encoded)
    # invert encoding
    # inverted = int_to_char[np.argmax(onehot_encoded[0])]
    # print(inverted)


def encode_sentence(sentence):
    result_sentence = []
    for i in sentence:
        word_small = []
        for word in i:
            if word != ' ':
                encoded_word = get_one_hot_label(word)
                word_small.append(encoded_word)
            else:
                encoded_word = get_one_hot_label('bnh')
                word_small.append(encoded_word)
        result_sentence.append(np.array(word_small))
    return result_sentence


def padding_sentences(input_sentences, padding_token, padding_sentence_length=None):
    # sentence_list = []
    # for line in input_sentences:
    #     sentence_list.append(nltk.word_tokenize(line))
    max_sentence_length = padding_sentence_length if padding_sentence_length is not None else max(
        [len(sentence) for sentence in input_sentences])
    result_sentence = []
    for sentence in input_sentences:
        if len(sentence) > max_sentence_length:
            sentence = sentence[:max_sentence_length]
            result_sentence.append(sentence)
        else:
            sentence.extend([padding_token] * (max_sentence_length - len(sentence)))
            result_sentence.append(sentence)
    return result_sentence, max_sentence_length


def load_data():
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels,x_test,y_test,sentence_raw = load_data_and_labels()
    length = [len(x) for x in sentences]
    max_sentence = max(length)
    sentences = padding_sentences(sentences,padding_sentence_length=max_sentence,padding_token='bnh')
    x_test = padding_sentences(x_test,padding_sentence_length=max_sentence,padding_token='bnh')
    # sentences_to_encode = encode_sentence(sentences[0])
    x_test = encode_sentence(x_test[0])
    print('index:{}'.format(length.index(max_sentence)))
    print('max_sentence:{}'.format(max_sentence))
    sentences_to_encode = np.array(sentences)
    x_test = np.array(x_test)
    return [sentences_to_encode,labels,x_test,y_test,sentence_raw]


if __name__ == '__main__':
    get_one_hot_label('123')
    print("1223")

