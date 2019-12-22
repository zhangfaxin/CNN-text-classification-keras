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
from pypinyin import pinyin, lazy_pinyin, Style
import pypinyin
import pkg_resources

from soundshapecode.four_corner import FourCornerMethod
import soundshapecode.ssc_similarity.compute_ssc_similarity as compute_ssc_similarity

fcm = FourCornerMethod()

from soundshapecode.variant_kmp import VatiantKMP

SIMILARITY_THRESHOLD = 0.35
SSC_ENCODE_WAY = 'SHAPE'  # 'ALL','SOUND','SHAPE'

yunmuDict = {'a': '1', 'o': '2', 'e': '3', 'i': '4',
             'u': '5', 'v': '6', 'ai': '7', 'ei': '7',
             'ui': '8', 'ao': '9', 'ou': 'A', 'iou': 'B',  # 有：you->yiou->iou->iu
             'ie': 'C', 've': 'D', 'er': 'E', 'an': 'F',
             'en': 'G', 'in': 'H', 'un': 'I', 'vn': 'J',  # 晕：yun->yvn->vn->ven
             'ang': 'F', 'eng': 'G', 'ing': 'H', 'ong': 'K'}

shengmuDict = {'b': '1', 'p': '2', 'm': '3', 'f': '4',
               'd': '5', 't': '6', 'n': '7', 'l': '7',
               'g': '8', 'k': '9', 'h': 'A', 'j': 'B',
               'q': 'C', 'x': 'D', 'zh': 'E', 'ch': 'F',
               'sh': 'G', 'r': 'H', 'z': 'E', 'c': 'F',
               's': 'G', 'y': 'I', 'w': 'J', '0': '0'}

shapeDict = {'⿰': '1', '⿱': '2', '⿲': '3', '⿳': '4', '⿴': '5',  # 左右结构、上下、左中右、上中下、全包围
             '⿵': '6', '⿶': '7', '⿷': '8', '⿸': '9', '⿹': 'A',  # 上三包、下三包、左三包、左上包、右上包
             '⿺': 'B', '⿻': 'C', '0': '0'}  # 左下包、镶嵌、独体字：0

strokesDict = {1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A',
               11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K',
               21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U',
               31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 0: '0'}

hanziStrokesDict = {}  # 汉子：笔画数
hanziStructureDict = {}  # 汉子：形体结构


def getSoundCode(one_chi_word):
    res = []
    shengmuStr = pinyin(one_chi_word, style=pypinyin.INITIALS, heteronym=False, strict=False)[0][0]
    if shengmuStr not in shengmuDict:
        shengmuStr = '0'

    yunmuStrFullStrict = pinyin(one_chi_word, style=pypinyin.FINALS_TONE3, heteronym=False, strict=True)[0][0]

    yindiao = '0'
    if yunmuStrFullStrict[-1] in ['1', '2', '3', '4']:
        yindiao = yunmuStrFullStrict[-1]
        yunmuStrFullStrict = yunmuStrFullStrict[:-1]

    if yunmuStrFullStrict in yunmuDict:
        # 声母，韵母辅音补码，韵母，音调
        res.append(yunmuDict[yunmuStrFullStrict])
        res.append(shengmuDict[shengmuStr])
        res.append('0')
    elif len(yunmuStrFullStrict) > 1:
        res.append(yunmuDict[yunmuStrFullStrict[1:]])
        res.append(shengmuDict[shengmuStr])
        res.append(yunmuDict[yunmuStrFullStrict[0]])
    else:
        res.append('0')
        res.append(shengmuDict[shengmuStr])
        res.append('0')

    res.append(yindiao)
    return res


def getShapeCode(one_chi_word):
    res = []
    structureShape = hanziStructureDict.get(one_chi_word, '0')  # 形体结构
    res.append(shapeDict[structureShape])

    fourCornerCode = fcm.query(one_chi_word)  # 四角号码（5位数字）
    if fourCornerCode is None:
        res.extend(['0', '0', '0', '0', '0'])
    else:
        res.extend(fourCornerCode[:])

    strokes = hanziStrokesDict.get(one_chi_word, '0')  # 笔画数
    if int(strokes) > 35:
        res.append('Z')
    else:
        res.append(strokesDict[int(strokes)])
    return res


def getHanziStrokesDict():
    strokes_filepath = pkg_resources.resource_filename(__name__, "../zh_data/utf8_strokes.txt")
    with open(strokes_filepath, 'r', encoding='UTF-8') as f:  # 文件特征：
        for line in f:
            line = line.split()
            hanziStrokesDict[line[1]] = line[2]


def getHanziStructureDict():
    structure_filepath = pkg_resources.resource_filename(__name__, "../zh_data/unihan_structure.txt")
    with open(structure_filepath, 'r', encoding='UTF-8') as f:  # 文件特征：U+4EFF\t仿\t⿰亻方\n
        for line in f:
            line = line.split()
            if line[2][0] in shapeDict:
                hanziStructureDict[line[1]] = line[2][0]


def generateHanziSSCFile():
    readFilePath = pkg_resources.resource_filename(__name__, "../zh_data/unihan_structure.txt")
    writeFilePath = pkg_resources.resource_filename(__name__, "../zh_data/hanzi_ssc_res.txt")
    writeFile = open(writeFilePath, "w", encoding='UTF-8')
    with open(readFilePath, 'r', encoding='UTF-8') as f:  # 文件特征：U+4EFF\t仿\t⿰亻方\n
        for line in f:
            line = line.split()
            soundCode = getSoundCode(line[1])
            shapeCode = getShapeCode(line[1])
            ssc = "".join(soundCode + shapeCode)
            if ssc != '00000000000':
                writeFile.write(line[0] + "\t" + line[1] + "\t" + ssc + "\n")
    writeFile.close()
    print('结束！')


hanziSSCDict = {}  # 汉子：SSC码


def getHanziSSCDict():
    hanzi_ssc_filepath = pkg_resources.resource_filename(__name__, "../zh_data/hanzi_ssc_res.txt")
    with open(hanzi_ssc_filepath, 'r', encoding='UTF-8') as f:  # 文件特征：U+4EFF\t仿\t音形码\n
        for line in f:
            line = line.split()
            hanziSSCDict[line[1]] = line[2]


def getSSC(hanzi_sentence, encode_way):
    hanzi_sentence_ssc_list = []
    for one_chi_word in hanzi_sentence:
        ssc = hanziSSCDict.get(one_chi_word, None)
        if ssc is None:
            soundCode = getSoundCode(one_chi_word)
            shapeCode = getShapeCode(one_chi_word)
            ssc = "".join(soundCode + shapeCode)
        if encode_way == "SOUND":
            ssc = ssc[:4]
        elif encode_way == "SHAPE":
            ssc = ssc[4:]
        else:
            pass
        hanzi_sentence_ssc_list.append(ssc)
    return hanzi_sentence_ssc_list


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
    return [sentence, np.array(y), x_test_sentence, y_test, x_test_review]


def load_dict_from_file():
    _dict = {}
    with open('./wubi21003+.txt', 'r', encoding='utf-8') as dict_file:
        for line in dict_file:
            (key, value) = line.strip().split('	')
            _dict[key] = value

    return _dict


def get_label(word):
    label = list(getSSC(word, SSC_ENCODE_WAY)[0])
    return list(map(int, label))


def load_data_and_labels():
    """
    Loads polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    df = pd.read_csv('../data/train_data_1.csv')
    review_part = df.review
    sentence = [[item for item in list(movestopwords(s))] for s in review_part]
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
    return [sentence, np.array(y), x_test_sentence, y_test, x_test_review]


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
    result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0]
    for i in onehot_encoded:
        result = np.add(result, i)
    return list(result)
    # print(onehot_encoded)
    # invert encoding
    # inverted = int_to_char[np.argmax(onehot_encoded[0])]
    # print(inverted)


def label_encode_sentence(sentence):
    result_sentence = []
    for i in sentence:
        word_small = []
        for word in i:
            if word != ' ':
                encoded_word = get_label(word)
                word_small.append(encoded_word)
            else:
                encoded_word = get_label('一')
                word_small.append(encoded_word)
        result_sentence.append(np.array(word_small))
    return result_sentence


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
    sentences, labels, x_test, y_test, sentence_raw = load_data_and_labels()
    length = [len(x) for x in sentences]
    max_sentence = max(length)
    sentences = padding_sentences(sentences, padding_sentence_length=max_sentence, padding_token='一')
    x_test = padding_sentences(x_test, padding_sentence_length=max_sentence, padding_token='一')
    sentences_to_encode = label_encode_sentence(sentences[0])
    x_test = label_encode_sentence(x_test[0])
    print('index:{}'.format(length.index(max_sentence)))
    print('max_sentence:{}'.format(max_sentence))
    sentences_to_encode = np.array(sentences_to_encode)
    x_test = np.array(x_test)
    return [sentences_to_encode, labels, x_test, y_test, sentence_raw, max_sentence]


if __name__ == '__main__':
    get_one_hot_label('123')
    print("1223")
