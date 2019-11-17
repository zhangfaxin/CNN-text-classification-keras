import jieba.posseg as pos
from pyhanlp import *
from pypinyin import lazy_pinyin,Style


def get_sentence(sentence):
    flag_list = ['a','eng','c','ns','v','n','uj','r','m','ns','nr','Ng']
    sentence_part = pos.cut(sentence)
    sentence_word = []
    for i in sentence_part:
        if i.flag in flag_list:
            sentence_word.append(i.word)
    # shengmu,tone = get_sentence_tone(sentence_word)
    return sentence_word


def get_pinyin_sentence(sentence):
    flag_list = ['a','eng','c','ns','v','n','uj','r','m','ns','nr','Ng']
    sentence_part = pos.cut(sentence)
    sentence_word = []
    for i in sentence_part:
        if i.flag in flag_list:
            sentence_word.append(i.word)
    # shengmu,tone = get_sentence_tone(sentence_word)
    return ''.join(sentence_word)


def get_sentence_tone(word):
    tone = []
    shengmu = []
    shengmu_list = ['b','p','m','f','d','t','n','l','g','k','h','j','q','x','zh','ch','sh','r','z','c','s','y','w']
    for i in word:
        if i.flag != 'eng':
            shengmu.extend(lazy_pinyin(i.word, style=Style.INITIALS))
            pinyin_list = HanLP.convertToPinyinList(i.word)
            for j in pinyin_list:
                tone.append(j.getTone())
        else:
            eng_list = list(i.word)
            for j in eng_list:
                if j in  shengmu_list:
                    shengmu.append(j)
                    tone.append(1)
    return shengmu,tone




if __name__ == '__main__':
    sentence = '一个黑头都没拽出来，tmd用了三次了，一点效果也没有，在也不信限时抢购货了，腊鸡'
    get_sentence(sentence)