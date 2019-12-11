from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import train_test_split
from data_helpers import load_data, load_pinyin_data, load_word_data
from keras.callbacks import EarlyStopping, TensorBoard
import keras
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegressionCV
from sklearn import svm
from cnn_radical_two.two_input_helper import load_pianpang_data, load_pianpang_eval_data

print('Loading data')
x, y, embeddings_matrix, x_eval, y_eval, x_raw = load_word_data()
x_pinyin, y_pinyin, embeddings_matrix, x_eval_pinyin, y_eval = load_pinyin_data()
x_radical, y, embeddings_matrix_2 = load_pianpang_data()
x_radical_eval, y_eval = load_pianpang_eval_data()

x, x_pinyin, x_radical, y = shuffle(x, x_pinyin, x_radical, y)
# x.shape -> (10662, 56)
# y.shape -> (10662, 2)
# len(vocabulary) -> 18765
# len(vocabulary_inv) -> 18765
model = keras.models.load_model('./checkpoint/cnn_word_weights.010-0.9775.hdf5')
model2 = keras.models.load_model('./checkpoint/cnn_word_pinyin_weights.009-0.9660.hdf5')
model3 = keras.models.load_model(
    './cnn_radical_two/checkpoint_pianpang/cnn_new_method_word_pinyin_pianpang_weights.008-0.9517.hdf5')

dense1_layer_model = Model(inputs=model.input, outputs=model.output)
dense2_layer_model = Model(inputs=model2.input, outputs=model2.output)
dense3_layer_model = Model(inputs=model3.input, outputs=model3.output)

# xtemp = dense1_layer_model.predict(x)
# xtemp_pinyin = dense2_layer_model.predict(x_pinyin)
# xtemp_radical = dense3_layer_model.predict(x_radical)
#
# xtemp_array = []
# xtemp_pinyin_array = []
# xtemp_radical_array = []
# for i in list(xtemp):
#     xtemp_array.append([i[1]])
# for i in list(xtemp_pinyin):
#     xtemp_pinyin_array.append([i[1]])
# for i in list(xtemp_radical):
#     xtemp_radical_array.append([i[1]])
# xtemp_array = np.array(xtemp_array)
# xtemp_pinyin_array = np.array(xtemp_pinyin_array)
# xtemp_radical_array = np.array(xtemp_radical_array)


x_eval = dense1_layer_model.predict(x_eval)
x_eval_pinyin = dense2_layer_model.predict(x_eval_pinyin)
x_eval_radical = dense3_layer_model.predict(x_radical_eval)

# x_eval_array = []
# x_eval_pinyin_array = []
# x_eval_radical_array = []
# for i in list(x_eval):
#     x_eval_array.append([i[1]])
# for i in list(x_eval_pinyin):
#     x_eval_pinyin_array.append([i[1]])
# for i in list(x_eval_radical):
#     x_eval_radical_array.append([i[1]])
# x_eval_array = np.array(x_eval_array)
# x_eval_pinyin_array = np.array(x_eval_pinyin_array)
# x_eval_radical_array = np.array(x_eval_radical_array)


# lr = LogisticRegressionCV(multi_class="ovr", fit_intercept=True, Cs=np.logspace(-2, 2, 20), cv=2, penalty="l2",
#                           solver="sag", tol=0.01)
# total_x = np.concatenate((xtemp_array,xtemp_pinyin_array,xtemp_radical_array),axis=1)
# print(total_x.shape)
# total_x = xtemp * 0.3 + xtemp_radical * 0.3 + xtemp_pinyin * 0.3
# total_eval = np.concatenate((x_eval_array,x_eval_pinyin_array,x_eval_radical_array),axis=1)
# total_eval = x_eval * 0.3 + x_eval_radical * 0.3 + x_eval_pinyin * 0.3
# clf = svm.LinearSVC()
y_true = []
for i in y:
    if i[0] == 1:
        y_true.append(0)
    else:
        y_true.append(1)

# lr.fit(total_x, y_true)
# clf.fit(total_x,y_true)
# print("权重：",lr.coef_)
# print("偏置：",lr.intercept_)

# pre = lr.predict(total_eval)
y_pre = []
pre_eval = x_eval * 0.03 + x_eval_pinyin * 0.47 + x_eval_radical * 0.50
for t in pre_eval:
    if t[0] > t[1]:
        y_pre.append(0)
    else:
        y_pre.append(1)
report = classification_report(y_eval, np.array(y_pre), digits=3)
#
# for i in range(0, 100, 1):
#     for j in range(0, 100-i, 1):
#         k = 100 - i - j
#         pre_eval = x_eval * i/100 + x_eval_pinyin * j/100 + x_eval_radical * k/100
#         y_pre = []
#         for t in pre_eval:
#             if t[0] > t[1]:
#                 y_pre.append(0)
#             else:
#                 y_pre.append(1)
#         report = classification_report(y_eval,np.array(y_pre),digits=3)
#         file = open('./record/weight.txt','a')
#         weight = str(i) + ' '+str(j)+' ' +str(k)+ ' '
#         file.write(weight)
#         file.write(report)
#         file.write('/n')
#         file.close()

evaluate = pd.DataFrame({'review': x_raw, 'trueL': y_eval, 'preL': np.array(y_pre)})
evaluate.to_csv('./pianpang/weight_comb-pianpang.csv', header=True, index=False)
print(confusion_matrix(y_eval, np.array(y_pre)))
print(report)
