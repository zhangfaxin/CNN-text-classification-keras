import keras
import sys
sys.path.append('..')
from cnn_radical_two.two_input_helper import load_pianpang_eval_data
from sklearn.metrics import confusion_matrix,classification_report
import pandas as pd
from data_helpers import load_word_data,load_pinyin_data
import numpy as np


if __name__ == '__main__':
    # model = keras.models.load_model('./checkpoint/cnn_sentence_weights.010-0.9652.hdf5')
    # model = keras.models.load_model('./checkpoint/cnn_word_pinyin_weights.009-0.9660.hdf5')
    # model = keras.models.load_model('./checkpoint/cnn_sentence_pinyin_weights.007-0.9593.hdf5')
    model = keras.models.load_model('./joint_checkpoint/adjust_weights_0.3_0.3_0.3.010-0.9743.hdf5')

    model.summary()
    x, y, embeddings_matrix, x_eval, y_eval,sentence_raw = load_word_data()
    x, y, embeddings_matrix, x_eval_pinyin, y_eval = load_pinyin_data()
    # x, y, embeddings_matrix, x_eval, y_eval,sentence_raw = load_word_data()
    # loss_and_metric = model.evaluate(x_eval,y_e val,batch_size=64)
    x_radical_eval, y_eval = load_pianpang_eval_data()
    # x_eval_pinyin = np.concatenate((x_eval_pinyin,x_radical_eval),axis=1)
    prediction = model.predict([x_eval,x_radical_eval,x_eval_pinyin])
    y_pre = []
    for i in prediction:
        if i[0]>i[1]:
            y_pre.append(0)
        else:
            y_pre.append(1)
    evaluate = pd.DataFrame({'review':sentence_raw,'trueL':y_eval,'preL':y_pre})
    evaluate.to_csv('./joint_checkpoint/pianpang_add_input_evaluate_pianpang_pinyin_adjust0.3_0.3_0.3-pianpang.csv',header=True,index=False)
    report = classification_report(y_eval,y_pre,digits=3)
    print(report)
    print('#########################################')
    metric = confusion_matrix(y_eval,y_pre)
    # print("混淆矩阵：")
    print(metric)