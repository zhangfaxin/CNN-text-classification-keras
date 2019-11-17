import keras
from data_helpers import load_data,load_pinyin_data,load_word_data
from sklearn.metrics import confusion_matrix,classification_report
import pandas as pd

if __name__ == '__main__':
    # model = keras.models.load_model('./checkpoint/cnn_sentence_weights.010-0.9652.hdf5')
    model = keras.models.load_model('./checkpoint/cnn_word_pinyin_weights.009-0.9660.hdf5')
    # model = keras.models.load_model('./checkpoint/cnn_sentence_pinyin_weights.007-0.9593.hdf5')

    model.summary()
    # x, y, embeddings_matrix, x_eval, y_eval,sentence_raw = load_data()
    x_pinyin, y, embeddings_matrix_2, x_pinyin_eval, y_eval = load_pinyin_data()
    x, y, embeddings_matrix, x_eval, y_eval,sentence_raw = load_word_data()
    # loss_and_metric = model.evaluate(x_eval,y_e val,batch_size=64)
    prediction = model.predict(x_pinyin_eval)
    y_pre = []
    for i in prediction:
        if i[0]>i[1]:
            y_pre.append(0)
        else:
            y_pre.append(1)
    evaluate = pd.DataFrame({'review':sentence_raw,'trueL':y_eval,'preL':y_pre})
    evaluate.to_csv('./data/cnn_word_pinyin_evaluate-pianpang.csv',header=True,index=False)
    report = classification_report(y_eval,y_pre,digits=3)
    print(report)
    print('#########################################')
    metric = confusion_matrix(y_eval,y_pre)
    print("混淆矩阵：")
    print(metric)