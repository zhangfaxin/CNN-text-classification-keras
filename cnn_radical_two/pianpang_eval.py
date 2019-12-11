import keras
from data_helpers import load_data
from sklearn.metrics import confusion_matrix,classification_report
from cnn_radical_two.two_input_helper import load_pianpang_eval_data
import pandas as pd

if __name__ == '__main__':
    # model = keras.models.load_model('./checkpoint/cnn_sentence_weights.010-0.9652.hdf5')
    # model = keras.models.load_model('./checkpoint/cnn_word_pinyin_weights.009-0.9660.hdf5')
    # model = keras.models.load_model('./checkpoint/cnn_sentence_pinyin_weights.007-0.9593.hdf5')
    model = keras.models.load_model('./checkpoint_pianpang/cnn_new_method_word_pinyin_pianpang_weights.008-0.9517.hdf5')
    # model = keras.models.load_model('./checkpoint_pianpang/cnn_word_pinyin_pianpang_weights.009-0.9522.hdf5')

    model.summary()

    x, y, embeddings_matrix, x_eval_data, y_eval,sentence_raw = load_data()
    x_eval, y_eval = load_pianpang_eval_data()
    # loss_and_metric = model.evaluate(x_eval,y_e val,batch_size=64)
    prediction = model.predict(x_eval)
    y_pre = []
    for i in prediction:
        if i[0]>i[1]:
            y_pre.append(0)
        else:
            y_pre.append(1)
    evaluate = pd.DataFrame({'review':sentence_raw,'trueL':y_eval,'preL':y_pre})
    evaluate.to_csv('../pianpang/pianpang_evaluate-pianpang.csv',header=True,index=False)
    report = classification_report(y_eval,y_pre,digits=3)
    print(report)
    print('#########################################')
    metric = confusion_matrix(y_eval,y_pre)
    print("混淆矩阵：")
    print(metric)