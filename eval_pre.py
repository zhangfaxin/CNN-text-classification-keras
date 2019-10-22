import keras
from data_helpers import load_data
from sklearn.metrics import confusion_matrix,classification_report

if __name__ == '__main__':
    model = keras.models.load_model('./weights.007-0.9505.hdf5')
    model.summary()
    x, y, embeddings_matrix, x_eval, y_eval = load_data()
    # loss_and_metric = model.evaluate(x_eval,y_eval,batch_size=64)
    prediction = model.predict(x_eval)
    y_pre = []
    for i in prediction:
        if i[0]>i[1]:
            y_pre.append(0)
        else:
            y_pre.append(1)
    report = classification_report(y_eval,y_pre)
    print(report)
    print('#########################################')
    metric = confusion_matrix(y_eval,y_pre)
    print("混淆矩阵：")
    print(metric)