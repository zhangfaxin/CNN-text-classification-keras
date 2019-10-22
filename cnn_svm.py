from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import train_test_split
from data_helpers import load_data,load_pinyin_data
from keras.callbacks import EarlyStopping,TensorBoard
import keras
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn import svm
print('Loading data')
x, y, embeddings_matrix, x_eval, y_eval = load_data()
x_pinyin, y_pinyin, embeddings_matrix, x_eval_pinyin, y_eval = load_pinyin_data()
# x.shape -> (10662, 56)
# y.shape -> (10662, 2)
# len(vocabulary) -> 18765
# len(vocabulary_inv) -> 18765
model = keras.models.load_model('./weights.007-0.9505.hdf5')
model2 = keras.models.load_model('./pinyin_weights.006-0.9433.hdf5')
dense1_layer_model = Model(inputs=model.input,outputs=model.layers[-1].output)
dense2_layer_model = Model(inputs=model2.input,outputs=model2.layers[-1].output)
xtemp = dense1_layer_model.predict(x)
xtemp_pinyin = dense2_layer_model.predict(x_pinyin)
x_eval = dense1_layer_model.predict(x_eval)
x_eval_pinyin = dense2_layer_model.predict(x_eval_pinyin)
lr = LogisticRegressionCV(multi_class="ovr",fit_intercept=True,Cs=np.logspace(-2,2,20),cv=2,penalty="l2",solver="lbfgs",tol=0.01)
total_x = np.concatenate((xtemp,xtemp_pinyin),axis=1)
total_eval = np.concatenate((x_eval,x_eval_pinyin),axis=1)
# clf = svm.LinearSVC()
y_true = []
for i in y:
    if i[0] == 1:
        y_true.append(0)
    else:
        y_true.append(1)


lr.fit(total_x,y_true)
# clf.fit(total_x,y_true)
pre = lr.predict(total_eval)
report = classification_report(y_eval,pre)
print(confusion_matrix(y_eval,pre))
print(report)
