from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D, LSTM, Bidirectional
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import train_test_split
from data_helpers import load_data
from keras.callbacks import EarlyStopping, TensorBoard

print('Loading data')
x, y, embeddings_matrix, x_eval, y_eval, x_eval_review = load_data()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

sequence_length = x.shape[1]  # 56
# vocabulary_size = len(vocabulary_inv) # 18765
embedding_dim = 200
num_filters = 128
drop = 0.5

epochs = 30
batch_size = 128

# this returns a tensor
print("Creating Model...")

model = Sequential()

model.add(Embedding(input_dim=len(embeddings_matrix), output_dim=embedding_dim, weights=[embeddings_matrix],
                    input_length=sequence_length, trainable=False))
model.add(Bidirectional(LSTM(300, dropout=0.3)))
model.add(MaxPool2D(pool_size=(1, 10), strides=(1, 1), padding='valid'))
model.add(Bidirectional(LSTM(300, dropout=0.3)))
model.add(MaxPool2D(pool_size=(1, 10), strides=(1, 1), padding='valid'))
model.add(Dense(units=2, activation='softmax'))
model.summary()

checkpoint = ModelCheckpoint('./lstm/lstm_word_pianpang_weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc',
                             verbose=1, save_best_only=True, mode='auto')
adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
# TensorBoard
tbCallBack = TensorBoard(log_dir='./lstm/', update_freq='batch')
# EalryStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=2)
print("Traning Model...")

x = model.get_layer(index=2).output
print(x)
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
          callbacks=[checkpoint, tbCallBack, early_stopping], validation_data=(X_test, y_test))  # starts training
