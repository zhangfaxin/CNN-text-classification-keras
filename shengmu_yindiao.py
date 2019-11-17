from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import train_test_split
from data_helpers import load_data,load_pinyin_data,load_shengmu_data
from keras.callbacks import EarlyStopping,TensorBoard
from sklearn.utils import shuffle
print('Loading data')
x, x_tone,y, embeddings_matrix, x_eval,x_eval_tone, y_eval,x_eval_raw,vocabulary, vocabulary_inv = load_shengmu_data()
# x.shape -> (10662, 56)
# y.shape -> (10662, 2)
# len(vocabulary) -> 18765
# len(vocabulary_inv) -> 18765
dev_sample_index = -1 * int(0.1 * float(len(y)))
X_train, X_test = x[:dev_sample_index], x[dev_sample_index:]
y_train, y_test = y[:dev_sample_index], y[dev_sample_index:]

x_tone, X_test_tone = x_tone[:dev_sample_index], x_tone[dev_sample_index:]
# y_train_pinyin, y_test_pinyin = y[:dev_sample_index], y[dev_sample_index:]

# X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.1, random_state=42)
# x_train_pinyin, X_test_pinyin, y_train_pinyin, y_test_pinyin = train_test_split( x_pinyin, y, test_size=0.1, random_state=42)

# X_train.shape -> (8529, 56)
# y_train.shape -> (8529, 2)
# X_test.shape -> (2133, 56)
# y_test.shape -> (2133, 2)


sequence_length = X_train.shape[1] # 56
sequence_length_2 = x_tone.shape[1]
# vocabulary_size = len(vocabulary_inv) # 18765
embedding_dim = 64
filter_sizes = [3,4,5]
num_filters = 64
drop = 0.5

epochs = 20
batch_size = 128

# this returns a tensor
inputs = Input(shape=(sequence_length,), dtype='int32',name='input_A')

embedding_layer = Embedding(input_dim=len(embeddings_matrix), output_dim=embedding_dim,weights=[embeddings_matrix],input_length=sequence_length,trainable=False,name='EA')
embedded_sequences = embedding_layer(inputs)
reshape = Reshape((sequence_length,embedding_dim,1))(embedded_sequences)

# x_pinyin_inputs = Input(shape=(sequence_length_2,), dtype='int32',name='input_B')
# embedding_layer_2 = Embedding(input_dim=len(vocabulary_inv), output_dim=embedding_dim,input_length=sequence_length_2,trainable=False,name='EB')
# embedded_sequences_2 = embedding_layer_2(x_pinyin_inputs)
# reshape_pinyin = Reshape((sequence_length_2,embedding_dim,1))(embedded_sequences_2)

conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu',name='A1')(reshape)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu',name='A2')(reshape)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu',name='A3')(reshape)

# conv_0_pinyin = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape_pinyin)
# conv_1_pinyin = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape_pinyin)
# conv_2_pinyin = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape_pinyin)

maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

# maxpool_0_pinyin = MaxPool2D(pool_size=(sequence_length_2 - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0_pinyin)
# maxpool_1_pinyin = MaxPool2D(pool_size=(sequence_length_2 - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1_pinyin)
# maxpool_2_pinyin = MaxPool2D(pool_size=(sequence_length_2 - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2_pinyin)

concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
flatten = Flatten()(concatenated_tensor)

# concatenated_tensor_pinyin = Concatenate(axis=1)([maxpool_0_pinyin, maxpool_1_pinyin, maxpool_2_pinyin])
# flatten_pinyin = Flatten()(concatenated_tensor_pinyin)
# flatten_concat = Concatenate(axis=1)([flatten,flatten_pinyin])
dropout = Dropout(drop)(flatten)
output = Dense(units=2, activation='softmax')(dropout)

# this creates a model that includes
model = Model(inputs=inputs, outputs=output)

checkpoint = ModelCheckpoint('./checkpoint/shengmu_weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
# TensorBoard
tbCallBack = TensorBoard(log_dir='./train_result/',update_freq='batch')
# EalryStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=2)
print("Traning Model...")
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint,tbCallBack, early_stopping], validation_data=(X_test, y_test) )  # starts training
