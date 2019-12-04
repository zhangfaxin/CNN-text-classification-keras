from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D,BatchNormalization
from keras.layers import Reshape, Flatten, Dropout, Concatenate, Add, Lambda
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from data_helpers import load_word_data, load_pinyin_data
import sys
sys.path.append('..')
from cnn_radical_two.two_input_helper import load_pianpang_data
from keras.callbacks import EarlyStopping, TensorBoard
from sklearn.utils import shuffle
from cnn_radical_two.attention_layer import AttLayer
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session



config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

print('Loading data')

x, y, embeddings_matrix, x_eval, y_eval, x_eval_raw = load_word_data()
x_pinyin, y_pinyin, embeddings_matrix_3, x_eval_pinyin, y_eval_pinyin = load_pinyin_data()
x_radical, y, embeddings_matrix_2 = load_pianpang_data()

# x_pinyin = x_pinyin + x_radical
x, x_pinyin, x_radical, y = shuffle(x, x_pinyin, x_radical, y)

dev_sample_index = -1 * int(0.1 * float(len(y)))
X_train, X_test = x[:dev_sample_index], x[dev_sample_index:]
y_train, y_test = y[:dev_sample_index], y[dev_sample_index:]

x_train_radical, X_test_radical = x_radical[:dev_sample_index], x_radical[dev_sample_index:]

x_train_pinyin, X_test_pinyin = x_pinyin[:dev_sample_index], x_pinyin[dev_sample_index:]

sequence_length = X_train.shape[1]

sequence_length_2 = x_train_radical.shape[1]

sequence_length_3 = x_train_pinyin.shape[1]
print("sequence_length_3:%d" % sequence_length_3)

embedding_dim = 128

filter_sizes = [3, 4, 5]
num_filters = 128
drop = 0.5

epochs = 15
batch_size = 128

# this returns a tensor
print("Creating Model...")
inputs = Input(shape=(sequence_length,), dtype='int32', name='input_A')
embedding_layer = Embedding(input_dim=len(embeddings_matrix), output_dim=embedding_dim, weights=[embeddings_matrix],
                            input_length=sequence_length, trainable=False, name='word')
embedded_sequences = embedding_layer(inputs)
reshape = Reshape((sequence_length, embedding_dim, 1))(embedded_sequences)

x_radical_inputs = Input(shape=(sequence_length_2,), dtype='int32', name='input_B')
embedding_layer_2 = Embedding(input_dim=len(embeddings_matrix_2), output_dim=embedding_dim,
                              weights=[embeddings_matrix_2], input_length=sequence_length_2, trainable=False,
                              name='pianpang')
embedded_sequences_2 = embedding_layer_2(x_radical_inputs)
reshape_radical = Reshape((sequence_length_2, embedding_dim, 1))(embedded_sequences_2)

x_pinyin_inputs = Input(shape=(sequence_length_3,), dtype='int32', name='input_C')
embedding_layer_3 = Embedding(input_dim=len(embeddings_matrix_3), output_dim=embedding_dim,
                              weights=[embeddings_matrix_3], input_length=sequence_length_3, trainable=False,
                              name='pinyin')
embedded_sequences_3 = embedding_layer_3(x_pinyin_inputs)
reshape_pinyin = Reshape((sequence_length_3, embedding_dim, 1))(embedded_sequences_3)

# reshape_pinyin = Add()([reshape_radical,reshape_pinyin])

conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal',
                activation='relu', name='A1')(reshape)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal',
                activation='relu', name='A2')(reshape)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal',
                activation='relu', name='A3')(reshape)

conv_0_radical = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid',
                        kernel_initializer='normal', activation='relu')(reshape_radical)
conv_1_radical = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid',
                        kernel_initializer='normal', activation='relu')(reshape_radical)
conv_2_radical = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid',
                        kernel_initializer='normal', activation='relu')(reshape_radical)

conv_0_pinyin = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid',
                       kernel_initializer='normal', activation='relu')(reshape_pinyin)
conv_1_pinyin = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid',
                       kernel_initializer='normal', activation='relu')(reshape_pinyin)
conv_2_pinyin = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid',
                       kernel_initializer='normal', activation='relu')(reshape_pinyin)

maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(conv_0)
maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1, 1), padding='valid')(conv_1)
maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1, 1), padding='valid')(conv_2)

maxpool_0_radical = MaxPool2D(pool_size=(sequence_length_2 - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(
    conv_0_radical)
maxpool_1_radical = MaxPool2D(pool_size=(sequence_length_2 - filter_sizes[1] + 1, 1), strides=(1, 1), padding='valid')(
    conv_1_radical)
maxpool_2_radical = MaxPool2D(pool_size=(sequence_length_2 - filter_sizes[2] + 1, 1), strides=(1, 1), padding='valid')(
    conv_2_radical)

maxpool_0_pinyin = MaxPool2D(pool_size=(sequence_length_3 - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(
    conv_0_pinyin)
maxpool_1_pinyin = MaxPool2D(pool_size=(sequence_length_3 - filter_sizes[1] + 1, 1), strides=(1, 1), padding='valid')(
    conv_1_pinyin)
maxpool_2_pinyin = MaxPool2D(pool_size=(sequence_length_3 - filter_sizes[2] + 1, 1), strides=(1, 1), padding='valid')(
    conv_2_pinyin)

concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
# concatenated_tensor = BatchNormalization()(concatenated_tensor)
flatten = Flatten()(concatenated_tensor)

concatenated_tensor_radical = Concatenate(axis=1)([maxpool_0_radical, maxpool_1_radical, maxpool_2_radical])
flatten_radical = Flatten()(concatenated_tensor_radical)

concatenated_tensor_pinyin = Concatenate(axis=1)([maxpool_0_pinyin, maxpool_1_pinyin, maxpool_2_pinyin])
flatten_pinyin = Flatten()(concatenated_tensor_pinyin)

# flatten = BatchNormalization()(flatten)


# weight_1 = Lambda(lambda x: x * 0.2)
# weight_2 = Lambda(lambda x: x * 0.2)
# weight_3 = Lambda(lambda x: x * 0.2)
#
# flatten = weight_1(flatten)
# flatten_radical = weight_2(flatten_radical)
# flatten_pinyin = weight_3(flatten_pinyin)

# flatten_radical = BatchNormalization()(flatten_radical)

flatten_pinyin = Add()([flatten_radical, flatten_pinyin])
# flatten_pinyin = BatchNormalization()(flatten_pinyin)
# flatten = BatchNormalization()(flatten)
# flatten_pinyin = BatchNormalization()(flatten_pinyin)
# flatten_concat = Concatenate(axis=1)([flatten, flatten_pinyin])
flatten_concat = Add()([flatten,flatten_pinyin])
# flatten_concat = BatchNormalization()(flatten_concat)
dropout = Dropout(drop)(flatten_concat)
# dropout = AttLayer()(dropout)
output = Dense(units=2, activation='softmax')(dropout)

# this creates a model that includes
model = Model(inputs=[inputs, x_radical_inputs, x_pinyin_inputs], outputs=output)

checkpoint = ModelCheckpoint('./adjust_weight/BatchNormalization.{epoch:03d}-{val_acc:.4f}.hdf5',
                             monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
# TensorBoard
tbCallBack = TensorBoard(log_dir='./adjust_weight/', update_freq='batch')
# EalryStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=2)
print("Traning Model...")
model.fit([X_train, x_train_radical, x_train_pinyin], y_train, batch_size=batch_size, epochs=epochs, verbose=1,
          callbacks=[checkpoint, tbCallBack, early_stopping],
          validation_data=([X_test, X_test_radical, X_test_pinyin], y_test))  # starts training
