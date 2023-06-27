#encoding=utf8

import tensorflow as tf 
import keras 
import numpy as np 
import os
import sys
sys.path.append('../')
from models.model import Three_D_SN

from utils.load_data import load_char_data,load_word_embed,load_char_embed,load_all_data,load_data

np.random.seed(1)
tf.set_random_seed(1)


base_params = {
    'num_classes':2,
    'max_features':1700,
    'embed_size':200,
    'filters':300,
    'kernel_size':3,
    'strides':1,
    'padding':'same',
    'conv_activation_func':'relu',
    'embedding_matrix':[],
    'w_initializer':'random_uniform',
    'b_initializer':'zeros',
    'dropout_rate':0.2,
    'mlp_activation_func':'relu',
    'mlp_num_layers':1,
    'mlp_num_units':128,
    'mlp_num_fan_out':128,
    'input_shapes':[(64,),(64,)],
    'task':'Classification',
    'dataset':'qqp',
}

params = base_params
backend = Three_D_SN(params)
file = params['dataset']
p, h, y = load_data('input/{}/train.csv'.format(file), data_size=None,maxlen=params['input_shapes'][0][0])
x = [p,h]
y = tf.keras.utils.to_categorical(y,num_classes=params['num_classes'])
p_eval, h_eval, y_eval = load_data('input/{}/dev.csv'.format(file), data_size=None,maxlen=params['input_shapes'][0][0])
x_eval = [p_eval,h_eval]
y_eval = tf.keras.utils.to_categorical(y_eval,num_classes=params['num_classes'])
p_test, h_test, y_test = load_data('input/{}/test.csv'.format(file), data_size=None,maxlen=params['input_shapes'][0][0])
x_test = [p_test,h_test]
y_test = tf.keras.utils.to_categorical(y_test,num_classes=params['num_classes'])
model = backend.build()
model.compile(
      loss='categorical_crossentropy', 
      optimizer='adam', 
      metrics=['accuracy']
      )
print(model.summary())

earlystop = keras.callbacks.EarlyStopping(
      monitor='val_accuracy', 
      patience=4, 
      verbose=2, 
      mode='max'
      )
bast_model_filepath = './output/best_emis_model.h5' 
checkpoint = keras.callbacks.ModelCheckpoint(
      bast_model_filepath, 
      monitor='val_accuracy', 
      verbose=1, 
      save_best_only=True,
      mode='max'
      )

def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.1

# 定义LearningRateScheduler回调函数
lr_scheduler_callback = keras.callbacks.LearningRateScheduler(lr_scheduler)

model.fit(
      x=x, 
      y=y, 
      batch_size=64, 
      epochs=15, 
      validation_data=(x_eval, y_eval), 
      shuffle=True, 
      callbacks=[earlystop,checkpoint,lr_scheduler_callback]
      )  
loss, acc = model.evaluate(
    x=x_test, 
    y=y_test, 
    batch_size=128, 
    verbose=1
    )
print("Test loss:",loss, "Test accuracy:",acc)


 