
import numpy as np
import pandas as pd
from bert4keras.backend import keras, set_gelu, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Dropout, Dense

set_gelu('tanh')  

maxlen = 64
batch_size = 8
config_path = 'E:/bert_pretrain_model/multi_cased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'E:/bert_pretrain_model/multi_cased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'E:/bert_pretrain_model/multi_cased_L-12_H-768_A-12/vocab.txt'
# config_path = 'E:/bert_weight_files/albert_large_google_zh/albert_config.json'
# checkpoint_path = 'E:/bert_weight_files/albert_large_google_zh/albert_model.ckpt'
# dict_path = 'E:/bert_weight_files/albert_large_google_zh/vocab.txt'

def load_data(filename):
    df = pd.read_csv(filename,header=0,encoding='utf8')
    f = df.values
    D = []
    # with open(filename, encoding='utf-8') as f:
    for i,l in enumerate(f):
        # print(l)
        text1, text2, label = l#.strip().split(',')
        D.append((text1, text2, int(label)))
    return D


train_data = load_data('../input/train.csv')
valid_data = load_data('../input/dev.csv')
test_data = load_data('../input/test.csv')


tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text1, text2, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                text1, text2, max_length=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []



bert = build_transformer_model(
    model='bert',
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    with_pool=True,
    return_keras_model=False,
)

output = Dropout(rate=0.1)(bert.model.output)
output = Dense(
    units=2, activation='softmax', kernel_initializer=bert.initializer
)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(2e-5),  
    # optimizer=PiecewiseLinearLearningRate(Adam(5e-5), {10000: 1, 30000: 0.1}),
    metrics=['accuracy'],
)


train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model.weights')
        test_acc = evaluate(test_generator)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )


evaluator = Evaluator()
model.fit_generator(
    train_generator.forfit(),
    steps_per_epoch=len(train_generator),
    epochs=10,
    callbacks=[evaluator]
)

model.load_weights('best_model.weights')
print(u'final test acc: %05f\n' % (evaluate(test_generator)))
