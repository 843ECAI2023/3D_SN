#encoding=utf8
import abc
import keras
from keras.models import Model
from tensorflow.keras import layers
import tensorflow as tf 
import numpy as np
from transformers import TFBertModel


class BaseModel(object):

    def __init__( self, params):
        """Init."""
        self._params = params

    
    def make_embedding_layer(self,input, embeddings_type='bert', name='embedding',embed_type='char',weights=None,**kwargs):


        if embeddings_type=='bert':
            bert = TFBertModel.from_pretrained('bert-base-uncased',from_pt=True,output_hidden_states=True, output_attentions=True)
            bert.trainable = False

            outputs_p = bert(input_ids=input[0], attention_mask=input[1], token_type_ids=input[2]).hidden_states
            outputs_h = bert(input_ids=input[3], attention_mask=input[4], token_type_ids=input[5]).hidden_states
            embedding =  tf.squeeze(np.array(outputs_p)), tf.squeeze(np.array(outputs_h))

        else:
            if embed_type == "char":
                input_dim = self._params['max_features']
                output_dim = self._params['embed_size']
            else:
                input_dim = self._params['word_max_features']
                output_dim = self._params['word_embed_size']

            embedding= keras.layers.Embedding(
                input_dim = input_dim,
                output_dim = output_dim,
                trainable = False,
                name = name,
                weights = weights,
                **kwargs)
    

        return embedding
    
    # def make_bert_embedding_layer(self,input,name='embedding',embed_type='char',**kwargs):

    #     def init_embedding(weights=None):
    #         if embed_type == "char":
    #             input_dim = self._params['max_features']
    #             output_dim = self._params['embed_size']
    #         else:
    #             input_dim = self._params['word_max_features']
    #             output_dim = self._params['word_embed_size']

    #         return keras.layers.Embedding(
    #             input_dim = input_dim,
    #             output_dim = output_dim,
    #             trainable = False,
    #             name = name,
    #             weights = weights,
    #             **kwargs)

    #     if embed_type == "char":
    #         embed_weights = self._params['embedding_matrix']
    #     else:
    #         embed_weights = self._params['word_embedding_matrix']

    #     if embed_weights == []:
    #         embedding = init_embedding()
    #     else:
    #         embedding = init_embedding(weights = [embed_weights])

    #     return embedding

    def _make_multi_layer_perceptron_layer(self) -> keras.layers.Layer:
        # TODO: do not create new layers for a second call
        def _wrapper(x):
            activation = self._params['mlp_activation_func']
            for _ in range(self._params['mlp_num_layers']):
                x = keras.layers.Dense(self._params['mlp_num_units'],
                                       activation=activation)(x)
            return keras.layers.Dense(self._params['mlp_num_fan_out'],
                                      activation=activation)(x)

        return _wrapper

    # def _make_inputs(self) -> list:
    #     input_left = keras.layers.Input(
    #         name='text_left',
    #         shape=self._params['input_shapes'][0]
    #     )
    #     input_right = keras.layers.Input(
    #         name='text_right',
    #         shape=self._params['input_shapes'][1]
    #     )
    #     return [input_left, input_right]
    
    def _make_inputs(self): 
        input_ids_p = layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
        attention_mask_p = layers.Input(shape=(None,), dtype=tf.int32, name="attention_mask")
        token_type_ids_p = layers.Input(shape=(None,), dtype=tf.int32, name="token_type_ids")
        input_ids_h = layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
        attention_mask_h = layers.Input(shape=(None,), dtype=tf.int32, name="attention_mask")
        token_type_ids_h = layers.Input(shape=(None,), dtype=tf.int32, name="token_type_ids")
        
        return input_ids_p, attention_mask_p, token_type_ids_p, input_ids_h, attention_mask_h, token_type_ids_h 
          

    def _make_output_layer(self) -> keras.layers.Layer:
        """:return: a correctly shaped keras dense layer for model output."""
        task = self._params['task']
        if task == "Classification":
            return keras.layers.Dense(self._params['num_classes'], activation='softmax')
        elif task == "Ranking":
            return keras.layers.Dense(1, activation='linear')
        else:
            raise ValueError(f"{task} is not a valid task type."
                             f"Must be in `Ranking` and `Classification`.")

    def _create_base_network(self):

        def _wrapper(x):

            pass

        return _wrapper

    def build(self):
        """
        Build model structure.
        """
        pass
    
        return model



