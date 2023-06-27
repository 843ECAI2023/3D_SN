#encoding=utf8
import keras
from keras.models import Model
import tensorflow as tf 
import numpy as np 
import sys
sys.path.append( '../')
from engine.base_model import BaseModel
from engine.modules import Spatial_Attention,Feature_Attention,Adapted_Feature_Extraction,RFM
from tensorflow.keras import layers


class Three_D_SN(BaseModel):

    def build(self):
        """
        Build the model.
        """
        inputs = self._make_inputs()

        # ---------- Embedding layer ---------- #
        embedding = self.make_embedding_layer()
        encodings = embedding(inputs)
        afe = Adapted_Feature_Extraction(encodings)
        sa = Spatial_Attention(afe)
        fa = Feature_Attention(afe)
        
        mul_a_atten = keras.layers.Lambda(lambda x: x[0]*x[1])([sa[0], fa[0]])
        mul_b_atten = keras.layers.Lambda(lambda x: x[0]*x[1])([sa[1], fa[1]])
        
        rfm = RFM([mul_a_atten,mul_b_atten])
        
        x=layers.concatenate([rfm[0],rfm[1],layers.multiply()([rfm[0],rfm[1]])],axis=-1)
        
        x = layers.GlobalAveragePooling2D()(x)
        
        # ---------- Classification layer ---------- #
        mlp = self._make_multi_layer_perceptron_layer()(x)
        mlp = keras.layers.Dropout(
            rate=self._params['dropout_rate'])(mlp)

        prediction = self._make_output_layer()(mlp)

        model = Model(inputs=inputs, outputs=prediction)        
        
        return model

        

