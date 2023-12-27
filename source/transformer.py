import os
import numpy as np
from input import *
import tensorflow as tf
import keras
from tensorflow.keras import layers
import sys
sys.path.append("../")
from input import *
###########################
input_shape_part = (n_constit,n_channels)
mlp_head_units = [64,n_channels]
###########################
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
def masked_fill(tensor, idx,value,n_constit):
    pad_mask = tf.math.equal(tensor[:,:,idx],0)
    mask = tf.where(pad_mask, tf.cast(tf.fill(tf.shape(tensor[...,-1]), value),tf.float32), tensor[:,:,idx])
    mask = tf.repeat(mask[:,tf.newaxis],n_constit,axis=1)
    return mask


def transformer_encoder_layer(x,heads,dim, dropout,mask,mlp_units):
    MHA = layers.MultiHeadAttention(num_heads=heads, key_dim=dim, dropout=dropout)
    x1 = layers.LayerNormalization()(x)
    attention_output = MHA(x1, x1, attention_mask = mask) 
    x2 = layers.Add()([attention_output, x])
    x3 = layers.LayerNormalization()(x2)
    x3 = mlp(x3, hidden_units=mlp_units, dropout_rate=dropout)
    # Skip connection 2.
    encoded_patches = layers.Add()([x3, x2])
    return encoded_patches

    
def create_Part_classifier():
    inputs = layers.Input(shape=input_shape_part) 
    if attention_mask:
        attentionmask= masked_fill(inputs,2,-np.inf,inputs.shape[1])
    else:
        attentionmask= None
    for _ in range(num_transformers):
        encoded_patches = transformer_encoder_layer(inputs,num_heads, hidden_dim,dropout_rate,attentionmask,mlp_head_units)
       
    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(dropout_rate)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_units, dropout_rate=dropout_rate)
    logits = layers.Dense(num_classes,activation='softmax')(features)
    model = keras.Model(inputs=inputs, outputs=logits)
    return model
