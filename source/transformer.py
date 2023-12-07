import os
import numpy as np
from input import *
import tensorflow as tf
import keras
from tensorflow.keras import layers
import sys

###########################
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
def masked_fill(tensor, idx,value):
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
    attention_mask_= masked_fill(inputs,2,-np.inf)
    for _ in range(num_transformers):
        encoded_patches = transformer_encoder_layer(inputs,num_heads, hidden_dim,0.1,attention_mask_,mlp_head_units)
       
    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.2)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_units, dropout_rate=0.2)
    logits = layers.Dense(num_classes,activation='softmax')(features)
    model = keras.Model(inputs=inputs, outputs=logits)
    return model
