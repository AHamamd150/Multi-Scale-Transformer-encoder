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

input_shape_part_1 = (n_constit_1,n_channels_1)
mlp_head_units_1 = [64,n_channels_1]

input_shape_part_1 = (n_constit_1,n_channels_1)
mlp_head_units_1 = [64,n_channels_1]

input_shape_part_2 = (n_constit_2,n_channels_2)
mlp_head_units_2 = [64,n_channels_2]

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


class transformer_encoder_layer_MHSA:
    def __init__(self,heads, dropout,mask,mlp_units,hidden_dim):
        self.heads = heads
        self. dropout=dropout
        self.mask = mask
        self.mlp_units = mlp_units
        self.hidden_dim=hidden_dim

    
    def layer(self,x):     
        MHA = layers.MultiHeadAttention(num_heads=self.heads, key_dim=self.hidden_dim, dropout=self.dropout)    
        x1 = layers.LayerNormalization()(x)
        if masked:
            attention_output,weights = MHA(x1, x1, attention_mask = self.mask,return_attention_scores=True)
        else:
            attention_output,weights = MHA(x1, x1, attention_mask = None,return_attention_scores=True)
        x2 = layers.Add()([attention_output, x])
        x3 = layers.LayerNormalization()(x2)
        x3 = mlp(x3, hidden_units=self.mlp_units, dropout_rate=self.dropout)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])
        return encoded_patches


class transformer_encoder_layer_MHCA:
    def __init__(self,heads, dropout,mask,mlp_units,hidden_dim):
        self.heads = heads
        self. dropout=dropout
        self.mask = mask
        self.mlp_units = mlp_units
        self.hidden_dim=hidden_dim

    
    def layer(self,x,y):     
        MHA = layers.MultiHeadAttention(num_heads=self.heads, key_dim=self.hidden_dim, dropout=self.dropout)    
        x1 = layers.LayerNormalization()(x)
        y1 = layers.LayerNormalization()(y)
        attention_output,weights = MHA(x1, y1, attention_mask = None,return_attention_scores=True)
        x2 = layers.Add()([attention_output, x])
        x3 = layers.LayerNormalization()(x2)
        x3 = mlp(x3, hidden_units=self.mlp_units, dropout_rate=self.dropout)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])
        return encoded_patches


    
def create_Part_classifier():
    inputs = layers.Input(shape=input_shape_part) 
    if attention_mask:
        attentionmask= masked_fill(inputs,2,-np.inf,inputs.shape[1])
    else:
        attentionmask= None
    transformer_encoder = transformer_encoder_layer_MHSA(num_heads,dropout_rate,attentionmask,mlp_head_units,inputs.shape[-1])
    encoded_patches = layers.LayerNormalization()(inputs)    
    for _ in range(num_transformers):
        encoded_patches = transformer_encoder.layer(encoded_patches)
       
    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(dropout_rate)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_units, dropout_rate=dropout_rate)
    logits = layers.Dense(num_classes,activation='softmax')(features)
    model = keras.Model(inputs=inputs, outputs=logits)
    return model
    
    
    
def create_Part_classifier_2():
    inputs_0 = layers.Input(shape=input_shape_part_1)
    inputs_1 = layers.Input(shape=input_shape_part_1)
    inputs_2 = layers.Input(shape=input_shape_part_2)
    inputs_3 = layers.Input(shape=input_shape_part_2)     
    
    if masked:
        attention_mask_1= masked_fill(inputs_1,2,-np.inf,inputs_1.shape[1])
        attention_mask_2= masked_fill(inputs_2,2,-np.inf,inputs_2.shape[1])
    else:
        attention_mask_1= None
        attention_mask_2= None
        
    transformer_encoder_0 = transformer_encoder_layer_MHSA(num_heads_1,0.1,attention_mask_1,mlp_head_units_1,inputs_0.shape[-1])
    encoded_patches_0 = layers.LayerNormalization()(inputs_0) 
    transformer_encoder_1 = transformer_encoder_layer_MHSA(num_heads_1,0.1,attention_mask_1,mlp_head_units_1,inputs_1.shape[-1])
    encoded_patches_1 = layers.LayerNormalization()(inputs_1)
    transformer_encoder_2 = transformer_encoder_layer_MHSA(num_heads_2,0.1,attention_mask_2,mlp_head_units_2,inputs_2.shape[-1])
    encoded_patches_2 = layers.LayerNormalization()(inputs_2)    
    transformer_encoder_3 = transformer_encoder_layer_MHCA(num_heads_3,0.1,attention_mask_2,mlp_head_units_2,inputs_2.shape[-1])
    encoded_patches_3 = layers.LayerNormalization()(inputs_3)
    
    
    for _ in range(num_transformers_1):
        encoded_patches_0 = transformer_encoder_0.layer(encoded_patches_0)
    for _ in range(num_transformers_1):
        encoded_patches_1 = transformer_encoder_1.layer(encoded_patches_1)
    for _ in range(num_transformers_2):
        encoded_patches_2 = transformer_encoder_2.layer(encoded_patches_2)
    encoded_patches_add = layers.Add()([encoded_patches_0, encoded_patches_1])    
    encoded_patches_3 = transformer_encoder_3.layer(encoded_patches_2,encoded_patches_add)
    for _ in range(num_transformers_3-1):
        encoded_patches_3 = transformer_encoder_3.layer(encoded_patches_3,encoded_patches_add)
         
    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches_3)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.1)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_units, dropout_rate=0.1)
    logits = layers.Dense(num_classes,activation='softmax')(features)
    model = keras.Model(inputs=[inputs_0,inputs_1,inputs_2], outputs=logits)
    return model
    
    
def create_Part_classifier_3():
    inputs_1 = layers.Input(shape=input_shape_part_1)
    inputs_2 = layers.Input(shape=input_shape_part_2)
    inputs_3 = layers.Input(shape=input_shape_part_2)
    
    if attention_mask:
        attention_mask_1= masked_fill(inputs_1,2,-np.inf,inputs_1.shape[1])
        attention_mask_2= masked_fill(inputs_2,2,-np.inf,inputs_2.shape[1])
    else:
        attention_mask_1= None
        attention_mask_2= None
        
    transformer_encoder_1 = transformer_encoder_layer_MHSA(num_heads_1,dropout_rate,attention_mask_1,mlp_head_units_1,inputs_1.shape[-1])
    encoded_patches_1 = layers.LayerNormalization()(inputs_1)
    transformer_encoder_2 = transformer_encoder_layer_MHSA(num_heads_2,dropout_rate,attention_mask_2,mlp_head_units_2,inputs_2.shape[-1])
    encoded_patches_2 = layers.LayerNormalization()(inputs_2)    
    transformer_encoder_3 = transformer_encoder_layer_MHCA(num_heads_cross,dropout_rate,attention_mask_2,mlp_head_units_2,inputs_3.shape[-1])
    encoded_patches3 = layers.LayerNormalization()(inputs_3)
    
    

    for _ in range(num_transformers_2):
        encoded_patches_1 = transformer_encoder_1.layer(encoded_patches_1)
    for _ in range(num_transformers_2):
        encoded_patches_2 = transformer_encoder_2.layer(encoded_patches_2)
    encoded_patches3 = transformer_encoder_3.layer(encoded_patches_2,encoded_patches_1) 
    for _ in range(num_transformers_cross-1):
        encoded_patches3 = transformer_encoder_3.layer(encoded_patches_2,encoded_patches_1) 
         
    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches3)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(dropout_rate)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_units, dropout_rate=dropout_rate)
    logits = layers.Dense(num_classes,activation='softmax')(features)
    model = keras.Model(inputs=[inputs_1,inputs_2], outputs=logits)
    return model        
