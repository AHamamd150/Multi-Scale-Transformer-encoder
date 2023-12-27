import os,sys
import numpy as np
import tensorflow as tf
import keras
sys.path.append("../")
from input import *
################
loss_func = keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate= lr)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

### Training loop function per training batch 
@tf.function
def train_step(x, y,model):
    with tf.GradientTape() as tape:
        logit = model(x, training=True)
        loss_value = loss_func(y, logit)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_accuracy.update_state(y, logit)
    return loss_value,train_accuracy.result()

### function to test the loss and accuracy per training batch
@tf.function
def test_step(x, y,model):
    val_logit = model(x, training=False)
    test_accuracy.update_state(y, val_logit)
### Function to the test the model accuracy on the model test data    
def test_acc(model,x, y):
    val_logit = model(x, training=False)
    test_accuracy.update_state(y, val_logit)
    return print(f'Test Accuracy:  {test_accuracy.result()*100 :.3f}%')    


def training_loop(model,x_train,y_train,epochs=20,batch_size=512):
    
    ## Lets create the batches first
    train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(x_train.shape[0]).batch(batch_size) 
    #test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_acc_avg = tf.keras.metrics.Mean()
    for epoch in range(epochs):
    # Iterate over the batches of the dataset.
        loss,acc = [],[]
        for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
            
            loss_value,acc_value = train_step(x_batch_train, y_batch_train,model)
            loss.append(loss_value)
            acc.append(acc_value)
            epoch_loss_avg.update_state(loss) 
            epoch_acc_avg.update_state(acc)  
            if step % 1 == 0:
                sys.stdout.write('\r'+'step %s :  loss = %2.5f  accuracy = %2.5f'%((step + 1),float(loss_value),float(acc_value)))
           
        # Display metrics at the end of each epoch.
        tf.print('|   Epoch {:2d}:  Loss (Avg): {:2.5f}  Accuracy (Avg): {:2.5f}'.format(epoch+1,epoch_loss_avg.result(),epoch_acc_avg.result()))  
        # Reset training metrics at the end of each epoch
        train_accuracy.reset_states()
