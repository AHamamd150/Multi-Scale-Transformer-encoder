import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#import warnings
#warnings.filterwarnings("ignore")
#from input import *
#from source.prepare_files import *
from source.transformer import *
from source.train import *
import tensorflow as tf
import time
import keras
from tensorflow.keras import layers
import sys
import sklearn 
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
## check for a GPU
if not tf.test.gpu_device_name():
   warnings.warn('No GPU found.....')
#  sys.exit()
else:
   print('Default GPU device :{}'.format(tf.test.gpu_device_name()))
#############################
print('###============================================###')
print('This is a demo version of transformer model to analysis files in /data directory')
print('###============================================###')
time.sleep(10)

k_event = 20_000 
test_ratio = 0.2 
attention_mask =True 
num_classes=2  
batch_size= 50  
epoch = 3     
mlp_units = [128, 64,32]  
dropout_rate = 0.2 
lr = 0.001 
compute_ROC = True 
sig_file = 'data/sig_J_TP1.npz'
bkg_file = 'data/bkg1_J_TP1.npz'
num_heads = 5  
hidden_dim= 5    
num_transformers= 3 
n_constit = 50    
n_channels = 7     
#################################
signal=np.load(sig_file,allow_pickle=True)['arr_0'][:k_event]
background =np.load(bkg_file,allow_pickle=True)['arr_0'][:k_event]
signal =  signal[:,:n_constit,:n_channels ] 
background = background[:,:n_constit,:n_channels ]
print('###============================================###')
print(f'''Shape of the signal= {signal.shape}
Shape of the background= {background.shape}''')
print('###============================================###')
time.sleep(10)
x1_data = np.concatenate((signal, background))
y1_data = np.array([1]*len(signal)+[0]*len(background))
x_data,y_data= sklearn.utils.shuffle(x1_data, y1_data) # shuffle both 
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data,shuffle=True, test_size=test_ratio)
x_train= np.array(X_train).astype('float32')
x_test= np.array(X_test).astype('float32')
print('###============================================###')
print(f'''Shape of the training set= {x_train.shape}
Shape of the test set= {x_test.shape}''')
print('###============================================###')
time.sleep(10)

model_part = create_Part_classifier()
model_part.summary()
#tf.keras.utils.plot_model(model_part,show_shapes=True,expand_nested=True,to_file="model_transformer.png")
time.sleep(10)
training_loop(model_part,x_train,y_train,epochs=epoch,batch_size=batch_size)
test_acc(model_part,X_test,y_test)
time.sleep(10)
####
if compute_ROC:
    import matplotlib.pyplot as plt
    score=model_part.predict(X_test);
    fpr, tpr, _ = roc_curve(y_test,score[:,0]);
    plt.figure(figsize=(6,6))
    plt.plot([0,1],[0,1],'k--',linewidth=1);
    plt.plot(tpr,fpr,linewidth=2,label=r'$AUC$= {:.2f}%'.format(float(auc(tpr, fpr))*100));
    plt.xlabel(r'True positive rate',fontsize=25,c='k');
    plt.ylabel(r'False positive rate',fontsize=25,c='k');
    plt.grid(linestyle='--',c='k')
    plt.legend(loc='best',fontsize=25);
    plt.tick_params(axis='both',labelsize=25)
    #plt.yscale('log')
    plt.tight_layout()
    plt.show()
