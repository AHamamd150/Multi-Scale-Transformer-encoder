import os
#import warnings
#warnings.filterwarnings("ignore")
from input import *
#from prepare_files import *
from transformer import *
from train import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import keras
from tensorflow.keras import layers
import sys
import sklearn
from sklearn.model_selection import train_test_split
## check for a GPU
#if not tf.test.gpu_device_name():
#   warnings.warn('No GPU found.....')
#  sys.exit()
#else:
#   print('Default GPU device :{}'.format(tf.test.gpu_device_name()))

#####

#prepare_inputs(sig_dir,bkg_dir,outdir)
#print('files stored in: %s' %(outdir))
signal=np.load(outdir+'signal.npz',allow_pickle=True)['arr_0']
background =np.load(outdir+'/background.npz',allow_pickle=True)['arr_0']
signal =  signal[:,:n_constit,:n_channels ]
background = background[:,:n_constit,:n_channels ]

print(f'''Shape for the signal= {signal.shape}
Shape for the background= {background.shape}''')

k = 15_000
x1_data = np.concatenate((signal[:k], background[:k]))
y1_data = np.array([1]*len(signal[:k])+[0]*len(background[:k]))
x_data,y_data= sklearn.utils.shuffle(x1_data, y1_data) # shuffle both 
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data,shuffle=True, test_size=0.2)
x_train= np.array(X_train).astype('float32')
x_test= np.array(X_test).astype('float32')

model_part = create_Part_classifier()
model_part.summary()
training_loop(model_part,x_train,y_train,epochs=epoch,batch_size=batch_size)


