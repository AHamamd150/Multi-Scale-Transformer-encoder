import os
#import warnings
#warnings.filterwarnings("ignore")
from input import *
#from source.prepare_files import *
from source.transformer import *
from source.train import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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

#####
signal=np.load(sig_file,allow_pickle=True)['arr_0'][:k_event]
background =np.load(bkg_file,allow_pickle=True)['arr_0'][:k_event]
signal =  signal[:,:n_constit,:n_channels ]
background = background[:,:n_constit,:n_channels ]

print(f'''Shape of the signal= {signal.shape}
Shape of the background= {background.shape}''')
time.sleep(10)
x1_data = np.concatenate((signal, background))
y1_data = np.array([1]*len(signal)+[0]*len(background))
x_data,y_data= sklearn.utils.shuffle(x1_data, y1_data) # shuffle both 
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data,shuffle=True, test_size=test_ratio)
x_train= np.array(X_train).astype('float32')
x_test= np.array(X_test).astype('float32')

print(f'''Shape of the training set= {x_train.shape}
Shape of the test set= {x_test.shape}''')
time.sleep(10)

model_part = create_Part_classifier()
model_part.summary()
time.sleep(10)
training_loop(model_part,x_train,y_train,epochs=epoch,batch_size=batch_size)
test_acc(model_part,X_test,y_test)
time.sleep(10)
if save_model:
  model_part.save(outdir+'/mode.h5')

#############################
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

