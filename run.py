import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#import warnings
#warnings.filterwarnings("ignore")
from input import *
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
if model ==1:
  print('###================================================###')
  print('Model-1: Transformer Encoder with self-attention heads ')
  print('###================================================###')
  time.sleep(10)
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
###############################################
###############################################
###############################################
if model ==2:
  print('###================================================###')
  print('Model-2: Multi-modal transformer network with cross attention ')
  print('###================================================###')
  time.sleep(10)
  signal_1=np.load(sig_file_1,allow_pickle=True)['arr_0'][:k_event]
  signal_2=np.load(sig_file_2,allow_pickle=True)['arr_0'][:k_event]
  signal_3=np.load(sig_file_3,allow_pickle=True)['arr_0'][:k_event]
  
  background_1 =np.load(bkg_file_1,allow_pickle=True)['arr_0'][:k_event]
  background_2 =np.load(bkg_file_2,allow_pickle=True)['arr_0'][:k_event]
  background_3 =np.load(bkg_file_3,allow_pickle=True)['arr_0'][:k_event]
  
  signal_1 =  signal_1[:,:n_constit_1,:n_channels_1 ] 
  background_1 = background_1[:,:n_constit_1,:n_channels_1 ]

  signal_2 =  signal_2[:,:n_constit_2,:n_channels_2 ] 
  background_2 = background_2[:,:n_constit_2,:n_channels_2]
  
  signal_3 =  signal_3[:,:n_constit_3,:n_channels_3 ] 
  background_3 = background_3[:,:n_constit_3,:n_channels_3 ]
  
  print('###============================================###')
  print(f'''Shape of the signal_1= {signal_1.shape}
Shape of the background_1= {background_1.shape}''')
  print('###============================================###')
  print('###============================================###')
  print(f'''Shape of the signal_2= {signal_2.shape}
Shape of the background_2= {background_2.shape}''')
  print('###============================================###')
  print('###============================================###')
  print(f'''Shape of the signal_3= {signal_3.shape}
Shape of the background_3= {background_3.shape}''')
  print('###============================================###')
  time.sleep(10)
  x0_data = np.concatenate((signal_1,background_1))
  x1_data = np.concatenate((signal_2, background_2))
  x2_data = np.concatenate((signal_3, background_3))
  y_data = np.array([1]*len(signal_1)+[0]*len(background_1))
  x0_data,x1_data,x2_data,y_data = sklearn.utils.shuffle(x0_data,x1_data,x2_data,y_data)
  ### Split the data to train and test samples ####
  k_split = int(np.floor((1-test_ratio)* len(y_data)))
  x0_train = x0_data[:k_split]
  x1_train = x1_data[:k_split]
  x2_train = x2_data[:k_split]
  y_train = y_data[:k_split]

  x0_test = x0_data[k_split:]
  x1_test = x1_data[k_split:]
  x2_test = x2_data[k_split:]
  y_test = y_data[k_split:]
  print('###============================================###')
  print(f'''Shape of the first training set= {x0_train.shape}
Shape of the first test set= {x0_test.shape}''')
  print('###============================================###')
  print('###============================================###')
  print(f'''Shape of the second training set= {x1_train.shape}
Shape of the second test set= {x1_test.shape}''')
  print('###============================================###')
  print('###============================================###')
  print(f'''Shape of the thirs training set= {x2_train.shape}
Shape of the third test set= {x2_test.shape}''')
  print('###============================================###')
  time.sleep(10)
  model_part = create_Part_classifier_2()
  model_part.summary()
  #tf.keras.utils.plot_model(model_part,show_shapes=True,expand_nested=True,to_file="model_transformer.png")
  time.sleep(10)
  training_loop_2(model_part,x0_train,x1_train,x2_train,y_train,epochs=epoch,batch_size=batch_size)
  test_acc_2(model_part,x0_test,x1_test,x2_test,y_test)
  if compute_ROC:
      import matplotlib.pyplot as plt
      score=model_part.predict([x0_test,x1_test,x2_test]);
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
      
     
     
  
  
###############################################
###############################################
###############################################
if model ==3:
  print('###================================================###')
  print('Model-3: Two-modal transformer network with cross attention ')
  print('###================================================###')
  time.sleep(10)
  signal_1=np.load(sig_file_1,allow_pickle=True)['arr_0'][:k_event]
  signal_2=np.load(sig_file_2,allow_pickle=True)['arr_0'][:k_event]

  
  background_1 =np.load(bkg_file_1,allow_pickle=True)['arr_0'][:k_event]
  background_2 =np.load(bkg_file_2,allow_pickle=True)['arr_0'][:k_event]

  
  signal_1 =  signal_1[:,:n_constit_1,:n_channels_1 ] 
  background_1 = background_1[:,:n_constit_1,:n_channels_1 ]

  signal_2 =  signal_2[:,:n_constit_2,:n_channels_2 ] 
  background_2 = background_2[:,:n_constit_2,:n_channels_2]
    
  print('###============================================###')
  print(f'''Shape of the signal_1= {signal_1.shape}
Shape of the background_1= {background_1.shape}''')
  print('###============================================###')
  print('###============================================###')
  print(f'''Shape of the signal_2= {signal_2.shape}
Shape of the background_2= {background_2.shape}''')
  print('###============================================###')
  print('###============================================###')
  time.sleep(10)
  x1_data = np.concatenate((signal_1, background_1))
  x2_data = np.concatenate((signal_2, background_2))
  y_data = np.array([1]*len(signal_1)+[0]*len(background_1))
  x1_data,x2_data,y_data = sklearn.utils.shuffle(x1_data,x2_data,y_data)
  ### Split the data to train and test samples ####
  k_split = int(np.floor((1-test_ratio)* len(y_data)))
  x1_train = x1_data[:k_split]
  x2_train = x2_data[:k_split]
  y_train = y_data[:k_split]
  x1_test = x1_data[k_split:]
  x2_test = x2_data[k_split:]
  y_test = y_data[k_split:]
  print('###============================================###')
  print(f'''Shape of the first training set= {x1_train.shape}
Shape of the first test set= {x1_test.shape}''')
  print('###============================================###')
  print('###============================================###')
  print(f'''Shape of the second training set= {x2_train.shape}
Shape of the second test set= {x2_test.shape}''')
  print('###============================================###')
  print('###============================================###')
  time.sleep(10)
  model_part = create_Part_classifier_3()
  model_part.summary()
  #tf.keras.utils.plot_model(model_part,show_shapes=True,expand_nested=True,to_file="model_transformer.png")
  time.sleep(10)
  training_loop_3(model_part,x1_train,x2_train,y_train,epochs=epoch,batch_size=batch_size)
  test_acc_3(model_part,x1_test,x2_test,y_test)
  if compute_ROC:
      import matplotlib.pyplot as plt
      score=model_part.predict([x1_test,x2_test]);
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
