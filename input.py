sig_file = 'data/sig_J_TP1.npz'
bkg_file = 'data/bkg1_J_TP1.npz'
outdir = 'output/'
##Hyper parameteres to be tuned##
k_event = 50_000 ### Total number of used events for each signal and background
test_ratio = 0.2 ## the ratio of the considered test data from the full data set
attention_mask =True ## Attention mask to be used
num_classes=2  ## Number of the output classe 
batch_size= 50  ## Training batch size
epoch = 3       ## Number of epochs
num_heads = 5   ## Number of the self attenion heads
hidden_dim= 5    ## Hidden dimension of the multi-attention heads. Usually is the same as the number of heads. Please look at the tensorflow manual for multi-heads self attention layer
num_transformers= 3 ## Number of reapted transformer layers
n_constit = 50     #Number of the particle tokens
n_channels = 7     # Number of the features of each particle token
mlp_units = [128, 64]  # number of neurons of the final MLP. if you want to add more FC layers just increase the list, e.g. [128,64,32] accounts for three FC layers
dropout_rate = 0.2 # Dropout rate to be used
lr = 0.001 #Learning rate for ADAM otpimizer
##
compute_ROC = True #Compute the ROC and the AUC
save_model = False
