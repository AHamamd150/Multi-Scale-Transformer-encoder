model = 1 # Model=1: is a normal transformer model 
# if model= 2: is a three-modal transformer with cross attention as described in Arxiv.xxx
# if mode=3:  is a two-modal transformer. Each modal has a transfrmer layers with self attention heads and the output from each modal fed to cross-attention layers.
### Common Hyper-parameters ######
k_event = 50_000 ### Total number of used events for each signal and background
test_ratio = 0.2 ## the ratio of the considered test data from the full data set
attention_mask =False ## Attention mask to be used
num_classes=2  ## Number of the output classe 
batch_size= 50  ## Training batch size
epoch = 3       ## Number of epochs
mlp_units = [128, 64]  # number of neurons of the final MLP. if you want to add more FC layers just increase the list, e.g. [128,64,32] accounts for three FC layers
dropout_rate = 0.2 # Dropout rate to be used
lr = 0.001 #Learning rate for ADAM otpimizer
compute_ROC = True #Compute the ROC and the AUC

################################################
##    If you use model 1, please fill this part only           ########
################################################
sig_file = 'sig.npz'
bkg_file = 'bkg.npz'

num_heads = 5   ## Number of the self attenion heads
hidden_dim= 5    ## Hidden dimension of the multi-attention heads. Usually is the same as the number of heads. Please look at the tensorflow manual for multi-heads self attention layer
num_transformers= 3 ## Number of reapted transformer layers
n_constit = 50     #Number of the particle tokens
n_channels = 7     # Number of the features of each particle token
####################################################
################################################
##    If you use model 2, please fill this part only           ########
################################################
sig_file_1 = 'sig_1.npz'
sig_file_2 = 'sig_2.npz'
sig_file_3 = 'sig_3.npz'

bkg_file_1 = 'bkg_1.npz'
bkg_file_2 = 'bkg_2.npz'
bkg_file_3 = 'bkg_3.npz'
######################################
## paremters of the first transformer#
######################################
num_heads_1 = 5
num_transformers_1= 3
n_constit_1 = 50
n_channels_1 = 3
#######################################
## paremters of the second transformer#
#######################################
num_heads_2 = 7
num_transformers_2= 2
n_constit_2 = 50
n_channels_2 = 3
#######################################
## paremters of the third transformer#
#######################################
num_heads_3 = 3
num_transformers_3= 4
n_constit_3 = 5
n_channels_3 = 8
###########################################
## paremters of the transformer with cross-attnetion heads#
###########################################
num_heads_cross = 3
num_transformers_cross= 1
#############################


####################################################
################################################
##    If you use model 3, please fill this part only           ########
################################################

sig_file_1 = 'sig_1.npz'
sig_file_2 = 'sig_2.npz'

bkg_file_1 = 'bkg_1.npz'
bkg_file_2 = 'bkg_2.npz'
######################################
## paremters of the first transformer#
######################################
num_heads_1 = 5
num_transformers_1= 3
n_constit_1 = 50
n_channels_1 = 3
#######################################
## paremters of the second transformer#
#######################################
num_heads_2 = 7
num_transformers_2= 2
n_constit_2 = 50
n_channels_2 = 3
###########################################
## paremters of the transformer with cross-attnetion heads#
###########################################
num_heads_cross = 3
num_transformers_cross= 1
#############################


