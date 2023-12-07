delphes_dir = '/Users/hammad/work/pheno/delphes/'
sig_dir = '/Users/hammad/work/data/dark_photon/TP1/'
bkg_dir = '/Users/hammad/work/data/dark_photon/bkg_TP1/bkg1/'
outdir = '/Users/hammad/work/transformer_package/out/'
n_cont = 100
##Hyper parameteres to be tuned
num_classes=2
batch_size= 50
epoch = 5
num_heads = 5
hidden_dim= 5
num_transformers= 3
n_constit = 50
n_channels = 7
input_shape_part = (n_constit,n_channels)
mlp_units = [128, 64]
mlp_head_units = [64,n_channels]

