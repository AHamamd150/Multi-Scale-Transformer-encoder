# Multi-Scale Cross-Attention Transformer Encoder for Event Calssification

 &emsp; Transformer encoder model that incorporate different scale information via multi-modal network with self and cross-attention layers. The packge based on [arXiv:XXXX [hep-ph]](https://arxiv.org/abs/XXX). The current version of the package is very geeneric and span the three configurations  as the following:

 * model_1: Transformer model with single input and self-attention heads.
 * model_2: Transformer model with three inputs, each input followed by transformer layers with self-attention heads and the output is fed to transformer layers with cross-attneion heads.
 * model_3: Transformer model with two inputs, each followed by transformer layers with self-attneion heads and the output is fed to transformer layers with cross-attention heads. 
 __________
## $~~~~~~~~~~~$  Table of content

$~~~~~~~~~~~$ $~~~~~~~~~~~$ [1. Introduction ](#Introduction)

$~~~~~~~~~~~$ $~~~~~~~~~~~$  [2. Requirements ](#Requirements)

$~~~~~~~~~~~$ $~~~~~~~~~~~$  [4. Get started ](#start)

$~~~~~~~~~~~$ $~~~~~~~~~~~$  [3. Package structure ](#structure)

$~~~~~~~~~~~$ $~~~~~~~~~~~$  [6. Flow Chart of the networks](#chart)
________________
<a name="Introduction"></a>
## Introduction
&emsp; Information about jet identification provides powerful insights into collision events and can %help separating 
help to separate different physics processes originating these. This information can be extracted from the elementary particles localized inside a jet. Recently, various methods have been used to exploit the substructure of a jet to probe new physics signatures using advanced Machine Learning (ML) techniques. Conversely, using the reconstructed kinematics from the final state jets for event classification spans the full phase space and exhibits large classification performance. Such high-level kinematics (i.e., encoding the global features of the final state particles), possibly together with the knowledge of the properties of (known or assumed) resonant intermediate particles, remains blind to the information encoded inside the final state jets. A possible way to extract information from both jet substructure and global jet kinematics is to concatenate the information extracted from a multi-modal network. However, such a simple concatenation leads to an imbalance of the extracted information, within which the kinematic information generally dominates. We present a novel method for incorporating different-scale information extracted from both global kinematics and substructure of jets via a transformer encoder with a cross-attention layer. The model initially extracts the most relevant information from each dataset individually using self-attention layers before incorporating these using a cross-attention layer. The method demonstrates a larger improvement in classification performance compared to the simple concatenation method.

To assess our results, we analyze the learned information by the transformer layers through the examination of the attention maps of the self- and cross-attention layers. 
Attention maps provide information about the (most) important particles the model focuses on when classifying signal and background events. 
However, they cannot highlight the region in the feature (e.g., phase) space crucial for model classification. For this purpose, we utilize Gradient-weighted Class Activation Mapping (Grad-CAM) to highlight the geometric region in the $\eta-\phi$ (detector) plane where the model focuses on classifying events.
____________________________
<a name="Requirements"></a>
## Requirements
&emsp; To run the package you need python3 with the following modules:
* Numpy
* TensorFlow
* sklearn
* matplotlib

Requirements can be easily installed by `pip3 install module` or the user can use the given `enviroment.yml` to create a conda enviroment.

_____________________________
<a name="start"></a>
## Get start
To run the package, the user has to fill the file `input.py` for the used model. For example, if the user want to use model_1 then the corresponding lines for model_1 only has to be filled while all other inputs are igonred by the code. To run the code, type in the terminal `python3 run.py`.  

The network assumes the signal events in one file  and all the backgrounds in one file in the numpy format ".npz" which can be easily obtained by the command `numpy..savez_compressed()`

A demo version is also provided in which the user can run it to test the package. To run the demo version, type in the terminal `python3 run_demo.py`

_____________________________
<a name="structure"></a>
## Structure of the code
&emsp; The package consists of the following files:
* `input.py` Input file which has to be filled by the user to control the network structure
* `run.py` The run file which takes as input the files in the source directory and the `input.py`
* `run_demo.py` A demo version of the code in wich the user can run it for test.
* `data/` Data directory contains the signal and background files for the demo.
* `source/transformer.py` Source code for the transformer network.
* `source/train.py` Source code for the training and teseting loop for each model.
* `source/Analysis.py` Example of Delphes analysis that the user can consider.
* `source/prepare_files.py` Source code to prepae the input files to the network.

_____________________________
<a name="chart"></a>
## Flow chart of model_1, the user controls the structure of the model from the input file 


![model_transformer_1](https://github.com/AHamamd150/Multi-Scale-Transformer-encoder/assets/68282212/883aa1d8-c62a-4674-82ab-15da7b13d7a8)


## Flow chart of model_2, the user controls the structure of the model from the input file 

![model_transformer_2](https://github.com/AHamamd150/Multi-Scale-Transformer-encoder/assets/68282212/d8ac4043-e36e-4d5b-aaec-8a3c36041183)

## Flow chart of model_3, the user controls the structure of the model from the input file 

![model_transformer_3](https://github.com/AHamamd150/Multi-Scale-Transformer-encoder/assets/68282212/96f84a93-3272-4624-9379-f7fce5fe899b)



