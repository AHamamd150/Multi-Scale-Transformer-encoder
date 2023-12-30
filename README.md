# Multi-Scale Cross-Attention Transformer Encoder for Event Calssification

 &emsp; Transformer encoder model that incorporate different scale information via multi-modal network with self and cross-attention layers. The packge based on [arXiv:2207.09959 [hep-ph]](https://arxiv.org/abs/2207.09959). The current version of the package is very geeneric and span the three configurations  as the following

 * model_1: Transformer model with single input and self-attention heads.
 * model_2: Transformer model with three inputs, each input followed by transformer layers with self-attention heads and the output is fed to transformer layers with cross-attneion heads.
 * model_3: Transformer model with two inputs, each followed by transformer layers with self-attneion heads and the output is fed to transformer layers with cross-attention heads. 
 __________
## $~~~~~~~~~~~$  Table of content

$~~~~~~~~~~~$ $~~~~~~~~~~~$ [1. Introduction ](#Introduction)

$~~~~~~~~~~~$ $~~~~~~~~~~~$  [2. Requirements ](#Requirements)

$~~~~~~~~~~~$ $~~~~~~~~~~~$  [3. Package structure ](#structure)

$~~~~~~~~~~~$ $~~~~~~~~~~~$  [4. Get started ](#start)

$~~~~~~~~~~~$ $~~~~~~~~~~~$ [5. Work flow](#flow)

$~~~~~~~~~~~$ $~~~~~~~~~~~$  [6. Toy examples](#toy)
________________
<a name="Introduction"></a>
## Introduction
&emsp; Information about jet identification provides powerful insights into collision events and can %help separating 
help to separate different physics processes originating these. This information can be extracted from the elementary particles localized inside a jet. Recently, various methods have been used to exploit the substructure of a jet to probe new physics signatures using advanced Machine Learning (ML) techniques. Conversely, using the reconstructed kinematics from the final state jets for event classification spans the full phase space and exhibits large classification performance. Such high-level kinematics (i.e., encoding the global features of the final state particles), possibly together with the knowledge of the properties of (known or assumed) resonant intermediate particles, remains blind to the information encoded inside the final state jets. A possible way to extract information from both jet substructure and global jet kinematics is to concatenate the information extracted from a multi-modal network. However, such a simple concatenation leads to an imbalance of the extracted information, within which the kinematic information generally dominates. We present a novel method for incorporating different-scale information extracted from both global kinematics and substructure of jets via a transformer encoder with a cross-attention layer. The model initially extracts the most relevant information from each dataset individually using self-attention layers before incorporating these using a cross-attention layer. The method demonstrates a larger improvement in classification performance compared to the simple concatenation method.

To assess our results, we analyze the learned information by the transformer layers through the examination of the attention maps of the self- and cross-attention layers. 
Attention maps provide information about the (most) important particles the model focuses on when classifying signal and background events. 
However, they cannot highlight the region in the feature (e.g., phase) space crucial for model classification. For this purpose, we utilize Gradient-weighted Class Activation Mapping (Grad-CAM) to highlight the geometric region in the $\eta-\phi$ (detector) plane where the model focuses on classifying events.
 
