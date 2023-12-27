# Multi-Scale Cross-Attention Transformer Encoder for Event Calssification

 &emsp; Scanner package assisted by  Machine learning models for faster convergence to the target area in high diemnsions. The packge based on [arXiv:2207.09959 [hep-ph]](https://arxiv.org/abs/2207.09959). The current version automate the scan for Spheno, HiggsBounds and HiggsSignals.
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
&emsp;  In this package we have implemented two broad classes of ML based efficient sampling methods of parameter spaces, using regression and classification. we employ an iterative process where the ML model is trained on a set of parameter values and their results from a costly calculation, with the same ML model later used to predict the location of additional relevant parameter values. The ML model is refined in every step of the process, therefore, increasing the accuracy of the regions it suggests. We attempt to develop a generic tool that can take advantage of the improvements brought by this iterative process. we set the goal in filling the regions of interest such that in the end we provide a sample of parameter values that densely spans the region as requested by the user. With enough points sampled, it should not be difficult to proceed with more detailed studies on the
implications of the calculated observables and the validity of the model under question. We pay special attention to give control to the user over the many hyperparameters involved in the process, such as the number of nodes, learning rate, training epochs, among many others, while also suggesting defaults that work in many cases. The user has the freedom to decide whether to use regression or classification to sample the parameter space, depending mostly on the complexity of the problem. For example, with complicated fast changing likelihoods it may be easier for a ML classifier to converge and suggest points that are likely inside the region of interest. However, with observables that are relatively easy to adjust, a ML regressor may provide information useful
to locate points of interest such as best fit points, or to estimate the distribution of the parameters. After several steps in the iterative process, it is expected to have a ML model that can accurately predict the relevance of a parameter point much faster than passing through the complicated time
consuming calculation that was used during training. As a caveat, considering that this process requires iteratively training a ML model during several epochs, which also requires time by itself, for cases where the calculations can be optimized to run rather fast, other methods may actually
provide good results in less time.
