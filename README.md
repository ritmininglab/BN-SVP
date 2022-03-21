# BN-SVP
Contains source code for the CVPR2022 paper titled "Bayesian Nonparametric Submodular Video Partition for Robust Anomaly Detection"

This respository consists of the following files

1. train_sanghaitech_hdp_hmm.py trains the model for SanghaiTech dataset
   Params:
    a. run -> replication number for the experiment
    d. out_th -> percentile used to determine \epsilon in eq. 13
   Returns:
       Stores evaluation AUC, losses in logs/SanghaiTech folder and best model in models/SanghaiTech folder
       
2. hdp_hmm.py
   Constructs the sticky HDP-HMM model and posterior parameters are inferred via Blocked Gibbs Sampling
        
All datasets are publlicly available.
