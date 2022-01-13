# TFA-Net

Codes for ``TFA-Net: A Deep Learning-Based Time-Frequency Analysis Tool''

1. Folder TFA-Net_train contains the codes for TFA-Net training
   --complexTrain.py is the main function for TFA-Net
   --stft_backbone.py is the main function for a RED-Net modified as our demands. That is, TFA is performed on the basis of the STFT results.
   --generate_dataset.py is used to generate the test set, and model_explanation.py is used for interference of the test set.
   --Env. Requirements: pytorch 1.7.1

2. Folder TFA-Net_inference contains the codes for experiments in paper
   --The subFolder TFA_exe is a flask interferece implementation for the model interferece, which is required to run first, Env. Requirements: pytorch 1.7.1. 
     * TFA_exe/config includes a configuration file for locating the experiment codes and service port used for calling a model interference
   --The experiments included in the paper:
     * exp1_simu.m is the first simulation in Section IV A
     * exp1_simu.m is the second simulation in Section IV A
     * exp2_bat.m is the TFA of bat signal in Section IV B (1)
     * exp2_heart.m is the TFA of ECG in Section IV B (2)
     * exp3_dianji.m is the TFA of micro-Doppler of reflectors in Section IV B (3)
     * exp3_breheart.m is the TFA of micro-Doppler of Vital Signs in Section IV B (4)
     * exp3_voice.m is the TFA of voice of a mammal in Section IV B (5).
     
   
