# SPIE2023
### Towards Lifting the Trade-off between Accuracy and Adversarial Robustness of Deep Neural Networks with Application on COVID 19 CT Image Classification and Medical Image Segmentation 

## For classification:

# requirements:
python3.8.10
pytorch1.9.0


# guidance:

1. go to "DNNRobustness/app/COVID19a/" for the COVID19 experiment:
1.1 run train.py to get baseline model;
1.2 run train_adv.py to get model trained with our proposed adversarial training.
2. go to "DNNRobustness/app/MedMNIST/" for the COVID19 experiment:
2.1 run train.py to get baseline model;
2.2 run train_adv.py to get model trained with our proposed adversarial training.

## For segmentation:

1. Please go to this link(https://github.com/MIC-DKFZ/nnUNet) for instructions on installing and configuring nnUnet and dependent libraries.
2. All the experimental data can be downloaded from http://medicaldecathlon.com/ or as shown in paper(Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2020). nnU-Net: a self-configuring method 
for deep learning-based biomedical image segmentation. Nature Methods, 1-9.)
3. Go to "nnunet/run/"
4. use "python run_training --task id" to get the baseline model
5. use "python run_our_training --task id" to get the model with our proposed adversarial training

The experiment was run on Tesla V100 GPUs, CentOS system.

Based on original nnUnet, we did modifications on:
1.nnunet/training/network_training/network_trainer.py
2.nnunet/training/network_training/nnUNetTrainer.py
3.nnunet/training/network_training/nnUNetTrainerV2.py
4.nnunet/training/loss_functions/dice_loss.py
5.nnunet/training/loss_functions/crossentropy.py
6.nnunet/training/loss_functions/deep_supervision.py
7.nnunet/training/dataloading/dataset_loading.py
8.nnunet/training/data_augmentation/data_augmentation_moreDA.py
9.nnunet/utilities/to_torch.py

## contact
Should you have any questions, please feel free to contact:
l.ma@miami.edu
liang@cs.miami.edu
