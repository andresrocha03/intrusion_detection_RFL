# Intrusion Detection System using Reinforcement Federated Learning

## Overview 
This is a project dedicated to implementing an algorithm of Reinforcement Federated Learning using the Flower and Stable-Baselines Library. Other Federated algorithms were implemented in order to compare the results. The approach was Anomaly-based, identifying whether there is or not an attack. The federated model was also trained in 5 raspberry-pi using Ansible to manage the training. This work was published in the [digital library of the Brazilian Computing Society](https://sol.sbc.org.br/index.php/sbrc_estendido/article/view/35889/35676)


## [Dataset]
Available on this [link](https://www.unb.ca/cic/datasets/ids-2017.html)

A detailed analysis of CICIDS2017 dataset for designing Intrusion Detection Systems.
This paper briefly describes the CICIDS2017 dataset. It suggests relabeling classes for reducing imbalance. [Link](https://www.researchgate.net/profile/Ranjit-Panigrahi/publication/329045441_A_detailed_analysis_of_CICIDS2017_dataset_for_designing_Intrusion_Detection_Systems/links/5c264977a6fdccfc706e43b6/A-detailed-analysis-of-CICIDS2017-dataset-for-designing-Intrusion-Detection-Systems.pdf)

A Detailed Analysis of the CICIDS2017 Data Set.
More deep description of the dataset. It proposes singular value decomposition for improving accuracy.
[Link](https://link.springer.com/chapter/10.1007/978-3-030-25109-3_9)


## Centralized
In this folder, there are the codes responsible for training, testing, and evaluating the centralized models.

## Federated
In this folder, there are the codes responsible for training, testing, and evaluating the federated models.

## Pre_processing
In this folder, there is the code responsible for the pre_processing

## Environments 
In this folder, it is possible to find the environments created for the reinforcement learning 

## Results
Data I collected from different models since I began the project

## Ansible
Playbooks and code used for managing and implementing the model in raspberry-pi clients.