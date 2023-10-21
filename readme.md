# Bidirectional Spatial-Temporal Adaptive Transformer for Urban Traffic Forecasting

## Introduction
Existing traffic forecasting methods focus on spatial-temporal dependency modeling, while two intrinsic properties of the traffic forecasting problem are overlooked. Inspired by the first property, each Transformer is designed to dynamically process the traffic streams according to their task complexities. Specifically, we realize this by the recurrent mechanism with a novel Dynamic Halting Module (DHM). Each Transformer performs iterative computation with shared parameters until DHM emits a stopping signal. Motivated by the second property, Bi-STAT utilizes one decoder to perform the present-to-past recollection task and the other decoder to perform the present-to-future prediction task. The recollection task supplies complementary information to assist and regularize the prediction task for better generalization. First, the complexity of diverse forecasting tasks is non-uniformly distributed across various spaces (e.g. suburb vs. downtown) and times (e.g. rush hour vs. off-peak).  Second, the recollection of past traffic conditions is beneficial to the prediction of future traffic conditions.  we propose a Bidirectional Spatial-Temporal Adaptive Transformer (Bi-STAT) for accurate traffic forecasting. Bi-STAT adopts an encoder-decoder architecture, where both the encoder and the decoder maintain a spatial-adaptive Transformer and a temporal-adaptive Transformer structure. 

## Framework
![Architecture of the Bi-STAT](https://github.com/chenchl19941118/Bi-STAT/assets/25497533/fc56f193-6b2b-4777-9b82-612a3cc7265b)

The framework of the Bidirectional Spatial-Temporal Adaptive Transformer (Bi-STAT). Bi-STAT consists of an encoder, a prediction decoder, and
a recollection decoder. In both the encoder and prediction decoder, we realize the task-adaptive computation (spatially and temporally) by the recurrent
mechanism with a novel Dynamic Halting Module (DHM). The recollection decoder performs the past reconstruction task to provide extra context information
and regularize the future prediction model.

## Results
![image](https://github.com/chenchl19941118/Bi-STAT/assets/25497533/e6269451-bec6-4ba2-8042-537028a50a44)
![image](https://github.com/chenchl19941118/Bi-STAT/assets/25497533/914db691-7ea7-48b1-949b-c9cb5e71c4bd)


## Requirements

Python 3.7.3   
Pytorch 1.9.0   
Numpy 1.19.5   
argparse

## Dataset

The datasets (PEMSD3, PEMSD4, PEMSD7 and PEMSD8) used in our experiments are available at [STSGCN](https://github.com/Davidham3/STSGCN).

## Project Structure

* lib: the codes to to construct the graph matrix and the spatial embedding matrix, and the common utils such as data loading, pre-processing and normalization, evaluation.

* models: implementation of our Bi-STAT model


## Run 

* (1) Get the sensor graph for the dataset
  
    python construct_adj.py 

* (2) Generate the spatial embedding for the dataset
  
    python generate_SE.py

* (3) Run our Bi-STAT model

    python run.py 

