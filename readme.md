# Bidirectional Spatial-Temporal Adaptive Transformer for Urban Traffic Forecasting

## Introduction

we propose a Bidirectional Spatial-Temporal Adaptive Transformer (Bi-STAT) for accurate traffic forecasting. 
Bi-STAT adopts an encoder-decoder architecture, where both the encoder and the decoder maintain a spatial-adaptive 
Transformer and a temporal-adaptive Transformer structure. 

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

