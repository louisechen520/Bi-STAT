# Bidirectional Spatial-Temporal Adaptive Transformer for Urban Traffic Forecasting

the datasets used in our experiments including PEMSD3, PEMSD4, PEMSD7 and PEMSD8 are available at [STSGCN](https://github.com/Davidham3/STSGCN).

## Structure:

* lib: containing methods to construct the graph matrix and the spatial embedding matrix, besides there are some self-defined modules for our work in utils, such as data loading, data pre-processing, normalization and evaluation metrics.

* models: implementation of our Bi-STAT model

## Requirements

Python 3.7.3, Pytorch 1.9.0, Numpy 1.19.5 and argparse

## Runs 
remember to change the data directory to your own

* to get the sensor graph for the dataset:
  
    python construct_adj.py 

* to generate the spatial embedding for the dataset:
  
    python generate_SE.py

* to run our Bi-STAT model:

    python run.py 

