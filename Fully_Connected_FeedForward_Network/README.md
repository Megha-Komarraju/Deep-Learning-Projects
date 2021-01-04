# Sentiment Analysis of Facebook comments using Fully Connected Feed Forward Network

## Steps followed in the project:

1. Load Data as a pandas dataframe
2. Data Preprocessing using Tfidf Vectorizer and creating training and testing sets
3. Traditional Machine Learning Model: Random Forest using sklearn with 10-fold cross CV and entropy criterion
4. Fully connected feedforward Neural Network using PyTorch 
   *Using 1 hidden layer with 50 neurons
   *Using Dropout, CrossEntropy loss and ADAM optimizer to gain better accuracy and avoid overfitting
   
## Results:
Training Accuracy achieved: 0.9937
Validation Accuracy achieved : 0.9650

## Libraries to be imported:
import pandas as pd
import numpy as np
from sklearn.model_selection 
import KFoldfrom sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F from torch.utils.data
import TensorDataset, DataLoader
import torch.optim as optim

