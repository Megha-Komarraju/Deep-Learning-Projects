# Sentiment Analysis of Amazon Review using RNN

## Steps followed in the project:
1. Create a dictionary: Using `TfidfVectorizer` from sklearn to generate tf-idf values for every word in each document from 500 documents each for positive and negative reviews
2. Data preprocessing and preparation:Each document is represented as a vector of size equal to maximum size of document (max_doc_size) in the dataset. 
   Each word in the document is represented as a vector using one hot encoding representation but by using the tfidf value for word in document calculated in the previous step instead of 1.
   Each word vector size is 200. The final result is an array of 1000 x max_doc_size x 200
3. Create the Train loader and Validation loader
4. Create a multi-layer RNN with hidden size of 256 neurons and a fully connected feed forward layer with 32 neurons
5. Traning and Validation: For each batch in training, print the average validation loss for all instances in the validation set.

Results:
Training loss achieved: 0.69297  
Validation loss achieved : 0.69339

Libraries to be imported:
import pandas as pd  
import numpy as np  
import torch  
import torch.nn as nn  
import torch.nn.functional as F from torch.utils.data  
import torch.optim as optim  
import os  
from skimage import transform  
from torchvision import transforms, utils  
from torch.utils.data import random_split  
from torch.utils.data import Dataset, DataLoader  
