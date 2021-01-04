# Handwritten Digit Recogniton using Convolution Neural Networks

## Steps followed in the project:
1. Load Data using data stored in google drive
2. Convolution Neural Network using PyTorch
      -Using 2 convolution layers with kernel size=(2,2),1 pooling layer with kernel size=2 and fully connected feed forward network with 1 hidden layer with 84 neurons
      -Using CrossEntropy loss and ADAM optimizer to gain better accuracy and avoid overfitting
      
Only first 100 images for each digit have been considered for faster processing. 2 userdefined transformations have been defined:
- Rescale: Used to Rescale each image to uniform size
- ToTensor: Used to convert all images to tensor objects
70% of data is considered for training and the rest 30% for testing the data
      
## Results:
Training Loss achieved: 0.117  
Validation accuracy for digit 0: 100.00  
Validation accuracy for digit 1: 96.42  
Validation accuracy for digit 2: 96.46  
Validation accuracy for digit 3: 97.40  
Validation accuracy for digit 4: 98.30  
Validation accuracy for digit 5: 96.52  
Validation accuracy for digit 6: 97.54  
Validation accuracy for digit 7: 95.95  
Validation accuracy for digit 8: 89.08  
Validation accuracy for digit 9: 92.28  
