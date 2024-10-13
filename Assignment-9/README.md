## Data Augmentation using Albumentations:
In the provided code, Albumentations is utilized to perform data augmentation on the CIFAR-10 dataset. Here's how it's done:

- Define Transformations:
  - train_transforms: A composition of augmentation techniques applied during training, including
    - **horizontal flips, shifting, scaling, rotation, and coarse dropout**
      
  - test_transforms: Transformation applied to the test set, which includes normalization
    
- Custom Dataset Class: Cifar10SearchDataset inherits from datasets.CIFAR10 and overrides the __getitem__ method to apply transformations to the images.

- Application of Transformations: The transform parameter in the dataset loader is set to the defined transformations, ensuring that images are augmented during training and properly processed during testing.

## Model Architecture Details:
The model architecture consists of several convolutional layers organized into different blocks. Here are the key points regarding the model architecture:

- Depthwise Separable Convolution: The model utilizes depthwise separable convolutions in some layers. This type of convolution decomposes standard convolution into two steps: depthwise convolution and pointwise convolution, reducing the computational cost and number of parameters.
- Dilation: Dilation is applied in certain convolutional layers to increase the receptive field of the network without increasing the number of parameters.
- Dropout: Dropout layers are used after some convolutional layers to prevent overfitting by randomly setting a fraction of input units to zero during training.
- Batch Normalization: Batch normalization layers are employed after convolutional layers to stabilize and accelerate the training process.
- ReLU Activation: Rectified Linear Unit (ReLU) activation functions are used after convolutional layers to introduce non-linearity into the network.
- Global Average Pooling: The model utilizes Global Average Pooling (GAP) before the final classification layer. GAP reduces spatial dimensions to a single value per feature map, aggregating spatial information.

## Observations

- Total parameters - 203,856
- Total Epocs - 55
- Training Accuracy: The training accuracy increases steadily with each epoch, reaching around 76% at the end of training.
- Training Loss: The training loss decreases consistently over epochs, indicating that the model is learning effectively from the training data.
- Test Accuracy: The test accuracy also increases over epochs, reaching around 85% at the end of training.
- Test Loss: The test loss decreases throughout training, indicating that the model generalizes well to unseen data.
- Misclassifications: The misclassification rate could be further analyzed to identify patterns or classes that are particularly challenging for the model. The misclassified images, along with their predicted and true labels, could be examined to gain insights into areas for improvement.
- Dilation and Depthwise Separable Convolution: The model architecture makes use of dilation and depthwise separable convolution to increase the receptive field and capture more contextual information from the input images. These techniques can help improve the model's performance, especially in tasks where capturing spatial dependencies is crucial.
