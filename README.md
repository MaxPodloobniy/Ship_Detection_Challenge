## Introduction  
This repository is my vision of solving Airbus Ship Detection Challenge. I 
used two models for this, ship detection model and ship segmentation model.
I decided to use two models instead of one because in training data only 35% 
of images has ships so in 65% of inputs there are nothing to segment.  
### Segmentation model  
Firstly let's talk about ship segmentation model, I used U-Net architecture, 
it consists of six downsampling blocks(Conv+Conv+MaxPool) with number of 
filters [8, 16, 32, ..., 256] and five upsampling blocks(TransposeConv+
Concatenate+Conv+Conv). The output layer is Convolution with 1x1 filter 
and sigmoid activation. I decided to use low filters number because I 
haven't ability to train the model on very powerful hardware and also ships 
are not very geometrically complicated objects so there is no need in large 
number of filters to detect large number of object features.  
### Detection model  
Now let's talk about ship detection model. It's a classical Convolutional 
Network with three Conv+MaxPool blocks, one fully connected Dense layer 
and one output Dense layer with sigmoid activation function. 
## Project Structure Overview
### Directories
- **ship_detection/**: Contains files for creating dataset, training and 
evaluating ship detection model.
- **ship_segmentation/**: Contains files for creating dataset, training, 
upgrading and evaluating ship detection model.
### File Descriptions
- **dataset_for_detection.py**: Script for ship detection dataset creation
- **ship_detection_model.py**: Script for creating and training model 
- **ship_detection_model.zip**: Compressed ship detection model
- **RLE_decoding**: Script for decoding RLE string and creating dataset for
ship segmentation model
- **segmentation_model.py: Script for creating and training segmentation model
- 





