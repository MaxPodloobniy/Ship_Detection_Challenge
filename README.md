## Introduction  
This repository presents my approach to tackling the Airbus Ship Detection 
Challenge, employing two key models: a ship detection model and a ship 
segmentation model. I used two models for this, ship detection model and 
ship segmentation model. I decided to use two models instead of one driven 
by the fact that only 35% of the training images include ships, there's 
a considerable portion of the dataset where ships are absent, making 
segmentation challenging.  
### Segmentation model  
Let's explore the ship segmentation model, which employs the U-Net architecture. 
It consists of six downsampling blocks(Conv+Conv+MaxPool) with number of 
filters [8, 16, 32, ..., 256] and five upsampling blocks(TransposeConv+
Concatenate+Conv+Conv). The output layer is Convolution with 1x1 filter 
and sigmoid activation. I chose a low number of filters because I lack the 
ability to train the model on powerful hardware. Additionally, ships are not 
geometrically complex objects, so a large number of filters is unnecessary 
for detecting their features.  
### Detection model  
Now turning to the ship detection model, it follows a traditional Convolutional 
Network approach. It consists of three Conv+MaxPool blocks, along with one fully 
connected Dense layer and an output Dense layer with a sigmoid activation function.
## Project Structure Overview
### Directories
- **ship_detection/**: Contains files for creating dataset, training and 
evaluating ship detection model.
- **ship_segmentation/**: Contains files for RLE decoding, training, 
upgrading and evaluating ship detection model.
### File Descriptions
- **model_evaluation.py**: Models evaluation
- **dataset_for_detection.py**: Script for ship detection dataset creation
- **ship_detection_model.py**: Script for creating and training model 
- **ship_detection_model.zip**: Compressed ship detection model
- **RLE_decoding**: Script for decoding RLE string and creating dataset for
ship segmentation model
- **segmentation_model.py**: Script for creating and training segmentation model
- **ship_segmentation_model.keras**: U-Net Segmentation model
- **data_augmentation.py**: Code for creating new dataset by using data augmentation
- **segmentation_upgrade.py**: Upgrading segmentation model by using new dataset
and loss function
- **ship_segmentation_model_v2.keras**: New segmentationmodel
- **what's_done.txt**: File where I wrote all my progress while doing thing 
project
- **resourses.txt**: File containing articles and examples from which I sourced 
the information
- **requirements.txt**: File with required Python modules
## Training the models
For both model fitting and dataset creating I used Google CLoud VM with Nvidia 
Tesla T4 GPU.  
##### Loss functions:  
- **Segmentation model**:  Utilized the custom function dice_p_bce(), which 
combines binary cross-entropy and Dice coefficient
- **Detection model**: Employed the built-in binary_crossentropy() loss 
function  

For training the segmentation model, I used a dataset with 42k pairs of 
images and masks, with each image containing at least one ship. Upon 
validation, the model achieved an accuracy of 0.75 as measured by the 
Dice coefficient.  
For the ship detection model, I employed a training dataset consisting 
of 50,000 samples, complemented by a validation dataset of 10,000 samples. 
Continuing the evaluation, the model demonstrated a validation accuracy 
of 0.8534.

### References
#### Models
....
#### Datasets
....