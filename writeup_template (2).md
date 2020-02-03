# **Behavioral Cloning** 



---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report




## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model2.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model2.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model2.h5
```

#### 3. Submission code is usable and readable

The model2.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy


#### 4. Appropriate training data

The training images provided by Udacity turned out to be sufficient.
Data augmentation was done by flipping the images horizontally and accordingly multiplying the steering angles by -1.

#### Loading the data

As per the suggestions of Udacity, a generator function was designed to save memory instead of loading the images all at once.
Within the generator function data augmentation by flipping the images was performed.

The steering angles for the 'left' camera images were increased by 0.2 and that of the 'right' ones were decreased by 0.2



#### Preprocessing

1> Shuffling
2> Normalization
3> Cropping - 60 pixels from the top and 25 pixels from the bottom 

Cropping was done to get rid of the irrelevant features like the trees and the environment and focus on the trajectory of the road.


[Image before normalization]: ./examples/image.PNG "Image before normalization"
[Normalized image]: ./examples/normalized.PNG "Normalized Image"
[Cropped Image]: ./examples/cropped.PNG "Cropped Image"

#### 1. Final model architecture

The description of the architecture can be found
[here](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

The model can be summarized by the picture below:

[Model Summary]: ./examples/architecture.PNG "Model Summary"

The validation set proportion was chosen to be 0.2.

After the input images with sizes (160, 320, 3) went through normalization and cropping and assuming the size (75, 320, 3),
they were passed through a series of five 2D convolutional neural layers with the number of channels being 24, 36, 48, 64 and 64 respectively. ReLu was chosen as the activation functions after each of the convolutional layers.

The output of the last convolutional layer was flattened and passed through 4 fully connected layers with node sizes 100, 50, 10 and 1 respectively. ReLu was again chosen as the activation function for each layer.

The loss function was chosen to be Mean Squared Error.
Adam was gradient optimizer algorithm.



#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer with keep probability 0.5 after the first fully connected layer (the one with 100 nodes) in order to reduce overfitting. 

The model was trained and validated on the dataset provided by Udacity. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was the default one.

The model was trained for 5 epochs.

Generator batch size was 32.

#### 4. Testing the model

The model was tested by driving the car with the simulator in the autonomous mode. 

#### 5. Output video

[Output Video]: run2.mp4 "Output video"

