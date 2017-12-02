# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolutional neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/cnn-architecture.png "Model Visualization"
[image2]: ./writeup_images/center.jpg "Normal Image"
[image3]: ./writeup_images/recover1.jpg "Recovery Image"
[image4]: ./writeup_images/recover2.jpg "Recovery Image"
[image5]: ./writeup_images/recover3.jpg "Recovery Image"
[image6]: ./writeup_images/mse_loss.png "MSE Loss"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolutional neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolutional neural network with kernel sizes between 3x3 and 5x5 and depths between 24 and 64 (model.py lines 79-83) 

The model includes RELU activations to introduce nonlinearity (code lines 79-83), and the data is normalized in the model using a Keras lambda layer (code line 77). 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 66-70). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 90).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving the course counter-clockwise, and flipping images during preprocessing.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to leverage a published architecture and then adapt it as necessary.  

My first step was to use a convolutional neural network model based on work published by nvidia:
https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

I thought this model might be appropriate because nvidia has used to to successfully drive actual cars.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a relatively high mean squared error on both the training and validation set. This implied that the model was underfitting.

To combat the underfitting, I modified added a preprocessing step to normalize the data for zero mean and equal variance (code  line 77).

Then I croped the tops and bottoms off the images to mostly show the road instead of the front of the car and surrounding scenery (code line 75).

The final step was to run the simulator to see how well the car was driving around track one. There vehicle tended to pull to the left and was unable to recover when it got off track.  To improve the driving behavior in these cases, I recorded additional center lane driving, added images from the left and right camera with a steering correction of 0.15 (code line 38), generated revcovery images by drving near the edge of the track and then recording the recovery, and generated additional data that would not favor left turns by creating flipped images and steering angles (code lines 50, 52).

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 75-88) consisted of a convolutional neural network with the following layers and layer sizes.  Note, this visualization of the architecture was provided by nvidia:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover when it got near the edge of the track. These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would reduce the model's tendency to steer to the left.

After the collection process, I had 82,450 data points. I then preprocessed this data by applying the following function to every pixel: (pixel / 255.0) - 0.5

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was between 3 and 7 as evidenced by the fact that the mean squared error (mse) loss on the validation set stopped decreasing beyond that.  The following chart shows mse loss over epochs.

![alt text][image6]

I used an adam optimizer so that manually training the learning rate wasn't necessary.


