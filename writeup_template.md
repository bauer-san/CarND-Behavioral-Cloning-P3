#**Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
###Files Submitted & Code Quality
####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

- model.py containing the script to create and train the model
- drive.py for driving the car in autonomous mode
- model.h5 containing a trained convolutional neural network 
- writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator (beta_simulator.exe) and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is instantiated in line 70 of model.py and the model layers are defined in 71-107 of model.py.  The model is based on the Nvidia autopilot model described [here](https://arxiv.org/pdf/1604.07316.pdf).  I added a normalization layer, a cropping layer, and dropouts after each convolution layer to the Nvidia model.

####2. Attempts to reduce overfitting in the model

I added dropout layers, each with 0.5 keep probability, after each convolution layer to reduce overfitting.  The model was trained and validated on different data sets using `validation_split=0.2` argument of Keras `model.fit` to ensure that the model was not overfitting.  The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 110-112).

####4. Appropriate training data

A total of 8380 images were recorded.  Approximately 8000 of the images were of center lane driving and the remainder were edge recovery from right road edge.  I flipped each image around vertical axis and changed sign of the steering input. 

For details about how I created the training data, see the section 3. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with an existing model, the Nvidia autopilot model, then add features as they seemed to be necessary.  Sort of a bottom-up approach.

The Nvidia model was a reasonable starting point because it was originally developed as an end-to-end model from images as input to steering as output.  I with the Nvidia model and I added the Lambda layer to **perform normalization** as a starting model.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...
|  Layer (type)                    | Output Shape         |  Param #     | Connected to                 |
----------------------------------------------------------| -------------| -----------------------------|                     
| lambda_1 (Lambda)                | (None, 160, 320, 3)  |  0           | lambda_input_1[0][0]         |      
| cropping2d_1 (Cropping2D)        | (None, 80, 320, 3)   |  0           | lambda_1[0][0]               |       
| convolution2d_1 (Convolution2D)  | (None, 38, 158, 24)  |  1824        | cropping2d_1[0][0]           |     
| dropout_1 (Dropout)              | (None, 38, 158, 24)  |  0           | convolution2d_1[0][0]        |     
| convolution2d_2 (Convolution2D)  | (None, 17, 77, 36)   |  21636       | dropout_1[0][0]              |     
| dropout_2 (Dropout)              | (None, 17, 77, 36)   |  0           | convolution2d_2[0][0]        |     
| convolution2d_3 (Convolution2D)  | (None, 7, 37, 48)    |  43248       | dropout_2[0][0]              |     
| dropout_3 (Dropout)              | (None, 7, 37, 48)    |  0           | convolution2d_3[0][0]        |     
| convolution2d_4 (Convolution2D)  | (None, 5, 35, 64)    |  27712       | dropout_3[0][0]              |     
| dropout_4 (Dropout)              | (None, 5, 35, 64)    |  0           | convolution2d_4[0][0]        |     
| convolution2d_5 (Convolution2D)  | (None, 3, 33, 64)    |  36928       | dropout_4[0][0]              |     
| dropout_5 (Dropout)              | (None, 3, 33, 64)    |  0           | convolution2d_5[0][0]        |     
| flatten_1 (Flatten)              | (None, 6336)         |  0           | dropout_5[0][0]              |     
| dense_1 (Dense)                  | (None, 1164)         |  7376268     | flatten_1[0][0]              |     
| dense_2 (Dense)                  | (None, 100)          |  116500      | dense_1[0][0]                |     
| dense_3 (Dense)                  | (None, 50)           |  5050        | dense_2[0][0]                |     
| dense_4 (Dense)                  | (None, 10)           |  510         | dense_3[0][0]                |     
| dense_5 (Dense)                  | (None, 1)            |  11          | dense_4[0][0]                |     
| Total params: 7,629,687 | | | | 
| Trainable params: 7,629,687 | | | | 
| Non-trainable params: 0 | | | | 

####3. Creation of the Training Set & Training Process

Using a dataset of center lane driving of one (or maybe two, I forget) and only the images from the center camera, the model trained pretty well with an accuracy in range of 70% but the validation set was never more than 30%.  Still, I decided to see how the model would perform in the simulator - and to prove the workflow.  I was jumping for joy when the car didn't immediately drive off the road but I stopped jumping when it drove in to the lake just before the bridge.

Here is an example image of center lane driving:
![](.\my_test_data\IMG\center_2017_02_11_11_03_04_094.jpg)

I realized that the model didn't steer away from the road edge because it had not been trained to do that yet.  I recorded one lap of recovery data where I stopped the vehicle while pointing at the right road edge and steered it back to the center of the lane.  I recorded only the right road edge recovery because the data is augmented by flipping the images and steering angles before the training.

Here is an example image of recovery driving:
![](.\my_test_data\IMG\right_2017_02_12_08_21_20_953.jpg)

At the same time, I added the **cropping layer** to reduce the amount of time for training.  I trimmed the original height of 160 pixels to 80.  Half of the data to crunch.  Nice!

I retrained the model and found the vehicle was steering wildly when it reached the lane edge.  This is because the recovery data I recorded was all recorded within *initial_speed=0*.  The steering angle was way too high for the speed that the simulator was driving, ~23 mph.  I decided to scale the steering inputs of the recovery data and it took a few iterations to find a suitable scale.  A scale of 0.5x improved the erratic steering at the road edge.  I was surprised when the car was able to drive laps without running off the road but it was still not nice because the car sort of rode the curb on the bridge.  This was when I started to think more about what was happening and I realized that I had converted the colorspace of the training data to YUV but I had not done the same in the `drive.py`.  I updated drive.py and at the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#####Summary of Training Process
I had 8380 data points which I preprocessed by converting to YUV colorspace.  The 8380 images and steering angles were flipped and the model was trained on 16760 images.  20% of the data was withheld for validation.

I also defined an early stopping callback to stop training when the loss was not improving. 

I used an adam optimizer so that manually training the learning rate wasn't necessary.
