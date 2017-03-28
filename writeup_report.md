#**Behavioral Cloning**

##Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./img/model.png "Model Visualization"
[image2]: ./img/center.jpg "center image"
[image3]: ./img/recovery_1.jpg "Recovery Image"
[image4]: ./img/recovery_2.jpg "Recovery Image"
[image5]: ./img/recovery_3.jpg "Recovery Image"
[image6]: ./img/model2.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and 3x3 filter sizes and depths between 32 and 64 (model.py lines 96-100)

The model includes RELU layers to introduce nonlinearity (code line 96-100), and the data is normalized in the model using a Keras BatchNormalization layer (code line 94).


####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 102).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 22). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 116).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and driving in reserved direction.
For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the Nvidia end to end pipline I thought this model might be appropriate because it promise a great effect.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that it can overcame the overfitting using the dropout layout of a possibility of 0.2. 

Then I changed the subsample fron 1x1 to 2x2, so the Fifth convlution layer can output something like 1xNxS similar to the origin NVIDIA pipeline.


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, to improve the driving behavior in these cases, I add some more trainging data which the car recover from the edge.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


####2. Final Model Architecture

The final model architecture (model.py lines 96-100) consisted of a convolution neural network with the following layers and layer sizes 

- Batch Normalization Layer
- Cropping2D Layer with a top 50px and button 20px croped out
- Convlution Layer with 24 filter 5x5 , valid padding , stride 2x2
- Convlution Layer with 36 filter 5x5 , valid padding , stride 2x2
- Convlution Layer with 48 filter 5x5 , valid padding , stride 2x2
- Convlution Layer with 64 filter 3x3 , valid padding , stride 2x2
- Convlution Layer with 64 filter 3x3 , valid padding , stride 1x1
- Flatten Layer
- Dropout Layer with a possiblity of 0.2
- Full Connect Layer with 100 neuron
- Full Connect Layer with 50 neuron
- Full Connect Layer with 10 neuron
- Full Connect Layer with 1 neuron
- Output Layer



```python

model = Sequential(
    [
        BatchNormalization(epsilon=1e-3, input_shape=(160,320,3)),
        Cropping2D(cropping=((50, 20), (0, 0))),
        Conv2D(24,5,5,activation="relu", border_mode="valid", subsample=(2,2)),
        Conv2D(36,5,5,activation="relu", border_mode="valid", subsample=(2,2)),
        Conv2D(48,5,5,activation="relu", border_mode="valid", subsample=(2,2)),
        Conv2D(64,3,3,activation="relu", border_mode="valid", subsample=(2,2)),
        Conv2D(64,3,3,activation="relu", border_mode="valid", subsample=(1,1)),
        Flatten(),
        Dropout(0.2),
        # Dense(1164),
        Dense(100),
        Dense(50),
        Dense(10),
        Dense(1)
    ]
)
```


Here is a visualization of the architecture
![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]


To augment the data sat, I also flipped images and angles thinking that this would ... For example.



After the collection process, I had 23104 number of data points. I then preprocessed this data by flipped both the center image, left cam image  and the right cam image.


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the training and validation loss, I used an adam optimizer so that manually training the learning rate wasn't necessary.
