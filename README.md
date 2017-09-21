## Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/before_norm.png "original"
[image3]: ./examples/after_norm.png "normalization"
[image4]: ./examples/0.jpg "Traffic Sign 1"
[image5]: ./examples/1.png "Traffic Sign 2"
[image6]: ./examples/2.jpg "Traffic Sign 3"
[image7]: ./examples/4.png "Traffic Sign 4"
[image8]: ./examples/9.jpg "Traffic Sign 5"
[image9]: ./examples/50-original.png "Traffic Sign 50 oiginal"
[image10]: ./examples/visualization50km.png "50 km/h samples number"
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/dramadan/Traffic_Sign_Classifier_CarND/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to not convert the images to grayscale to make use of the color information which increase the classification. accuracy. To acieve consistancy of the training set images, I normalizing(Which means standrizing) the training set.
Here is an example of a traffic sign image before and after normalization.

![alt text][image2]


![alt text][image3]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.



My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| (0) Input         		| 32x32x3 RGB image   							| 
| (1) Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| (2) RELU					|												|
| (3)Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| (4) dropout					|										0.5		|
| (5) Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16     									|
| (6) RELU					|												|
| (7) Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| (8) Convolution 5x5	    | 1x1 stride, valid padding, outputs 1x1x400     									|
| (9) RELU					|												|
| (10) Flatten 7 and 9					|									400, 400			|
| (11) Concatenate					|							800					|
| (12) dropout					|										0.5		|
| (13) Fully connected		| outputs 43 classes        									|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used Adam optimizer, batch size is 128, 30 epochs, 0.001 learning rate.
The code for training the model is located in the cell number 13 of the ipython notebook.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 93.5 %
* validation set accuracy of 99.0 %
* test set accuracy of 95 %

First, I used the LeNet model which described in the lectures preceeding the project after modifying it to accept RGB images as input and 43 classes as output and tuning the other parameters. Unfortunatley this model had some limitations such as overfitting the train data causing poor validation accuracy. Accordingly, I came across different architecture specially "Traffic Sign Recognition using Multi-Scale Convolutional Network" https://github.com/dramadan/Traffic_Sign_Classifier_CarND/blob/master/sermanet-ijcnn-11.pdf
I implemented the mentioned architecture. There was a noticeble improvement over the LeNet but Overfitting was still an issue. After invistigating and trying possible modifications to improve the accuracy of the sermanet architecture. These modifications include:

* Changing the filter dimensions to accept RGB input instead of greyscale images to make use of color information which increase the classification accuracy.
* Adding a dropout after the first convolution layer to decrease the effect of over-fitting.
* Tunning the parameters of learning rate, batch size, epochs count to reduce the gap between the validation and training accuracy. After tunning, the below parameters give the best results:
- Learning rate of 0.001
- Batch size of 128
- Epoch count of 30
- Keep probability of 0.5

The added dropout layer achieved its aim of reducing the effect of over-fitting.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The 50 km/h speed limit sign is the image might be difficult to classify by the proposed model. We see that the 50 km/h speed sign is the class with the maximum number of training samples in the training set.

![alt text][image9]

The 50km/h image might be difficult to classify because due to the fact that, unlike the training data set, the '50' digits are not centered within the red circle. This can be an indication of over-fitting the model to only recognize speed limit signs with the digits in the exact center of the red circle.
![alt text][image5] ![alt text][image10]
  

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 50 km/h      		| 30 km/h  									| 
| 20 km/h    			| 20 km/h 										|
| 30 km/h					| 30 km/h										|
| priorty road      		| priorty road					 				|
| Yield 		| Yield    							|


The model was able to correctly guess 19 of the 20 traffic signs, which gives an accuracy of 95%. This compares favorably to the accuracy on the test set of 93%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the cell number 20 of the Ipython notebook.

For the first image, the model is relatively sure that this is a 20 km/h sign(probability of 0.99). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99       			| 20 km/h   									| 
| 0.0    				| 30 km/h 										|
| 0.0				| 70 km/h										|
| 0.0	      			| 120 km/h			 				|
| 0.0				    | 	Children crossing    							|



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


