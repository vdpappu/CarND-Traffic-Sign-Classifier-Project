**Building a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

###Data Load
* German traffic signs data was dowloaded from source and unzipped to get the pickle files
* Train, Validation and Test datasets were extracted from pickle files

###Data Set Summary & Exploratory Analysis
* Data Set Summary
  * Basic summary statistics like number of train, test and validation samples, size of the input images and number of classes are calculated
  * Number of per class samples in each of train, test and validation samples are calculated
* Exploratory Analysis
  * Samples images from train dataset are printed to get an idea of the dataset
  * Plotted a histogram with classes on X-axis and Count on Y-axis
[//]: # (Image References)

[image1]: ./writeup_images/Hist.jpg
[image2]: ./writeup_images/Actual_Img.png
[image3]: ./writeup_images/Grey_Scale.png
[image4]: ./writeup_images/11.jpg
[image5]: ./writeup_images/13.jpg
[image6]: ./writeup_images/15.jpg
[image7]: ./writeup_images/21.jpg
[image8]: ./writeup_images/38.jpg

![alt text][image1]


###Training pipeline setup
This section briefs the steps in the pipeline for training Sign classifier
* <b>Preprocessing</b>
  * As a first step, images are converted to grey scale - this step is needed for following reasons:
    * Grey scale images are faster to train - 3X less space and compute compared to RGB images
    * The dataset is color agnostic - in a way, change is color doesn't change the class of the image<br>
![alt text][image2] ![alt text][image3]
  * Image data is normalized and the pixel intensities are brought down to the scale of [0,0.9] from [0,255]
    * Normalization is needed for stabilizing the training process
    * Normalization helps in faster convergence of the deep learning model
* <b>Model Architecture</b>
  * As a first try, I have trained LeNet on the dataset. LeNet architecture is not complex enough to capture necessary features. 
  This is evident from the low validation accuracy (~89%)
  * I have made following changes to the network which has given the acceptable validation and test accuracies:
    * Changed number of convolution filters in conv1 and conv2 from [6,20] to [16,32] - this helps in learning more features
    * Changed stride in Maxpooling on Conv2 from 2 to 1 - this will ensure that minute details are not ignored
    * Added dropout in fc1 - this acts as a regularizer and minimizes overfitting to the training data
  * With above changes, validation and test accuracies are ~94%
  * Final model architecture is as follows
    * Conv(20X5x5) -> Stride(1)
    * ReLU
    * Maxpool(2x2) -> Stride(2)
    * Conv(32X5x5) -> Stride(1)
    * ReLU
    * Maxpool(2x2) -> Stride(1)
    * FCLayer
    * ReLU
    * DropOut(50%)
    * FCLayer
    * ReLU
    * Softmax
* <b>Model Training</b> 
  * A batch size of 256 was used, having higher batch size ensures faster stabilization
  * Learning rate was set to 0.001
    * Learning rates of 0.0001 and 0.1 are tried, with the former learning rate the learning was slower and with the later, 
    the network has become instable - inconsistent learning, high train-validation accuracy difference, 
    abrupt changes in accuracy measures are observed
   * softmax_cross_entropy_with_logits was used as function to be minimized and AdamOptimizer as the optimizer
    * softmax_cross_entropy_with_logits is proven to help with stabilizing the learning compared to MSE - 
    helps get rid of gradients saturation during backprop operation
* <b>Solution Approach</b> 
   The approach taken to achieve >93% validation accuracy was discussed in earlier sections
   * Final Results on Train, Test and Validation Sets (accuracy)
    * Train set - 99.2%
    * Validation set - 95.6%
    * Test set - 93.8%
    
###Test a Model on New Images
* <b>Acquiring New Images</b>
  * Following images were downloaded from web and the trained model was run on them to check the performance of the trained net
![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]
* <b>Performance on New Images</b>
  * Of the five images selected, four were classified correctly ie ~80% accuracy
  * "Double curve" sign was classified as "Beware of ice/snow" and the confiden scores are close
    * <b>Beware of ice/snow</b> - 58.7%
    * <b>Double curve/snow</b> - 33.4%
   * This needs further analysis to find the reason - possible reason could be insufficient training data, overfitting.
* <b>Model Certainty - Softmax Probabilities</b>
  * In cases where the model classification was accurate, the soft max probabilities of predicted classes are close to 99%,
  this could be because the downloaded images are close to the images in the dataset provided
  * In case of one samle, "No vehicles", the confidence score was low - ~58.7% and the next closest prediction was 
  "Priority road" with a confidence score of ~33.4%. This needs further analysis

###Observed limitations of current model
* The network seem to be overfitting to the dataset provided as a whole
* Performance on the images that are not similar to dataset provided is not in acceptable range

###Suggested Improvements
* Data augmentation can improve the performance of the net
* Using external data can help model generalize
* Appropriate use of regularization layers

    
    
    
