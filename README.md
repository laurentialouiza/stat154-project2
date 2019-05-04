README

Project title: Cloud classification over the Arctic
A little info about your project and/ or overview that explains what the project is about
In this project, we are building a classification model to distinguish the presence or absence of clouds in the images collected by the MISR sensor abroad the NASA satellite Terra. 

Motivation
A short description of the motivation behind the creation and maintenance of the project. This should explain why the project exists.
In the future, we will be able to use this model to classify clouds from non-clouds on a large number of images. The ability to classify the presence of clouds will improve our understanding of the flow of visible and infrared radiation, and how these can help us studying the changes in arctic climate and its relationship with the increasing concentrations of atmospheric carbon dioxide.
# stat154-project2

EDA
From eda, data is not i.i.d

Data splitting (library: dplyr)
To prepare for training the data, we first need to do data cleaning by binding all the images into one dataframe and filtering out data that has unlabeled expert labels. Leaving out unlabeled expert labels will help us train our data better in distinguishing clouds from non-clouds.
After cleaning the data, we will split the entire data into three sets which are training, validation, and test. In this project, we take into account that the data is not i.i.d before splitting the data with two non-trivial different ways; described below:
Blocking: separate the whole image by n x n blocks. A function blocking were created with inputs image and n (number of blocks), dividing the range of x and y coordinate of the image by n and outputs n^2 divided blocks. Another function called split_way1 were created with inputs image and n that calls in the function blocking and randomly pick 75% of the blocks as training set, 12.5% as validation set, and 12.5% as test set. This function will returns splitted image.
Superpixels:

Generic Cross Validation
A function CVgeneric were created with inputs as the following:
generic_classifier: glm, lda, qda, svm, tree, rf
Lda_classifier: inputs fold_data, holdout_data, loss_func outputs prediction using lda/loss func
qda_classifier: same inputs
logit_classifier
training_features
training_labels
tum_folds
Loss_func
Hinge loss, logistic loss, etc? 
Logistic loss perceptron, 

Trying out classifiers #provide example under USAGE tabs
Comparing different methods using ROC curves
Why choose randomforest
Explain about what might be better classifier

Packages used:
dplyr
ggplot2
GGally
reshape2
MASS
pROC
ROCR
randomForest
