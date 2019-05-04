README

Project title: Cloud classification over the Arctic
A little info about your project and/ or overview that explains what the project is about
In this project, we are building a classification model to distinguish the presence or absence of clouds in the images collected by the MISR sensor abroad the NASA satellite Terra. 

Motivation
A short description of the motivation behind the creation and maintenance of the project. This should explain why the project exists.
In the future, we will be able to use this model to classify clouds from non-clouds on a large number of images. The ability to classify the presence of clouds will improve our understanding of the flow of visible and infrared radiation, and how these can help us studying the changes in arctic climate and its relationship with the increasing concentrations of atmospheric carbon dioxide.
# stat154-project2

EDA
We use corplot,ggplot, scatterplot, and line plot. We also look at distribution of features and CI of the data to assess the iid pattern of the data. We also interested in looking at interesting pattern and correlation between features. 

Data splitting (library: dplyr)
To prepare for training the data, we first need to do data cleaning by binding all the images into one dataframe and filtering out data that has unlabeled expert labels. Leaving out unlabeled expert labels will help us train our data better in distinguishing clouds from non-clouds.
After cleaning the data, we will split the entire data into three sets which are training, validation, and test. In this project, we take into account that the data is not i.i.d before splitting the data with two non-trivial different ways; described below:
Blocking: separate the whole image by n x n blocks. A function blocking were created with inputs image and n (number of blocks), dividing the range of x and y coordinate of the image by n and outputs n^2 divided blocks. Another function called split_way1 were created with inputs image and n that calls in the function blocking and randomly pick 75% of the blocks as training set, 12.5% as validation set, and 12.5% as test set. This function will returns splitted image.

?	split_way_1 function to incorporate the first non-trivial splitting method, which is by blocking, in order to remove the dependence of the data in ?x? and ?y? and enable random data splitting. 
?	A function blocking was created with inputs image and n (number of blocks), dividing the range of x and y coordinate of the image by n and outputs nxn divided blocks.  
?	split_way1 were created with inputs image and n that calls in the function blocking and randomly pick 75% of the blocks as training set, 12.5% as validation set, and 12.5% as test set. This function will returns list of splitted image. 
?	split_way_2 function to incorporate the second non-trivial splitting method, which is by smoothing, in order to remove the dependence of the data in ?x? and ?y? and enable random data splitting. 
?	A function smoothing was created with inputs image and n (number of the rolling window). Incorporating the function rollmeanr in package zoo in r for taking rolling average with n neighbors and taking the non-overlapping window to make the windows independent. By taking average in n2 neighboring data points, we are trying to create independent smoothed data points. The majority of expert labels from the n x n window is used to label the smoothed data points. Confusing data points with 5(or 6) labels of group 1 and 4(or 3) labels or group 2 are removed. This function returns the smoothed data points that are no longer highly correlated in x and y neighbors. 
?	split_way2 were created with inputs image and n that calls in the function smoothing and randomly pick 75% of the smoothed data as training set, 12.5% as validation set, and 12.5% as test set. This function will returns list of splitted image. 

Example usage:
split_way1(train_data, n=10)


Generic Cross Validation
The CVgeneric function can take on 6 inputs, which are generic classifier, training features, training labels, number of folds(k), loss function, and an additional input of splitting method 1 or 2 (with default of using method1 if not specified). 
(Recommended Input data = training data + validation data)
?	Generic classifier input includes the option of using ?lda?, ?qda?, ?logistic regression?, ?svm?, ?classification tree?, and ?random forest?.
?	Training features is the data frame with all the predictors. 
?	Requires variables x and y to be inputted along with the predictors for splitting the non-iid data. 
?	Training labels is the data on the expert labels
?	Number of folds with the default of 10-fold. 
?	Loss function with the default input of classification accuracy test with the function accuracy_test.
?	Splitting method with default of using blocking to split the training + validation data into blocks (CVgeneric_way1). If method 1 is used, hold-out data will be one the data in one block and the rest of the k-1 blocks will be used as fold data for model fitting. If CVgeneric function is called with CVgeneric_way2 as the input, the cross validation will be executed using method 2, which is smoothing, the training + validation data will be smoothed first and then randomly separated into folds. 
The output will be a vector of all the k-losses score. The CVgeneric function can be found on the github.
Example of usage, using second method of splitting, use lda classifier:
CVgeneric(?lda?, training_features, training_labels, 5, accuracy_test, CVgeneric_way2) 

Comparing different methods using ROC curves to look at the AUC. 
We also find optimal treshold to set the cut-off value for our models by using ROC curve

We also explore Precision, Recall, and F1 scores to determine the best model.

Diagnostic 
We use various of plots to show convergence and stability of models visually. It is also used to identify misclassified points and explore the patterns. We hope that this part would suggest improvement approach for future use to get better accuracy. 

Packages used:
dplyr
ggplot2
GGally
reshape2
MASS
pROC
ROCR
randomForest
zoo
plyr
caret
car
SparseM
e1071
ROCR
graphics
rpart

