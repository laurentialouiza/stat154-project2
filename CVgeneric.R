library(dplyr)
library(zoo)
library(plyr)
library(ggplot2)
library(caret)
library(MASS)
library(car)
library(SparseM)
library(e1071)
library(ROCR)

set.seed(123)
### split data way 1 ------------------------------------------
blocking <- function(image,n) {
  min_x <- min(image$x)
  min_y <- min(image$y)
  max_x <- max(image$x)
  max_y <- max(image$y)
  
  min_x_i <- matrix(rep(seq(min_x, max_x, length.out = n+1),n), nrow = n, byrow = TRUE)
  min_y_i <- matrix(rep(seq(min_y, max_y, length.out = n+1),n), ncol = n, byrow = FALSE)
  block <- list()
  for (i in 1:n) {
    for (j in 1:n){
      index <- j-1 + n*(i-1) + 1
      block[[index]] <- image[ which(image$x >= min_x_i[i,j] & image$x < min_x_i[i,j+1] & image$y >= min_y_i[i,j] & image$y < min_y_i[i+1, j]),]
    }
    ##adding data in with max of x values
    block[[index]] <- image[ which(image$x >= min_x_i[i,j] & image$x <= min_x_i[i,j+1] & image$y >= min_y_i[i,j] & image$y < min_y_i[i+1, j]),]
  }
  ##############################
  ##adding the data in the max of y values
  i=n
  for (j in 1:n) {
    index <- j-1 + n*(i-1) + 1
    block[[index]] <- image[ which(image$x >= min_x_i[i,j] & image$x < min_x_i[i,j+1] & image$y >= min_y_i[i,j] & image$y <= min_y_i[i+1, j]),]
  } 
  block[[index]] <- image[ which(image$x >= min_x_i[i,j] & image$x <= min_x_i[i,j+1] & image$y >= min_y_i[i,j] & image$y <= min_y_i[i+1, j]),]
  ##############################
  return(block)
}

split_way1 <- function(image, n) {
  block <- blocking(image, n)
  
  #randomly pick 12 blocks = 75% training data
  num_blocks_train <- floor(n*n*0.75)
  train_blocks <- sample(1:(n^2), num_blocks_train)
  
  train_image <- data.frame()
  for (i in train_blocks){
    one_block <- block[[i]]
    train_image <- rbind(train_image, one_block)
  }
  
  indices <- seq(1:(n^2))
  rest_blocks <- indices[-train_blocks] 
  num_blocks_val <- floor(n*n*0.125)
  val_blocks <- sample(rest_blocks, num_blocks_val)
  val_image <- data.frame()
  for (i in val_blocks){
    one_block <- block[[i]]
    val_image <- rbind(val_image, one_block)
  }
  
  test_blocks <- rest_blocks[which(rest_blocks != val_blocks[1] & rest_blocks != val_blocks[2])]
  test_image <- data.frame()
  for (i in test_blocks){
    one_block <- block[[i]]
    test_image <- rbind(test_image, one_block)
  }
  result <- list(train_image, val_image, test_image)
  return(result)
}

### split data way 2 ----------------------------------------------------------
smoothing <- function(image, n = 9) {
  sorted_image <- image[order(image$x, image$y),]
  r <- rollmeanr(sorted_image, n)
  r <- as.data.frame(r)
  
  index <- seq(1, nrow(r), n)
  averaged_image <- as.data.frame(r[index,])
  ## replacing the expert label by the majority label of the 9 data points
  library(plyr)
  lab <- c()
  seq <- seq(1,nrow(image), n)
  for (i in 1:length(seq)) {
    df <- table(sorted_image$expert_label[seq[i]:(seq[i]+n-1)])
    #df <- count(sorted_image$expert_label[1+(9*(i-1)):9+(9*(i-1))])
    lab[i] <- names(which.max(df))
    if((max(df)) < 7 ){
      lab[i] <- NA
    }
  }
  lab <- lab[1:length(index)]
  averaged_image$expert_label <- lab
  #  is.na(averaged_image$expert_label)
  averaged_image <- na.omit(averaged_image)
  return(averaged_image) 
}

split_way2 <- function(image,n =9) {
  averaged_image <- smoothing(image,n)
  
  num_train <- floor(nrow(averaged_image)*0.75)
  train_indices <- sample(1:nrow(averaged_image), num_train)
  train_data_2 <- averaged_image[train_indices, ]
  
  indices <- seq(1:nrow(averaged_image))
  rest_indices <- indices[-train_indices] 
  num_val <- floor(nrow(averaged_image)*0.125)
  val_indices <- sample(rest_indices, num_val)
  val_data_2 <- averaged_image[val_indices, ]
  
  test_indices <- indices[-train_indices]
  test_indices <- test_indices[-val_indices]
  test_data_2 <- averaged_image[test_indices, ]
  
  result <- list(train_data_2, val_data_2, test_data_2)
  return(result)
}

###### CV function ---------------------------------------------------------------
CVgeneric_way1 <- function(generic_classifier, training_features, training_labels, num_folds = 10, loss_func = accuracy_test) {
  #blocks <- blocking(training_features, n = num_folds)
  full_train <- cbind(training_features, training_labels)
  colnames(full_train)[colnames(full_train) == "training_labels"] <- "expert_label"
  blocks <- blocking(full_train, n = num_folds)
  block_folds <- createFolds(blocks, k = num_folds)
  features <- colnames(training_features)
  features <- features[!features %in% c("x", "y")]
  
  result <- rep(NA, length(block_folds))
  for (f in 1: length(block_folds)){
    holdout_data <- data.frame()
    for (i in 1: length(block_folds[[f]])){
      num <- block_folds[[f]][i]
      holdout_data <- rbind(holdout_data, blocks[num][[1]])
    }
    
    fold_data <- full_train[!((rownames(full_train)) %in% as.numeric(rownames(holdout_data))),]
    #accuracy_folds[f] <- lda_classifier(fold_data, holdout_data)
    
    if (generic_classifier == "lda") {
      result[f] = lda_classifier(features, fold_data, holdout_data, loss_func)
    }
    if (generic_classifier == "qda") {
      result[f] = qda_classifier(features, fold_data, holdout_data, loss_func)
    }
    if (generic_classifier == "logistic regression") {
      result[f] = logit_classifier(features, fold_data, holdout_data, loss_func)
    }
    if (generic_classifier == "classification tree") {
      result[f] = classification_tree_classifier(features, fold_data, holdout_data, loss_func)
    }
    if (generic_classifier == "svm") {
      result[f] = svm_classifier(features, fold_data, holdout_data, loss_func)
    } 
    if (generic_classifier == "random forest") {
      result[f] = random_forest_classifier(features, fold_data, holdout_data, loss_func)
    }
  }
  return(result)
}

CVgeneric_way2 <- function(generic_classifier, training_features, training_labels, num_folds, loss_func) {
  full_train <- cbind(training_features, training_labels)
  colnames(full_train)[colnames(full_train) == "training_labels"] <- "expert_label"
  smooth_data <- smoothing(full_train, n = num_folds)
  
  folds <- createFolds(smooth_data$expert_label, k = num_folds)
  features <- colnames(training_features)
  features <- features[!features %in% c("x", "y")]
  
  result <- rep(NA, length(folds))
  for (f in 1: length(folds)){
    holdout_data <- smooth_data[folds[[f]],]
    fold_data <- smooth_data[-folds[[f]],]
    
    if (generic_classifier == "lda") {
      result[f] = lda_classifier(features, fold_data, holdout_data, loss_func)
    }
    if (generic_classifier == "qda") {
      result[f] = qda_classifier(features, fold_data, holdout_data, loss_func)
    }
    if (generic_classifier == "logistic regression") {
      result[f] = logit_classifier(features, fold_data, holdout_data, loss_func)
    }
    if (generic_classifier == "classification tree") {
      result[f] = classification_tree_classifier(features, fold_data, holdout_data, loss_func)
    }
    if (generic_classifier == "svm") {
      result[f] = svm_classifier(features, fold_data, holdout_data, loss_func)
    } 
    if (generic_classifier == "random forest") {
      result[f] = random_forest_classifier(features, fold_data, holdout_data, loss_func)
    }
  }
  return(result)
}

CVgeneric <- function(generic_classifier, training_features, training_labels, num_folds, loss_func, method = CVgeneric_way1) {
  result <- method(generic_classifier, training_features, training_labels, num_folds, loss_func)
  return(result)
}
##### CLASSIFIERS 
lda_classifier <- function(features, fold_data, holdout_data, loss_func){
  formula <- as.formula(paste("expert_label", "~", paste(features, collapse = "+"), sep = ""))
  fold_lda <- lda(formula , data = fold_data)
  holdout_pred <- predict(fold_lda, holdout_data)$class
  result <- loss_func(holdout_data$expert_label, holdout_pred)
  return(result)
}

qda_classifier <- function(features, fold_data, holdout_data, loss_func){
  formula <- as.formula(paste("expert_label", "~", paste(features, collapse = "+"), sep = ""))
  fold_qda <- qda(formula, data = fold_data)
  holdout_pred <- predict(fold_qda, holdout_data)$class
  result <- loss_func(holdout_data$expert_label, holdout_pred)
  return(result)
}
logit_classifier <- function(features, fold_data, holdout_data, loss_func){
  fold_data$expert_label <- as.factor(fold_data$expert_label)
  formula <- as.formula(paste("expert_label", "~", paste(features, collapse = "+"), sep = ""))
  logit <- glm(formula, data=fold_data, family='binomial')
  holdout_pred <- predict(logit, newdata=holdout_data, type='response')
  pred_logit <- rep(-1,length(holdout_pred))
  pred_logit[holdout_pred>=0.5] <- 1
  result <- loss_func(holdout_data$expert_label, pred_logit)
  return(result)
}
svm_classifier <- function(features, fold_data, holdout_data, loss_func){
  ##
  reduced_fold_data <- sample_n(fold_data, 0.1*nrow(fold_data))
  reduced_fold_data$expert_label <- as.factor(reduced_fold_data$expert_label)
  cut = 0.0879152 ##get through the ROC curve of several cost values
  formula <- as.formula(paste("expert_label", "~", paste(features, collapse = "+"), sep = ""))
  svm_model <- svm(formula, data=reduced_fold_data, scale= FALSE, kernel = 'radial', cost = cut)
  holdout_pred <- predict(svm_model, holdout_data)
  accuracy <- loss_func(holdout_data$expert_label, holdout_pred)
  return(accuracy)
}
classification_tree_classifier <- function(features, fold_data, holdout_data, loss_func){
  formula <- as.formula(paste("expert_label", "~", paste(features, collapse = "+"), sep = ""))
  tree <- rpart(formula, data = fold_data, method = "class")
  tree_pruned<- prune(tree, cp=tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"])
  holdout_pred <- predict(tree_pruned, holdout_data, type = "class")
  result <- loss_func(holdout_data$expert_label, holdout_pred)
  return(result)
}

random_forest_classifier <- function(features, fold_data, holdout_data, loss_func) {
  fold_data$expert_label <- as.factor(fold_data$expert_label)
  formula <- as.formula(paste("expert_label", "~", paste(features, collapse = "+"), sep = ""))
  set.seed(123)
  rf <- randomForest(formula, data = fold_data, mtry = 2, nodesize = 20)
  holdout_pred <- predict(rf , holdout_data, type = "class")
  result <- accuracy_test(holdout_data$expert_label, holdout_pred)
  return(result)
}
accuracy_test <- function(actual, predicted) {
  accuracy <- sum(actual == predicted)/length(actual)
  return(accuracy)
}

