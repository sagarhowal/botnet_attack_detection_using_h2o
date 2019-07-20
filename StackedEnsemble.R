#Load dataset if not already in environment
#data_unsupervised <- read.csv("data_small.csv")


#Implementation of Stacked Ensemble using H2o

#Install h2o ensemble
#library(devtools)
#install_github("h2oai/h2o-3/h2o-r/ensemble/h2oEnsemble-package")



#### Start H2O Cluster
library(h2oEnsemble)  # This will load the `h2o` R package as well
h2o.init(nthreads = 2)  # Start an H2O cluster with nthreads = num cores on your machine
h2o.removeAll() # Clean slate - just in case the cluster was already running


#### Load Data into H2O Cluster
#
#First, import a sample binary outcome train and test set into the H2O cluster.
train <- h2o.importFile("train.csv")
test <- h2o.importFile("test.csv")
y <- c("C1", "class")
x <- setdiff(names(train), y)
y <- "class"


train[,y] <- as.factor(train[,y])  
test[,y] <- as.factor(test[,y])

#### Specify Base Learners & Metalearner
#For this example, we will use the default base learner library for `h2o.ensemble`, which includes the default H2O GLM, Random Forest, GBM and Deep Neural Net (all using default model parameter values).  We will also use the default metalearner, the H2O GLM.
#
learner <- c("h2o.glm.wrapper", "h2o.randomForest.wrapper", 
             "h2o.gbm.wrapper")
metalearner <- "h2o.glm.wrapper"

#### Train an Ensemble
#Train the ensemble (using 5-fold internal CV) to generate the level-one data.  Note that more CV folds will take longer to train, but should increase performance.
fit <- h2o.ensemble(x = x, y = y, 
                    training_frame = train, 
                    family = "binomial", 
                    learner = learner, 
                    metalearner = metalearner,
                    cvControl = list(V = 5))

#### Predict 
#Generate predictions on the test set.
pred <- predict(fit, test)
predictions <- as.data.frame(pred$pred)[,3]  #third column is P(Y==1)
labels <- as.data.frame(test[,y])[,1]

pred_class <- ifelse(predictions > 0.5, 1, 0)
pred_class <- as.factor(pred_class)

#load temporary test dataframe
test_temp <- read.csv("test.csv")
test_class <- test_temp[length(test_temp)]
test_class <- as.factor(test_class$class)
rm(test_temp)



library(caret)
confusionMatrix(pred_class, test_class)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    0    1
# 0 4172    1
# 1    0 4479
# 
# Accuracy : 0.9999     
# 95% CI : (0.9994, 1)
# No Information Rate : 0.5178     
# P-Value [Acc > NIR] : <2e-16     
# 
# Kappa : 0.9998     
# 
# Mcnemar's Test P-Value : 1          
#                                      
#             Sensitivity : 1.0000     
#             Specificity : 0.9998     
#          Pos Pred Value : 0.9998     
#          Neg Pred Value : 1.0000     
#              Prevalence : 0.4822     
#          Detection Rate : 0.4822     
#    Detection Prevalence : 0.4823     
#       Balanced Accuracy : 0.9999     
#                                      
#        'Positive' Class : 0 

#### Model Evaluation
#
#Since the response is binomial, we can use Area Under the ROC Curve ([AUC](https://www.kaggle.com/wiki/AUC)) to evaluate the model performance.  We first generate predictions on the test set and then calculate test set AUC using the [cvAUC](https://cran.r-project.org/web/packages/cvAUC/) R package.
#
##### Ensemble test set AUC
library(cvAUC)
cvAUC::AUC(predictions = predictions, labels = labels)
# 1
#
##### Base learner test set AUC
#We can compare the performance of the ensemble to the performance of the individual learners in the ensemble.  Again, we use the `AUC` utility function to calculate performance.
#
L <- length(learner)
auc <- sapply(seq(L), function(l) cvAUC::AUC(predictions = as.data.frame(pred$basepred)[,l], labels = labels)) 
data.frame(learner, auc)

#                    learner       auc
# 1          h2o.glm.wrapper 0.9999994
# 2 h2o.randomForest.wrapper 1.0000000
# 3          h2o.gbm.wrapper 1.0000000


#Try to optimize further

#### Specifying new learners
#
#Trying again with a more extensive set of base learners.  The **h2oEnsemble** packages comes with four functions by default that can be customized to use non-default parameters. 
#
#generating new custom learner wrappers:
#
h2o.glm.1 <- function(..., alpha = 0.0) h2o.glm.wrapper(..., alpha = alpha)
h2o.glm.2 <- function(..., alpha = 0.5) h2o.glm.wrapper(..., alpha = alpha)
h2o.glm.3 <- function(..., alpha = 1.0) h2o.glm.wrapper(..., alpha = alpha)
h2o.randomForest.1 <- function(..., ntrees = 200, nbins = 50, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
h2o.randomForest.2 <- function(..., ntrees = 200, sample_rate = 0.75, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
h2o.randomForest.3 <- function(..., ntrees = 200, sample_rate = 0.85, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
h2o.randomForest.4 <- function(..., ntrees = 200, nbins = 50, balance_classes = TRUE, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, balance_classes = balance_classes, seed = seed)
h2o.gbm.1 <- function(..., ntrees = 100, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, seed = seed)
h2o.gbm.2 <- function(..., ntrees = 100, nbins = 50, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
h2o.gbm.3 <- function(..., ntrees = 100, max_depth = 10, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
h2o.gbm.4 <- function(..., ntrees = 100, col_sample_rate = 0.8, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.5 <- function(..., ntrees = 100, col_sample_rate = 0.7, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.6 <- function(..., ntrees = 100, col_sample_rate = 0.6, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.7 <- function(..., ntrees = 100, balance_classes = TRUE, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, balance_classes = balance_classes, seed = seed)
h2o.gbm.8 <- function(..., ntrees = 100, max_depth = 3, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)


#Optional Deep Learning Models
h2o.deeplearning.1 <- function(..., hidden = c(500,500), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.2 <- function(..., hidden = c(200,200,200), activation = "Tanh", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.3 <- function(..., hidden = c(500,500), activation = "RectifierWithDropout", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.4 <- function(..., hidden = c(500,500), activation = "Rectifier", epochs = 50, balance_classes = TRUE, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, balance_classes = balance_classes, seed = seed)
h2o.deeplearning.5 <- function(..., hidden = c(100,100,100), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.6 <- function(..., hidden = c(50,50), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.7 <- function(..., hidden = c(100,100), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
#




#Let's grab a subset of these learners for our base learner library and re-train the ensemble.
#
#### Customized base learner library
learner <- c("h2o.glm.wrapper",
             "h2o.randomForest.1", "h2o.randomForest.2",
             "h2o.gbm.1", "h2o.gbm.6", "h2o.gbm.8",
             "h2o.deeplearning.1", "h2o.deeplearning.6", "h2o.deeplearning.7")

#
#Train with new library:
fit <- h2o.ensemble(x = x, y = y, 
                    training_frame = train,
                    family = "binomial", 
                    learner = learner, 
                    metalearner = metalearner,
                    cvControl = list(V = 5))



# Generate predictions on the test set:
pred <- predict(fit, test)
predictions <- as.data.frame(pred$pred)[,3]
labels <- as.data.frame(test[,y])[,1]
#
#Evaluate the test set performance: 
cvAUC::AUC(predictions = predictions , labels = labels)
# 0.7904223
#We see an increase in performance by including a more diverse library.
#
#Base learner test AUC (for comparison)
L <- length(learner)
auc <- sapply(seq(L), function(l) cvAUC::AUC(predictions = as.data.frame(pred$basepred)[,l], labels = labels)) 
data.frame(learner, auc)


#
#What happens to the ensemble if we remove some of the weaker learners?  
#Here is a more stripped down version of the base learner library used above:
learner <- c("h2o.randomForest.1", "h2o.randomForest.2",
             "h2o.gbm.1", "h2o.gbm.6", "h2o.gbm.8")
#
#Again re-train the ensemble:
fit <- h2o.ensemble(x = x, y = y, 
                    training_frame = train,
                    family = "binomial", 
                    learner = learner, 
                    metalearner = metalearner,
                    cvControl = list(V = 5))

# Generate predictions on the test set
pred <- predict(fit, test)
predictions <- as.data.frame(pred$pred)[,3]  #third column, p1 is P(Y==1)
labels <- as.data.frame(test[,y])[,1]

# Ensemble test AUC 
cvAUC::AUC(predictions = predictions , labels = labels)

#
#We actually lose performance by removing the weak learners!  This demonstrates the power of stacking.
#
#At first thought, you may assume that removing less performant models would increase the perforamnce of the ensemble.  However, each learner has it's own unique contribution to the ensemble and the added diversity among learners usually improves performance.  The Super Learner algorithm learns the optimal way of combining all these learners together in a way that is superior to other combination/blending methods.
#

#### All done, shutdown H2O
h2o.shutdown(prompt=FALSE)


