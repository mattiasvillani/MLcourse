library(caret)  # for fitting model with different hyperparameters using CV
library(MLeval) # for plotting ROC curves and more
library(mlbench) # for the Sonar data
library("RColorBrewer") # for pretty colors
colors = brewer.pal(12, "Paired")[c(1,2,7,8,3,4,5,6,9,10)];
options(repr.plot.width = 12, repr.plot.height = 12, repr.plot.res = 100) # plot size

data(Sonar)

# Split the data in training and test
set.seed(123)
inTrain <- createDataPartition(
  y = Sonar$Class,
  p = .75,
  list = FALSE
)
training <- Sonar[ inTrain,]
testing  <- Sonar[-inTrain,]

# Set up the options for the hyperparameter tuning, e.g. cross-validation 
ctrl <- trainControl(
  method = "cv", # default is bootstrap. Other option: "repeatedcv"
  number = 10,            # number of CV folds
  classProbs = TRUE,     
  summaryFunction = twoClassSummary,
  savePredictions = TRUE  # Need to save predictions for MLeval's ROC curve
)

# Fit glmnet models for each combination of hyperparameters
set.seed(123) # always set the seed before train to get same folds for all models
glmnetFit <- train(
  Class ~ .,
  data = training,
  method = "glmnet",
  preProc = c("center", "scale"),
  tuneLength = 10, # 5 values for each hyperparameter gives 5*5 = 25 combinations
  #tuneGrid = expand.grid(alpha = c(0,1), lambda = c(0.01,0.02)),
  trControl = ctrl,
  metric = "ROC"
)
glmnetFit$modelInfo$parameters # List the tuning parameter for the model
glmnetFit
ggplot(glmnetFit)

# Looking at the resampling variability
glmnetFit$bestTune    # Optimal tuning parameter
glmnetFit$results     # Performance for each tuning param averaged over folds.
glmnetFit$preProcess  # List of the pre-processing that was been done.
glmnetFit$resample    # Results for each CV fold


# Fit random forest models for each combination of hyperparameters
set.seed(123)
rfFit <- train(
  Class ~ .,
  data = training,
  method = "rf",
  preProc = c("center", "scale"),
  tuneLength = 10,
  trControl = ctrl,
  metric = "ROC"
)
ggplot(rfFit)

# Fit gradient boosted tree for some selected combination of hyperparameters
gbmGrid <-  expand.grid(interaction.depth = c(2, 10, 15), 
                        n.trees = c(100,500,1000), 
                        shrinkage = 0.1,
                        n.minobsinnode = 20)
set.seed(123)
gbmFit <- train(
  Class ~ .,
  data = training,
  method = "gbm",
  preProc = c("center", "scale"),
  tuneGrid = gbmGrid,
  trControl = ctrl,
  metric = "ROC"
)
ggplot(gbmFit)

# k-NN 
set.seed(123)
knnFit <- train(
  Class ~ .,
  data = training,
  method = "knn",
  preProc = c("center", "scale"),
  tuneGrid = data.frame(k = 1:10),
  trControl = ctrl,
  metric = "ROC"
)
ggplot(knnFit)

# Use the MLeval package to plot the ROC curve for each model
glmnetEval = evalm(glmnetFit, plots='r', rlinethick=0.8, fsize=8) # plots='r' gives ROC
rfEval = evalm(rfFit, plots='r', rlinethick=0.8, fsize=8)
gbmEval = evalm(gbmFit, plots='r', rlinethick=0.8, fsize=8)
knnEval = evalm(knnFit, plots='r', rlinethick=0.8, fsize=8)

# Or compare ROC curves in the same plot
glmnet_rf_Eval = evalm(list(glmnetFit, rfFit, gbmFit, knnFit), 
                       gnames = c("elastic net","random forest","gradBoost", "k-NN"), 
                       col = colors[c(1,2,3,4)],
                       plots='r', rlinethick=0.8, fsize=8, title = "Sonar data")

# Prediction on the test data
glmnetClasses <- predict(glmnetFit, newdata = testing)
confTest_glmnet = confusionMatrix(data = glmnetClasses, testing$Class)

rfClasses <- predict(rfFit, newdata = testing)
confTest_rf = confusionMatrix(data = rfClasses, testing$Class)

gbmClasses <- predict(gbmFit, newdata = testing)
confTest_gbm = confusionMatrix(data = gbmClasses, testing$Class)

# Summarizing all the test performance on all methods
res = rbind(confTest_glmnet$overall, confTest_rf$overall, confTest_gbm$overall)
row.names(res) <- c("glmnet","rf","gbm")
res
