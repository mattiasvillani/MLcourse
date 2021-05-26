install.packages("tm") # Text mining
library(tm)

# The data can be found on github here:
# https://github.com/sloria/textfeel-web

reviewsNeg <- Corpus(DirSource("/home/mv/Dropbox/Teaching/MLcourse/Data/movie_reviews/neg", encoding = "UTF-8"))
reviewsPos <- Corpus(DirSource("/home/mv/Dropbox/Teaching/MLcourse/Data/movie_reviews/pos", encoding = "UTF-8"))
inspect(reviewsNeg[1:2])
reviewsNeg[[10]]
reviews <- c(reviewsNeg,reviewsPos)

# The tm package can do a lot of text manipulations:
#reviews <- tm_map(reviews, stripWhitespace)
#reviews <- tm_map(reviews, tolower)
#reviews <- tm_map(reviews, removeWords, stopwords("english"))
#reviews <- tm_map(reviews, stemDocument)

dtmReviews <- DocumentTermMatrix(reviews, control = list(removePunctuation = TRUE, 
                                                         stopwords = TRUE, 
                                                         minDocFreq = 5, 
                                                         removeNumbers=TRUE ))
?termFreq
?getTokenizers

# Convert to a matrix
dtmMat <- as.matrix(dtmReviews)

# Inspect the DTM
dim(dtmMat)
dtmReviews
inspect(dtmReviews[1:10,10:100])

# Terms that occure at least 1000 times (>1000 tokens per word type)
freqWords <- findFreqTerms(dtmReviews, 1000)

# Only use these word types
dtmReduced <- DocumentTermMatrix(reviews, list(dictionary = freqWords))

# Create a data.frame for supervised learning
dtmDataFrame <- as.data.frame(as.matrix(dtmReduced))

y = c(rep('Neg',1000),rep('Pos',1000)) # Creates the response vector (i.e. the labels)

# Shuffling the data randomly, creating training and test datasets
set.seed(42)
randomIdx <- sample(2000)
y <- y[randomIdx]
X <- dtmDataFrame[randomIdx,]

# Training on 1500 observations - testing on 500 observations
yTrain <- as.factor(y[1:1500])
XTrain <- X[1:1500,]
yTest <- as.factor(y[1501:2000])
XTest <- X[1501:2000,]

# Remove terms that were never seen in training data
appearAtLeastOnce <- colSums(XTrain)>0
XTrain <- XTrain[,appearAtLeastOnce]
XTest <- XTest[,appearAtLeastOnce]

# Training an off-the-shelf SVM
install.packages("e1071")
library(e1071) # Contains the svm() function
svmModel <- svm(x = XTrain, y = yTrain)
print(svmModel)

# Predict the test data
pred <- predict(svmModel, XTest)
confusionMatrix <- table(pred, yTest)
Precision <- confusionMatrix[1,1]/sum(confusionMatrix[1,])
Recall <- confusionMatrix[1,1]/sum(confusionMatrix[,1])


# We can also easily use the caret package
library(caret)

# Fit a LASSO logistic regression
fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 3,
                           verboseIter = TRUE)

# Create dataset for caret package
dat <- cbind(y=yTrain, XTrain)

# Calculate confusion matrix
fit_logistic <- train(y ~.,
                      data = dat,
                      method = "glmnet",
                      trControl = fitControl,
                      family = "binomial")

# Predict values
pred_classes <- predict(fit_logistic, newdata = XTest)

# Calculate confusion matrix
confusionMatrix(data = pred_classes, yTest)

