# Using keras to fit a neural network to the MNIST data
# Based on the keras blog post https://tensorflow.rstudio.com/guide/keras/

library(keras)

# Load the MNIST data
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

dim(x_train)    # 60000-by-28-by-28 3D array with 60000 training images (28-by-28 pixels)
length(y_train) # 60000 element vector with training labels

# reshape
x_train <- array_reshape(x_train, c(nrow(x_train), 784)) # 60000-by-784 matrix
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
# rescale
x_train <- x_train / 255
x_test <- x_test / 255

# One-hot versions of the labels (0-9)
y_train <- to_categorical(y_train, 10) # 60000-by-10 matrix, each row is one-hot
y_test <- to_categorical(y_test, 10)

# Set up the model using the pipe %>% command to chain things together
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')
summary(model)

# Compile the model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# Fit the model for 30 epochs using batches of 128 images
history <- model %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
)
plot(history)

# Evaluate performance on test data (10000 hold out images)
model %>% evaluate(x_test, y_test)

# Predict the test data using pipes (%>%)
model %>% predict_classes(x_test)

# Predict the test data without using pipes [note: predict_classes() in blog is deprecated.]
yProbs = predict(model, x_test)
yPreds = apply(yProbs, 1, function(x) which.max(x)-1)

# Plot the predictive distribution for first test case
barplot(names.arg = 0:9, yPreds[1,], col = "blue")
