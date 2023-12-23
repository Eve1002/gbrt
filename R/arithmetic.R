
#' Imitate gradient boost regression trees
#'
#' @param train_data split the original data randomly into 2 datasets, one is training dataset, train data is its features
#' @param test_data split the original data randomly into 2 datasets, another one is testing dataset, test_data is its features
#' @param target dependent variable to predict
#' @param learning_rates it controls how much we adjust the weights of our network with respect to the loss gradient
#' @param n_trees number of trees
#'
#' @return prediction of the target variable
#' @export
#'
#' @examples
#' library(rpart)
#' library(tidyr)
#' library(tidyverse)
#' library(gbm)
#' library(palmerpenguins)
#' library(caret)
#' data("penguins")
#' penguins <- na.omit(penguins)
#' penguins <- penguins %>% select(-year)
#' set.seed(123)
#' indices <- sample(1:nrow(penguins), 0.7 * nrow(penguins))
#' train_data <- penguins[indices, ]
#' test_data <- penguins[-indices, ]
#' target <- "body_mass_g"
#' predictions <- gbrt(train_data,test_data,target)
gbrt <- function(train_data,test_data, target, learning_rates = 0.1, n_trees = 100) {
  models <- list() # store the sequence of the trees that will be created
  residuals <- train_data[[target]] # Initially is just the target value, but later will be updated to (observed - predicted)

  # Iterate n_trees times to build the sequence of trees
  for (i in 1:n_trees) {
    tree <- rpart(residuals ~ ., data = train_data, method = "anova") # use anova for regression problem
    predictions <- predict(tree, train_data)

    # The most crucial step in GBRT! Update the previous residual by subtracting the scaled predictions
    residuals <- residuals - learning_rates * predictions

    # Update the target col in the dataset to new residuals. In the next iteration, this will be the new response variable.
    train_data[[target]] <- residuals
    models[[i]] <- tree
  }
    # make predictions
    predictions <- rep(0, nrow(test_data))
    for (tree in models) {
    predictions <- predictions + learning_rates * predict(tree, test_data)
  }

    return(predictions)
}





#' use cross-validation to tune hyperparameter(learning rate)
#'
#' @param data dataset use here
#' @param target dependent variable to predict
#' @param learning_rates it controls how much we adjust the weights of our network with respect to the loss gradient
#' @param n_trees number of trees
#' @param folds number of folds
#'
#' @return contains information about the learning rate and its corresponding rmse
#' @export
#'
#' @examples
#' library(rpart)
#' library(tidyr)
#' library(tidyverse)
#' library(gbm)
#' library(palmerpenguins)
#' library(caret)
#' data("penguins")
#' penguins <- na.omit(penguins)
#' penguins <- penguins %>% select(-year)
#' learning_rates <- c(0.01, 0.05, 0.1, 0.2, 0.5)
#' cv_results <- gbrt_cv(penguins, target = "body_mass_g", learning_rates,n_trees = 100,folds = 5)
gbrt_cv <- function(data, target, learning_rates,n_trees = 100,folds = 5) {
  set.seed(123)  # Set seed for reproducibility

  # Split data to test and train
  indices <- sample(1:nrow(data), 0.7 * nrow(data))
  train_data <- data[indices, ]
  test_data <- data[-indices, ]

  # Initialize results data frame
  cv_results <- data.frame(learning_rates = numeric(), rmse = numeric())

  for (lr in learning_rates) {
    # Perform k-fold cross-validation
    folds_indices <- createFolds(train_data[[target]], k = folds, list = TRUE, returnTrain = FALSE)
    rmse_values <- numeric()

    for (fold in folds_indices) {
      train_indices <- unlist(fold)
      cv_train_data <- train_data[train_indices, ]
      cv_test_data <- train_data[-train_indices, ]

      # Make predictions
      predictions <- gbrt(cv_train_data, cv_test_data,target)

      # Evaluate RMSE
      rmse <- function(actual, predicted) {
        sqrt(mean((actual - predicted)^2))
      }
      rmse_value <- rmse(cv_test_data[[target]], predictions)
      rmse_values <- c(rmse_values, rmse_value)
    }

    # Calculate average RMSE across folds
    avg_rmse <- mean(rmse_values)

    # Store results
    cv_results <- rbind(cv_results, data.frame(learning_rates = lr, rmse = avg_rmse))
  }

  # Find the learning rate with the minimum RMSE
  best_lr <- cv_results$learning_rates[which.min(cv_results$rmse)]

  # Print results
  print(cv_results)
  cat("Best Learning Rate:", best_lr, "\n")

  # Return the results
  return(cv_results)
}
