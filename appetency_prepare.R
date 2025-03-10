evaluate_models_appetency <- function(data, dataset_name) {
  set.seed(123)
  

  
  
  target_col <- ncol(data)
  X <- data[, -target_col]
  y <- data[, target_col]
  
  X <- mutate_if(X, is.character, as.factor)
  
  # Split data
  trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
  trainData <- X[trainIndex, ]
  trainLabel <- y[trainIndex]
  testData <- X[-trainIndex, ]
  testLabel <- y[-trainIndex]
  
  # Convert labels to numeric
  trainLabel <- as.numeric(trainLabel)
  testLabel <- as.numeric(testLabel)
  
  ###### CatBoost ######
  params_catboost <- list(
    iterations = 500,
    learning_rate = 0.02,
    depth = 10,
    loss_function = 'Logloss',
    eval_metric = 'Logloss',
    random_seed = 55,
    od_type = 'Iter',
    metric_period = 50,
    od_wait = 20,
    use_best_model = TRUE
  )
  
  catboost_pool_train <- catboost.load_pool(trainData, label = trainLabel)
  catboost_pool_test <- catboost.load_pool(testData, label = testLabel)
  
  catboost_model <- catboost.train(catboost_pool_train, params = params_catboost)
  
  catboost_pred_prob <- catboost.predict(catboost_model, catboost_pool_test, prediction_type = "Probability")
  
  # Compute Log Loss & 0-1 Loss
  log_loss_catboost <- LogLoss(catboost_pred_prob, testLabel)
  catboost_pred_class <- ifelse(catboost_pred_prob > 0.5, 1, 0)
  zero_one_loss_catboost <- mean(catboost_pred_class != testLabel)
  
  
  ###### XGBoost ######
  
  data <- data
  X <- data[, -target_col]
  y <- data[, target_col]
  X <- X[, sapply(X, function(col) length(unique(na.omit(col))) > 1)]
  convert_factors_to_numeric <- function(df) {
    df[] <- lapply(df, function(x) {
      if (is.factor(x) || is.character(x)) {
        as.numeric(as.factor(x))  # Convert factor to numeric labels
      } else {
        x  # Keep numeric columns unchanged
      }
    })
    return(df)
  }
  
  # Apply function to your dataframe
  X <- convert_factors_to_numeric(X)
  
  X_train <- X[trainIndex, ]
  y_train <- y[trainIndex]
  X_test <- X[-trainIndex, ]
  y_test <- y[-trainIndex]
  
  
  dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
  dtest <- xgb.DMatrix(data = as.matrix(X_test), label = y_test)
  
  # Entferne Spalten mit nur einer Kategorie
  # na_cols <- which(colSums(is.na(X)) == nrow(X))
  # if(length(na_cols) != 0) {
  #   X <- X[, -na_cols]
  
  # }
  
  #  X <- model.matrix(~ . + 0, data = X)  # One-hot encoding
  #  if(length(trainIndex) < nrow(X)) {
  #    trainIndex <- trainIndex[trainIndex <= nrow(X)]
  #  }
  
  
  
  params_xgb <- list(
    objective = "binary:logistic",
    eval_metric = "logloss",
    eta = 0.02,
    max_depth = 10,
    seed = 55
  )
  
  xgb_model <- xgb.train(
    params = params_xgb,
    data = dtrain,
    nrounds = 500,
    watchlist = list(train = dtrain, test = dtest),
    early_stopping_rounds = 20
  )
  
  xgb_pred_prob <- predict(xgb_model, dtest)
  log_loss_xgb <- LogLoss(xgb_pred_prob, testLabel)
  
  xgb_pred_class <- ifelse(xgb_pred_prob > 0.5, 1, 0)
  zero_one_loss_xgb <- mean(xgb_pred_class != testLabel)
  
  ###### LightGBM ######
  dtrain_lgb <- lgb.Dataset(data = as.matrix(X_train), label = y_train)
  
  params_lgb <- list(
    objective = "binary",
    metric = "logloss",
    learning_rate = 0.02,
    num_leaves = 31,
    max_depth = 10,
    seed = 55
  )
  
  lgb_model <- lgb.train(params_lgb, dtrain_lgb, nrounds = 500)
  
  lgb_pred_prob <- predict(lgb_model, as.matrix(X_test))
  log_loss_lgb <- LogLoss(lgb_pred_prob, testLabel)
  
  lgb_pred_class <- ifelse(lgb_pred_prob > 0.5, 1, 0)
  zero_one_loss_lgb <- mean(lgb_pred_class != testLabel)
  
  # Return results
  return(list(
    name = dataset_name,
    log_loss_xgb = log_loss_xgb,
    zero_one_loss_xgb = zero_one_loss_xgb,
    log_loss_catboost = log_loss_catboost,
    zero_one_loss_catboost = zero_one_loss_catboost,
    log_loss_lgb = log_loss_lgb,
    zero_one_loss_lgb = zero_one_loss_lgb
  ))
}
