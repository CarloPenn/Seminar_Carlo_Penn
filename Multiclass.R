###### This function does catboost and xgboost for multiclass classification tasks and outputs losses, name

# Helper function one hot encoding
one_hot_encode <- function(df) {
  df_encoded <- df %>%
    mutate(across(where(is.factor), as.character)) %>%  # Convert factors to characters
    mutate(across(where(is.character), ~ ifelse(is.na(.), "Missing", .))) %>%  # Replace NA with "Missing"
    mutate(across(where(is.character), as.factor))  # Convert back to factors
  
  # Create one-hot encoding manually
  df_one_hot <- model.matrix(~ . + 0, data = df_encoded, na.action = na.pass)
  
  return(as.data.frame(df_one_hot))  # Convert matrix back to dataframe
}


evaluate_models_mc <- function(data, dataset_name) {
  set.seed(123)
  
  target_col <- ncol(data)
  X <- data[, -target_col]
  y <- data[, target_col]
  
  X <- mutate_if(X, is.character, as.factor)
  
  # Convert target to numeric starting from 0 (required for CatBoost)
  y <- as.numeric(as.factor(y)) - 1  
  
  # Split data
  trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
  trainData <- X[trainIndex, ]
  trainLabel <- y[trainIndex]
  testData <- X[-trainIndex, ]
  testLabel <- y[-trainIndex]
  
  ###### CatBoost ######
  params_catboost <- list(
    iterations = 500,
    learning_rate = 0.02,
    depth = 10,
    loss_function = "MultiClass",
    eval_metric = "MultiClass",
    random_seed = 55
  )
  
  catboost_pool_train <- catboost.load_pool(trainData, label = trainLabel)
  catboost_pool_test <- catboost.load_pool(testData, label = testLabel)
  
  catboost_model <- catboost.train(catboost_pool_train, params = params_catboost)
  
  catboost_pred_prob <- catboost.predict(catboost_model, catboost_pool_test, prediction_type = "Probability")
  
  # Compute Multi-Class Log Loss
  log_loss_catboost <- MultiLogLoss(catboost_pred_prob, testLabel)
  
  # Convert probabilities to class predictions
  catboost_pred_class <- max.col(catboost_pred_prob) - 1
  zero_one_loss_catboost <- mean(catboost_pred_class != testLabel)
  
  ###### XGBoost ######

  # Entferne Spalten mit nur einer einzigartigen Kategorie oder komplett NA
  X <- X[, sapply(X, function(col) length(unique(na.omit(col))) > 1)]
  X <- mutate_if(X, is.character, as.factor)
  
  
  X <- one_hot_encode(X)
  #  trainIndex <- trainIndex[trainIndex <= nrow(X)]
  X_train <- X[trainIndex, ]
  X_test <- X[-trainIndex, ]
  trainLabel <- y[trainIndex]
  testLabel <- y[-trainIndex]

  # Convert to DMatrix
  class(X_train)
  # [1] "data.frame"
  dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = trainLabel)
  dtest <- xgb.DMatrix(data = as.matrix(X_test), label = testLabel)
  

  
  params_xgb <- list(
    objective = "multi:softprob",
    eval_metric = "mlogloss",
    eta = 0.02,
    max_depth = 10,
    num_class = length(unique(y)),  # Set number of classes
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
  xgb_pred_prob <- matrix(xgb_pred_prob, ncol = length(unique(y)), byrow = TRUE)
  
  log_loss_xgb <- MultiLogLoss(xgb_pred_prob, testLabel)
  
  xgb_pred_class <- max.col(xgb_pred_prob) - 1
  zero_one_loss_xgb <- mean(xgb_pred_class != testLabel)
  
  ###### LightGBM ######

  dtrain_lgb <- lgb.Dataset(data = as.matrix(X_train), label = trainLabel)
  dtest_lgb <- lgb.Dataset(data = as.matrix(X_test), label = testLabel)
  
  num_classes <- length(unique(trainLabel))
  
  params_lgb <- list(
    objective = "multiclass",   # Multiclass classification
    metric = "multi_logloss",   # Multiclass log loss
    num_class = num_classes,    # Number of classes
    learning_rate = 0.02,
    num_leaves = 31,
    max_depth = 10,
    seed = 55
  )
  
  lgb_model <- lgb.train(params_lgb, dtrain_lgb, nrounds = 500)

  
  lgb_pred_prob <- predict(lgb_model, as.matrix(X_test))
  lgb_pred_prob <- matrix(lgb_pred_prob, ncol = num_classes, byrow = FALSE)
  
  log_loss_lgb <- MultiLogLoss(lgb_pred_prob, testLabel)
  
  lgb_pred_class <- max.col(lgb_pred_prob) - 1  # LightGBM labels are 0-based
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

