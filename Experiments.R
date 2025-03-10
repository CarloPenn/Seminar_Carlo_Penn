evaluate_models <- function(data, dataset_name) {
  set.seed(123)
  


  
  target_col <- ncol(data)
  X <- data[, -target_col]
  y <- data[, target_col]
  
  X <- mutate_if(X, is.character, as.factor)
  identical(click,data)
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
  X <- model.matrix(~ . + 0, data = X)
length(trainIndex)
nrow(X)
  X_train <- X[trainIndex, ]
  y_train <- y[trainIndex]
  X_test <- X[-trainIndex, ]
  y_test <- y[-trainIndex]
  

  dtrain <- xgb.DMatrix(data = X_train, label = y_train)
  dtest <- xgb.DMatrix(data = X_test, label = y_test)
  
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
  dtrain_lgb <- lgb.Dataset(data = X_train, label = y_train)
  
  params_lgb <- list(
    objective = "binary",
    metric = "logloss",
    learning_rate = 0.02,
    num_leaves = 31,
    max_depth = 10,
    seed = 55
  )
  
  lgb_model <- lgb.train(params_lgb, dtrain_lgb, nrounds = 500)
  
  lgb_pred_prob <- predict(lgb_model, X_test)
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


##### Adult Dataset #####
data <- read.arff("data/phpMawTba.arff")
unique(data$workclass)
# Replace '?' with NA in the entire dataset
data[data == "?"] <- NA
data$class <- gsub(">50K", 1, data$class)
data$class <- gsub("<=50K", 0, data$class)
data$class <- as.numeric(data$class)
Adult <- data
adult <- evaluate_models(data, dataset_name = "Adult")

##### Click Dataset #####
data <-  click

data$target <- gsub("1.0", 1, data$target)
data$target <- gsub("0.0", 0, data$target)
data$target <- as.numeric(data$target)
Click <- data
click <- evaluate_models(data, "Click")

##### Appetency Dataset ##### 
data <- read.arff("data/KDDCup09_appetency.arff")
table(data$APPETENCY)

data$APPETENCY <- gsub(-1, 0, data$APPETENCY)
data$APPETENCY <- gsub(1, 1, data$APPETENCY)
data$APPETENCY <- as.numeric(data$APPETENCY)
Appetency <- data
appetency <- evaluate_models_appetency(data, "Appetency")

##### Churn Dataset ##### 
data <- read.arff("data/churn.arff")

data$class <- gsub(1, 1, data$class)
data$class <- gsub(0, 0, data$class)
data$class <- as.numeric(data$class)
Churn <- data
churn <- evaluate_models(data, "Churn")

##### Internet Dataset ##### 
data <- read.arff("data/phpPIHVvG.arff")

data$class <- gsub("noad", 1, data$class)
data$class <- gsub("ad", 0, data$class)
data$class <- as.numeric(data$class)
Internet <- data
internet <- evaluate_models(data, "Internet")


##### Upselling Dataset ##### 
data <- read.arff("data/upselling.arff")
str(data)

data$UPSELLING <- gsub(1, 1, data$UPSELLING)
data$UPSELLING <- gsub(-1, 0, data$UPSELLING)
data$UPSELLING <- as.numeric(data$UPSELLING)
Upselling <- data
upselling <- evaluate_models(data, "Upselling")

##### kick Dataset ##### 
data <- read.arff("data/kick.arff")
data <- data[, c(setdiff(names(data), names(data)[1]), names(data)[1])]
str(data)


data$IsBadBuy <- gsub(1, 1, data$IsBadBuy)
data$IsBadBuy <- gsub(0, 0, data$IsBadBuy)
data$IsBadBuy <- as.numeric(data$IsBadBuy)
Kick <- data
kick <- evaluate_models_alternative(data, "Kick")

##### Anneal Dataset ( Multi class) ##### 
data <- read.arff("data/dataset_1_anneal.arff")

table(data$class)
data$class <- gsub(1, 0, data$class)
data$class <- gsub(2, 1, data$class)
data$class <- gsub(3, 2, data$class)
data$class <- gsub(5, 3, data$class)
data$class <- gsub("U", 4, data$class)

data$class <- as.numeric(data$class)
Anneal <- data
anneal <- evaluate_models_mc(data, dataset_name = "Anneal")

##### Digits Dataset ( Multi class) ##### 
data <- read.arff("data/dataset_28_optdigits.arff")
table(data$class)

data$class <- as.numeric(data$class)
Digits <- data
digits <- evaluate_models_mc(data, dataset_name = "Digits")

##### Bank marketing Dataset ##### 
data <- read.arff("data/phpkIxskf.arff")

data$Class <- gsub(1, 1, data$Class)
data$Class <- gsub(2, 0, data$Class)
data$Class <- as.numeric(data$Class)
Bank_marketing <- data
bank_marketing <- evaluate_models(data, dataset_name = "Bank Marketing")

##### Blood transfusion Dataset ##### 
data <- read.arff("data/php0iVrYT.arff")

data$Class <- gsub(1, 1, data$Class)
data$Class <- gsub(2, 0, data$Class)
data$Class <- as.numeric(data$Class)
Blood_transfusion <- data
blood_transfusion <- evaluate_models(data, dataset_name = "Blood transfusion")

##### Gas concentration Dataset (Multi class)  ##### 
data <- read.arff("data/phpN4gaxw.arff")
table(data$Class)

data$Class <- gsub(1, 0, data$Class)
data$Class <- gsub(2, 1, data$Class)
data$Class <- gsub(3, 2, data$Class)
data$Class <- gsub(4, 3, data$Class)
data$Class <- gsub(5, 4, data$Class)
data$Class <- gsub(6, 5, data$Class)

data$Class <- as.numeric(data$Class)
Gas_concentration <- data
gas_concentration <- evaluate_models_mc(data, dataset_name = "Gas concentration")

##### Car Dataset (Multi class) ##### 
data <- read.arff("data/php2jDIhh.arff")
table(data$class)

data$class <- gsub("unacc", 0, data$class)
data$class <- gsub("vgood", 1, data$class)
data$class <- gsub("acc", 2, data$class)
data$class <- gsub("good", 3, data$class)

data$class <- as.numeric(data$class)
Car <- data
car <- evaluate_models_mc(data, dataset_name = "car")

##### Cover type Dataset (Multi class)  ##### 
data <- read.arff("data/phpQOf0wY.arff")
table(data$class)

data$class <- gsub(1, 0, data$class)
data$class <- gsub(2, 1, data$class)
data$class <- gsub(3, 2, data$class)
data$class <- gsub(4, 3, data$class)
data$class <- gsub(5, 4, data$class)
data$class <- gsub(6, 5, data$class)
data$class <- gsub(7, 6, data$class)

data$class <- as.numeric(data$class)
Cover_type <- data
cover_type <- evaluate_models_mc(data, dataset_name = "Cover type")

##### Biodeq Dataset ##### 
data  <- read.arff("data/phpGUrE90.arff")

table(data$Class)
data$Class <- gsub(1, 0, data$Class)
data$Class <- gsub(2, 1, data$Class)
data$Class <- as.numeric(data$Class)
Qsar_biodeq <- data
qsar_biodeq <- evaluate_models(data, dataset_name = "qsar_biodeq")

##### Gesture (multi class) Dataset  ##### 
data <- read.arff("data/phpYLeydd.arff")

table(data$Phase)

data$Phase <- gsub("D", 0, data$Phase)
data$Phase <- gsub("H", 1, data$Phase)
data$Phase <- gsub("P", 2, data$Phase)
data$Phase <- gsub("R", 3, data$Phase)
data$Phase <- gsub("S", 4, data$Phase)
data$Phase <- as.numeric(data$Phase)
Gesture <- data
gesture <- evaluate_models_mc(data, dataset_name = "Gesture")

##### Dresses Dataset  ##### 
data <- read.arff("data/phpcFPMhq.arff")

table(data$Class)

data$Class <- gsub(1, 0, data$Class)
data$Class <- gsub(2, 1, data$Class)

data$Class <- as.numeric(data$Class)
Dresses <- data
dresses <- evaluate_models_alternative(data, dataset_name = "Dresses")

##### Higgs Dataset   ##### 
data <- read.arff("data/phpZLgL9q.arff")
data <- data %>% select(-1, everything(), 1)

table(data$class)

data$class <- gsub(0, 0, data$class)
data$class <- gsub(1, 1, data$class)

data$class <- as.numeric(data$class)
Higgs <- data
higgs <- evaluate_models_alternative(data, dataset_name = "Higgs")

##### Mice (multi class) Dataset ##### 
data <- read.arff("data/phpchCuL5.arff")

table(data$class)

data$class <- gsub("c-CS-m", 0, data$class)
data$class <- gsub("c-CS-s", 1, data$class)
data$class <- gsub("c-SC-m", 2, data$class)
data$class <- gsub("c-SC-s", 3, data$class)
data$class <- gsub("t-CS-m", 4, data$class)
data$class <- gsub("t-CS-s", 5, data$class)
data$class <- gsub("t-SC-m", 6, data$class)
data$class <- gsub("t-SC-s", 7, data$class)

data$class <- as.numeric(data$class)
Mice <- data
mice <- evaluate_models_mc_mice(data, dataset_name = "Mice")
##### eye_movement  Dataset  ##### 
data <- read.arff("data/eye_movement.arff")

table(data$label)

data$label <- gsub(1, 1, data$label)
data$label <- gsub(0, 0, data$label)

data$label <- as.numeric(data$label)
Eye_movement <- data
eye_movement <- evaluate_models(data, dataset_name = "Eye Movement")

###### Overall output ######
# Create output data with  dataset_name, log_loss_xgb, zero_one_loss_xgb,
# log_loss_catboost, zero_one_loss_catboost, number of classes, number of features,
# number of numeric features, number of categorical features, number of rows

results <- list(
  adult, 
  click,
  appetency, 
  churn,
  internet,
  upselling,
  kick,
  anneal,
  digits,
  bank_marketing,
  blood_transfusion,
  gas_concentration,
  car,
  cover_type,
  qsar_biodeq,
  gesture,
  dresses,
  higgs,
  mice,
  eye_movement

)

results_df <- bind_rows(lapply(results, as_tibble))

dataset_info <- data.frame(
  name = c("Adult", "Click",
              "Appetency",
              "Churn", "Internet", "Upselling", "Kick", "Anneal", 
                   "Digits", "Bank Marketing", "Blood transfusion", "Gas concentration", "car", 
                   "Cover type", "qsar_biodeq", "Gesture", "Dresses", "Higgs", "Mice", "Eye Movement"),
  Number_of_Features = c(14, 11,
                         230,
                         20, 1558, 49, 32, 38, 64, 16, 4, 129, 6, 54, 41, 32, 12, 28, 81, 23),
  Number_of_Rows = c(48842, 1000000,
                     50000, 
                     5000, 3279, 5128, 72983, 898, 5620, 45211, 748, 13910, 1728, 581012, 
                     1055, 9873, 500, 98050, 1080, 7608),
  Numerical_Features = c(6, 11,
                         192, 
                         16, 3, 34, 14, 6, 64, 7, 4, 129, 0, 10, 41, 32, 1, 20, 77, 20),
  Categorical_Features = c(8, 0, 
                           38,
                           4, 1555, 16, 18, 32, 0, 9, 0, 0, 6, 44, 0, 0, 11, 0, 4, 3), 
  task = c("Binary", "Binary",
           "Binary",
           "Binary", "Binary", "Binary", "Binary", "MultiClass", "MultiClass", "Binary", "Binary", "MultiClass", "MultiClass", "MultiClass",
    "Binary", "MultiClass", "Binary", "Binary", "MultiClass", "MultiClass"),
  Number_Missing_Values = c(6465, 0, 8024152, 0, 0, 0, 149271, 22175, 0, 0, 0, 0, 0, 0, 0, 0, 835, 0, 1396, 0)
)



results <- left_join(results_df, dataset_info, by = "name" )

# Check which log Loss is best and decide for the model
results <- results %>%
  mutate(superior = case_when(
    log_loss_xgb > log_loss_catboost & log_loss_lgb > log_loss_catboost  ~ "CatBoost",
    log_loss_xgb < log_loss_catboost & log_loss_xgb < log_loss_lgb  ~ "XGBoost",
    log_loss_lgb < log_loss_catboost & log_loss_lgb < log_loss_xgb  ~ "LightGBM",
    
    TRUE ~ "notClear"
  ))

# Add meta features about the dataset
# sample feature ratio
results$sample_feature_ratio <- results$Number_of_Rows / results$Number_of_Features

# Sparsity - percentage of missing
results$missing_values_ratio <- results$Number_Missing_Values / (results$Number_of_Features*results$Number_of_Rows)

# Imbalanced dataset, create variable majority_class_percentage

# Create a list with datasets
dataset_names <- list(Adult, Click, Appetency, Churn, Internet, 
                   Upselling, Kick, Anneal, Digits, Bank_marketing, 
                   Blood_transfusion, Gas_concentration, Car, 
                   Cover_type, Qsar_biodeq, Gesture, Dresses, 
                   Higgs, Mice, Eye_movement)




majority <- function(dataset_names, results) {
  majority_vector <- c()
  # Iterate over the list of dataframes
  for (i in seq_len(20)) {
    data <- dataset_names[[i]]
    data <- as.data.frame(data)
    # Get the last column of the dataframe
    last_col <- data[[ncol(data)]]
    
    # Calculate the majority class percentage
    majority_class_percentage <- max(prop.table(table(last_col))) * 100
    majority_vector <- c(majority_vector, majority_class_percentage)
    
    # Add the new column with the majority class percentage

  }
  return(majority_vector)
}

results$majority_class_percentage <- majority(dataset_names, results)  


