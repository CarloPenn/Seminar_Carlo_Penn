# In this file we analyze which meta features contribute to which model performs better

##### packages #####
# install.packages("nnet")  # If not installed
if (!require("nnet", quietly = TRUE)) install.packages("nnet")



###### Multinomial model #####
table(results$superior)
results$superior <- as.factor(results$superior)
results$superior <- relevel(factor(results$superior), ref = "CatBoost")
results$categorical_ratio <- results$Categorical_Features / results$Number_of_Features
# Fit multinomial logistic regression model
logit_model <- multinom(superior ~ Number_of_Features + Number_of_Rows + 
                           + categorical_ratio + task + Number_Missing_Values + sample_feature_ratio + missing_values_ratio + majority_class_percentage, 
                        data = results)
model_summary <- summary(logit_model)
# Manual computation of p-values
coefficients <- model_summary$coefficients
se <- model_summary$standard.errors

# Calculate z-values (coefficient / standard error)
z_values <- coefficients / se

# Calculate p-values (2 * (1 - pnorm(abs(z-values)))) to get two-tailed p-values
p_values <- 2 * (1 - pnorm(abs(z_values)))

# Display p-values
p_values


###### Simple graphical analysis #####
# Number of features
plot_number_of_features <- ggplot(results, aes(x = log10(Number_of_Features), y = superior)) +
  geom_point() + ggtitle("Number of Features vs. Superior Model") +
  labs(x = "Number of features(log10-scaled)", y = "Model") +
  theme(plot.title = element_text(hjust = 0.5))  # Center the title
ggsave("Bilder/number_of_features.png", plot = plot_number_of_features, width = 8, height = 6)


# Number of rows
plot_number_of_rows <- ggplot(results, aes(x = log10(Number_of_Rows), y = superior)) +
  geom_point() + ggtitle("Number of Rows  vs. Superior Model") +
  labs(x = "Number of rows(log10-scaled)", y = "Model") +
  theme(plot.title = element_text(hjust = 0.5))  # Center the title
ggsave("Bilder/number_of_rows.png", plot = plot_number_of_rows, width = 8, height = 6)


# Number of numerical features
plot_number_of_numerical_features <- ggplot(results, aes(x = log10(Numerical_Features), y = superior)) +
  geom_point() + ggtitle("Number of numerical features vs. Superior Model") +
  labs(x = "Number of numerical features(log10-scaled)", y = "Model") +
  theme(plot.title = element_text(hjust = 0.5))  # Center the title
ggsave("Bilder/number_of_numerical_features.png", plot = plot_number_of_numerical_features, width = 8, height = 6)


# Number of categorical features
plot_number_categorical_features <- ggplot(results, aes(x = log10(Categorical_Features), y = superior)) +
  geom_point() + ggtitle("Number of categorical features vs. Superior Model") +
  labs(x = "Number of categorical features(log10-scaled)", y = "Model") +
  theme(plot.title = element_text(hjust = 0.5))  # Center the title
ggsave("Bilder/number_categorical_features.png", plot = plot_number_categorical_features, width = 8, height = 6)


# Binary or multiclass task
plot_task <- ggplot(data = results) +
  geom_mosaic(aes(x = product(task), fill=superior)) +
  theme_mosaic() +
  ggtitle("Binary/Multiclass vs. Superior Model") +
  labs(x = "Type of classification", y = "Model") +
  scale_fill_manual(values = c("grey", "lightblue", "grey20")) +  # Custom colors 
theme(plot.title = element_text(hjust = 0.5))  # Center the title
  
ggsave("Bilder/task.png", plot = plot_task, width = 8, height = 6)


# Number of missing values
plot_number_of_missing_values <- ggplot(results, aes(x = log10(Number_Missing_Values + 1), y = superior)) +
  geom_point() + ggtitle("Number of missing values vs. Superior Model") +
  xlim(0, 8) +
  labs(x = "Number of missing values(log10-scaled)", y = "Model") +
  theme(plot.title = element_text(hjust = 0.5))  # Center the title
ggsave("Bilder/number_of_missing_values.png", plot = plot_number_of_missing_values, width = 8, height = 6)


# Sample feature ratio
plot_sample_feature_ratio <- ggplot(results, aes(x = log10(sample_feature_ratio), y = superior)) +
  geom_point() + ggtitle("Sample-feature ratio vs. Superior Model") +
  labs(x = "Sample-feature ratio(log10-scaled)", y = "Model") +
  theme(plot.title = element_text(hjust = 0.5))  # Center the title
ggsave("Bilder/sample_feature_ratio.png", plot = plot_sample_feature_ratio, width = 8, height = 6)


# Missing value ratio compared to whole dataset
plot_missing_values_ratio <- ggplot(results, aes(x = missing_values_ratio, y = superior)) +
  geom_point() + ggtitle("Missing values ratio vs. Superior Model") +
  labs(x = "Missing values ratio", y = "Model") +
  theme(plot.title = element_text(hjust = 0.5))  # Center the title
ggsave("Bilder/missing_values_ratio.png", plot = plot_missing_values_ratio, width = 8, height = 6)


# Missing value ratio compared to whole dataset
plot_majority_class_percentage <- ggplot(results, aes(x = majority_class_percentage, y = superior)) +
  geom_point() + ggtitle("Majority class percentage vs. Superior Model") +
  labs(x = "Majority class percentage", y = "Model") +
  theme(plot.title = element_text(hjust = 0.5))  # Center the title
ggsave("Bilder/majority_class_percentage.png", plot = plot_majority_class_percentage, width = 8, height = 6)

plot_rows_features_superior <- ggplot(results, aes(x = log10(Number_of_Rows), y = log10(Number_of_Features), color = superior)) +
  geom_point() + ggtitle("test") +
  labs(x = "Number of rows(log10-scaled)", y = "Number of features(log10-scaled)") +
  theme(plot.title = element_text(hjust = 0.5)) +  # Center the title
  scale_color_manual(values = c("grey", "lightblue", "grey20"))   # Custom colors 
  
ggsave("Bilder/plot_rows_features_superior.png", plot = plot_rows_features_superior, width = 8, height = 6)


###### decision tree #####
tree_model <- rpart(superior ~ ., data = results, method = "class")
plot(tree_model)
text(tree_model, pretty = 1)

##### Random forest with cv 4/5 #####
training_index <- createDataPartition(1:20, p = 0.75, list = FALSE)

train_data <- results[training_index, ]
test_data <- results[-training_index, ]
model <- randomForest(formula = superior ~ ., data = train_data, ntree = 1000, mtry = 5)
model$confusion
prediction <- predict(model, newdata = test_data)

table(prediction, test_data$superior)

mean(prediction != test_data$superior)

##### Difference in performance from CatBoost #####
results$Difference_Cat_XG_logloss <- ((results$log_loss_xgb - results$log_loss_catboost) / results$log_loss_catboost)*100
results$Difference_Cat_Light_logloss <- ((results$log_loss_lgb - results$log_loss_catboost) / results$log_loss_catboost)*100

##### Excluding datasets with less thank 5k rows

###### Multinomial model #####
results_large_data <- results
results_large_data$superior <- as.factor(results$superior)
results_large_data$superior <- relevel(factor(results_large_data$superior), ref = "CatBoost")
# Fit multinomial logistic regression model
logit_model_large_data <- multinom(superior ~ Number_of_Features + Number_of_Rows + 
                          Numerical_Features + Categorical_Features + task + Number_Missing_Values + sample_feature_ratio + missing_values_ratio + majority_class_percentage, 
                        data = results_large_data)
model_summary_large_data <- summary(logit_model_large_data)
table(results_large_data$superior)
