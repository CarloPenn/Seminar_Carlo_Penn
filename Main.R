###### Seminar: Automated Machine Learning ######
# Author: Carlo Penn, 12236597
# Start date: 2025.02.25
# Last edit: 
# This projecct deals with "catboost" and "xgboost".
# Specific topic: "


###### Read packages ######
install.packages(c("catboost", "xgboost", "remotes", "foreign", "MLmetrics", "lightgbm", "ggmosaic"))
remotes::install_url('https://github.com/catboost/catboost/releases/download/v1.2.7/catboost-R-windows-x86_64-1.2.7.tgz', INSTALL_opts = c("--no-multiarch", "--no-test-load"))

library(foreign)
library(Metrics)  # For RMSE
library(farff)      # For reading .arff files
library(mlr3)       # ML framework
library(mlr3learners) # Additional learners
library(xgboost)    # XGBoost library
library(catboost)   # CatBoost library
library(caret)      # For train/test split
library(dplyr)      # For data manipulation
library(MLmetrics)
library(lightgbm)
library(ggplot2)
library(ggmosaic)
library(nnet)
library(randomForest)
library(rpart)

source("Multiclass.R")
source("multiclass_mice.R")
source("model_evaluation_alternative.R")
source("appetency_prepare.R")
              
                    

###### read data ######
# -       Find datasets for benchmark


# List the smaller ARFF files in the 'data/' folder
file_list <- list.files("data/", pattern = "small_dataset_part_.*\\.arff", full.names = TRUE)

# Initialize an empty list to store the data
click <- NULL

# Loop through each file and load the data
for (file in file_list) {
  # Read the ARFF file
  data_chunk <- read.arff(file)
  
  # Append the data to the combined dataset
  click <- rbind(click, data_chunk)
}







