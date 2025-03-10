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
"adult <- read.arff("data/phpMawTba.arff")
appetency <- read.arff("data/KDDCup09_appetency.arff")
churn <- read.arff("data/churn.arff")
Internet <- read.arff("data/phpPIHVvG.arff")
upselling <- read.arff("data/upselling.arff")
kick <- read.arff("data/kick.arff")
anneal <- read.arff("data/dataset_1_anneal.arff")
digits <- read.arff("data/dataset_28_optdigits.arff")
bank_marketing <- read.arff("data/phpkIxskf.arff")
blood_transfusion <- read.arff("data/php0iVrYT.arff")
gas_concentration <- read.arff("data/phpN4gaxw.arff")
car <- read.arff("data/php2jDIhh.arff")
cover_type <- read.arff("data/phpQOf0wY.arff")
qsar_biodeg <- read.arff("data/phpGUrE90.arff")
gesture <- read.arff("data/phpYLeydd.arff")
dresses <- read.arff("data/phpcFPMhq.arff")
higgs <- read.arff("data/phpZLgL9q.arff")
mice <- read.arff("data/phpchCuL5.arff")
eye_movement <- read.arff("data/eye_movement.arff")


"


##### Practice task on all datasets #####





