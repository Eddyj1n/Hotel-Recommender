
# Load packages -----------------------------------------------------------
library(dplyr)
library(tidyr)
library(recommenderlab)

# Clear work space 
rm(list=ls())

# Read and Preprocess Data ------------------------------------------------

# Read in required data sets [Change code to automatically read in]
activity_df <- read.delim("~/xero_assessment/data/activity.txt")

# Engineer Features to Parse into Models -----------------------------------
  
# Create simple user item data frame to parse into IBCF model 
user_item_df <- activity_df %>%
  mutate(value = 1) %>%
  group_by(user, hotel) %>%
  summarise(value = sum(value), .groups = 'drop') %>%
  pivot_wider(names_from = hotel, values_from = value, values_fill = list(value = 0))

# Create user item matrix
user_item_matrix <- as.matrix(user_item_df[, -1])
user_item_matrix_real  <- as(user_item_matrix, "binaryRatingMatrix")

# Train Models and generate predictions ------------------------------------------------------------
# Train ICBF model 
ibcf_model <- Recommender(user_item_matrix_real, method = "IBCF", parameter = list(method = "Cosine", k=30))

# Train Popular model 
popular_model <- Recommender(user_item_matrix_real, method = "POPULAR")

# Predict the top N recommendations for UBCF model 
ibcf_predictions <- predict(ibcf_model,user_item_matrix_real, type = "topNList", n = 1)

# Predict the top N recommendations for POPULAR model
popular_predictions <- predict(popular_model,user_item_matrix_real, type = "topNList", n = 1)

# Retrieve Top Recommendations --------------------------------------------

# UBCF Model 
# Get the top 1 recommendation for each user
ibcf_top_1_pred <- bestN(ibcf_predictions, n = 1)

# Extract the top 1 recommendation for each user
ibcf_top_1_list <- getList(ibcf_top_1_pred)

# Convert the list to a dataframe
# Filter out users with no recommendations
ibcf_top_1_df <- do.call(rbind, lapply(names(ibcf_top_1_list), function(user) {
  if(length(ibcf_top_1_list[[user]]) > 0) {
    data.frame(User = user, Hotel = ibcf_top_1_list[[user]])
  }
}))

# POPULAR model 
# Get the top 1 recommendation for each user
popular_top_1_pred <- bestN(popular_predictions, n = 1)

# Extract the top 1 recommendation for each user
popular_top_1_list <- getList(popular_top_1_pred)

# Convert the list to a dataframe
popular_top_1_df <- do.call(rbind, lapply(names(popular_top_1_list), function(user) {
  data.frame(User = user, Hotel = popular_top_1_list[[user]])
}))



  
