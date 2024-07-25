
# Load packages -----------------------------------------------------------
library(dplyr)
library(tidyr)
library(recommenderlab)

# Clear work space 
rm(list=ls())

# Read and Preprocess Data ------------------------------------------------

# Read in required data sets [Change code to automatically read in]
users_df <- read.delim("~/xero_assessment/data/users.txt")
activity_df <- read.delim("~/xero_assessment/data/activity.txt")

# Filter for two user id sets : A: Users with >= 2 interactions, B: remaining users
# Group A will be assigned Augmented User Collaborative Filtering Model Recommendation 
a_users <- activity_df %>%  
  group_by(user) %>%  
  filter(n()<=2) %>%  
  distinct(user)

# Group B will be assigned Popular Model Recommendation
b_users <- activity_df %>%  
  group_by(user) %>% 
  filter(n() >2) %>%  
  distinct(user)

# Engineer Features to Parse into Models -----------------------------------

# Create augmented user item data frame for UBCF model 
 user_item_df_aug <- activity_df  %>%  
  
  # 1. One hot encode hotel 
  mutate(value = 1) %>%
  group_by(user, hotel) %>%
  summarise(value = sum(value), .groups = 'drop') %>%
  pivot_wider(names_from = hotel, values_from = value, values_fill = list(value = 0))  %>% 
  
  # 2. One hot encode Gender
  left_join(users_df, by = c("user")) %>%  
  select(-c("home.continent")) %>%  
  mutate(gender = case_when(gender =="male"~1, TRUE ~ 0)) %>%   # Numerically encode gender 

  # 3. One hot encode home.continent
  left_join(users_df %>% select(-c("gender")) , by = c("user")) %>%  
  mutate(home.continent = home.continent+1000,
          value = 1) %>%  
  pivot_wider(names_from = home.continent, values_from = value, values_fill = list(value = 0))

# Create simple user item data frame for Popular Model 
user_item_df <- activity_df %>%
  mutate(value = 1) %>%
  group_by(user, hotel) %>%
  summarise(value = sum(value), .groups = 'drop') %>%
  pivot_wider(names_from = hotel, values_from = value, values_fill = list(value = 0))

# Create user item matrices
user_item_matrix_aug <- as.matrix(user_item_df_aug [, -1])
user_item_matrix_aug_real  <- as(user_item_matrix_aug, "binaryRatingMatrix")

user_item_matrix <- as.matrix(user_item_df[, -1])
user_item_matrix_real  <- as(user_item_matrix, "binaryRatingMatrix")

# Train Models and generate predictions ------------------------------------------------------------
# Train UBCF model 
ubcf_model <- Recommender(user_item_matrix_aug_real, method = "UBCF", parameter = list(method = "pearson"))

# Train Popular model 
popular_model <- Recommender(user_item_matrix_real, method = "POPULAR")

# Predict the top N recommendations for UBCF model 
ubcf_predictions <- predict(ubcf_model,user_item_matrix_aug_real, type = "topNList", n = 1)

# Predict the top N recommendations for POPULAR model
popular_predictions <- predict(popular_model,user_item_matrix_real, type = "topNList", n = 1)

# Retrieve Top Recommendations --------------------------------------------

# UBCF Model 
# Get the top 1 recommendation for each user
ubcf_top_1_pred <- bestN(ubcf_predictions, n = 1)

# Extract the top 1 recommendation for each user
ubcf_top_1_list <- getList(ubcf_top_1_pred)

# Convert the list to a dataframe
# Filter out users with no recommendations
ubcf_top_1_df <- do.call(rbind, lapply(names(ubcf_top_1_list), function(user) {
  if(length(ubcf_top_1_list[[user]]) > 0) {
    data.frame(User = user, Hotel = ubcf_top_1_list[[user]])
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



  
