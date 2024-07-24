library(dplyr)
library(tidyr)
library(ggplot2)
library(recommenderlab)

# Clear work space 
rm(list=ls())

# Set seed for reproducibility
set.seed(123)

# Read in data and join ---------------------------------------------------

# Read in data sets 
users_df <- read.delim("~/xero_assessment/data/users.txt")
hotels_df <- read.delim("~/xero_assessment/data/hotels.txt")
activity_df <- read.delim("~/xero_assessment/data/activity.txt")

# Merge datasets to create a unified panel
merged_df <- activity_df %>%
  left_join(users_df, by = "user") %>%
  left_join(hotels_df, by = "hotel")

# Feature Engineer --------------------------------------------------------

# Feature Engineering population metric
popularity_df <- merged_df %>%
  group_by(hotel) %>%
  summarise(popularity = n()) %>%
  ungroup() %>% 
  left_join(hotels_df)

# Calculate the number interactions per user 
interaction_counts <- merged_df %>%  
  group_by(user) %>%  
  summarise(num_interactions = n()) %>%  
  ungroup()

# Count total number of interactions 
interaction_counts_summary <- interaction_counts %>%  
  group_by(num_interactions) %>% 
  summarise(total = n())

# Simple EDA: Check the count of interactions  --------------------------------------------------------------

# # Plot a smoothed histogram of average 
# ggplot(interaction_counts, aes(x = num_interactions)) +
#   geom_histogram(aes(y = ..density..), bins = 30, fill = "blue", alpha = 0.6) +
#   geom_density(color = "red", size = 1) +
#   labs(title = "Density Histogram of Interactions per User",
#        x = "Number of Interactions",
#        y = "Density") +
#   theme_minimal()

# Test Model 1: Collaborative Filtering Recommendation  -------------------------------------

# Filter out users with fewer than three interactions
filtered_activity_df <- activity_df %>%
  group_by(user) %>%
  filter(n() >= 3) %>%
  ungroup()

# Create a user-item matrix
user_item_matrix <- filtered_activity_df %>%
  mutate(value = 1) %>%
  group_by(user, hotel) %>%
  summarise(value = sum(value), .groups = 'drop') %>%
  pivot_wider(names_from = hotel, values_from = value, values_fill = list(value = 0))

# Convert to matrix format suitable for recommenderlab
user_item_matrix_matrix <- as.matrix(user_item_matrix[, -1])
user_item_matrix_real <- as(user_item_matrix_matrix, "realRatingMatrix")

# Define evaluation scheme with k-fold cross-validation
evaluation_scheme <- evaluationScheme(user_item_matrix_real, method = "cross-validation", k = 5, given = 3, goodRating = 1)

# Define the recommender model using User-Based Collaborative Filtering (UBCF) (Default Cosine similarity)
ubcf_model <- Recommender(getData(evaluation_scheme, "train"), method = "UBCF")

# Predict ratings for the test set
predictions <- predict(ubcf_model, getData(evaluation_scheme, "known"), type = "ratings")

# Get the prediction matrix
prediction_matrix <- as(predictions, "matrix")

# Function to exclude already visited hotels
exclude_visited <- function(user_id, prediction_matrix, original_data) {
  visited_hotels <- original_data %>%
    filter(user == user_id) %>%
    pull(hotel)
  prediction_matrix[user_id, colnames(prediction_matrix) %in% visited_hotels] <- NA
  return(prediction_matrix)
}

# Apply the exclusion function to each user's predictions
user_ids <- rownames(prediction_matrix)
for (user_id in user_ids) {
  prediction_matrix <- exclude_visited(user_id, prediction_matrix, activity_df)
}

# Convert the prediction matrix back to a realRatingMatrix format
filtered_predictions <- as(prediction_matrix, "realRatingMatrix")

# Calculate the prediction accuracy
error <- calcPredictionAccuracy(filtered_predictions, getData(evaluation_scheme, "unknown"))

# Display the Root Mean Squared Error (RMSE)
rmse <- error["RMSE"]
print(paste("RMSE:", rmse))

# Get Recommendations from UBCF Model -------------------------------------

# Function to get the top recommended hotel for each user
get_top_recommendation <- function(user_id, prediction_matrix) {
  user_predictions <- prediction_matrix[user_id, , drop = FALSE]
  top_hotel <- colnames(user_predictions)[which.max(user_predictions)]
  return(top_hotel)
}

# Create a dataframe with users and their top recommended hotel
recommendations <- data.frame(
  user = user_ids,
  top_recommended_hotel = sapply(user_ids, get_top_recommendation, prediction_matrix = prediction_matrix)
)

