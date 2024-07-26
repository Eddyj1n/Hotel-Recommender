# Install and load required packages --------------------------------------
required_packages <- c("dplyr", "tidyr", "recommenderlab", "here")

# Define Function to install missing packages
install_if_missing <- function(p) {
  if (!require(p, character.only = TRUE)) {
    install.packages(p, dependencies = TRUE)
    library(p, character.only = TRUE)
  }
}

# Install missing packages
invisible(sapply(required_packages, install_if_missing))

# Load packages -----------------------------------------------------------
library(dplyr)
library(tidyr)
library(recommenderlab)
library(here)

# Read and Preprocess Data ------------------------------------------------

# Read in required data sets [Change code to automatically read in]
activity_df <- read.delim(here("data", "activity.txt"))

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

# Predict the top N recommendations for UBCF model 
ibcf_predictions <- predict(ibcf_model,user_item_matrix_real, type = "topNList", n = 1)

# Retrieve Top Recommendations --------------------------------------------

# UBCF Model 
# Get the top 1 recommendation for each user
ibcf_top_1_pred <- bestN(ibcf_predictions, n = 1)

# Extract the top 1 recommendation for each user
ibcf_top_1_list <- getList(ibcf_top_1_pred)

# Convert the list to a data  frame
ibcf_top_1_df <- do.call(rbind, lapply(names(ibcf_top_1_list), function(user) {
  # Put control statement if edge cases exists where user has not recommendation 
  if(length(ibcf_top_1_list[[user]]) > 0) {
    data.frame(User = user, Hotel = ibcf_top_1_list[[user]])
  }
}))

# Output Predictions ------------------------------------------------------

# Ensure the predictions directory exists
if (!dir.exists(here("predictions"))) {
  dir.create(here("predictions"))
}

# Export top recommendations to a tab delimited file named "Top_hotel_recommendation.txt" 
output_path <- here("predictions", "Top_hotel_recommendation.txt")
write.table(ibcf_top_1_df, file = output_path, 
            sep = "\t", 
            row.names = FALSE, 
            col.names = TRUE, 
            quote = FALSE)

# Print completion message
cat("PREDICTION COMPLETE - Output located in", output_path, "\n")
  
