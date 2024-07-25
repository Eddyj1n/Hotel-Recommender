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

# Feature Engineer --------------------------------------------------------

# Merge datasets to create a unified panel
merged_df <- activity_df %>%
  left_join(users_df, by = "user") %>%
  left_join(hotels_df, by = "hotel")

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

# Group by user and hotel, and count the number of visits
visit_counts <- activity_df %>%
  group_by(user, hotel) %>%
  summarise(visits = n(), .groups = 'drop')

# Check average revisits
average_revisits_per_user <- visit_counts %>%
  group_by(user) %>%
  summarise(average_revisits = mean(visits), .groups = 'drop')

# # Plot a smoothed histogram of average 
# ggplot(interaction_counts, aes(x = num_interactions)) +
#   geom_histogram(aes(y = ..density..), bins = 30, fill = "blue", alpha = 0.6) +
#   geom_density(color = "red", size = 1) +
#   labs(title = "Density Histogram of Interactions per User",
#        x = "Number of Interactions",
#        y = "Density") +
#   theme_minimal()

# Test Model 1: User based Collaborative Filtering Recommendation (User Interactions Only)  -------------------------------------

  # Filter out users with fewer than three interactions
  filtered_activity_df <- activity_df %>%
    group_by(user) %>%
    #filter(n() >= 3) %>%
    ungroup()
  
  # Create a user-item matrix
  user_item_df <- filtered_activity_df %>%
    mutate(value = 1) %>%
    group_by(user, hotel) %>%
    summarise(value = sum(value), .groups = 'drop') %>%
    pivot_wider(names_from = hotel, values_from = value, values_fill = list(value = 0))  
    
  # Handle edge case, if user user_id 4232, replace row value which is equal to 1
    
  # Convert to matrix format suitable for recommenderlab
  user_item_matrix <- as.matrix(user_item_df[, -1])
  user_item_matrix_real <- as(user_item_matrix, "binaryRatingMatrix") # Check this 
  
  # Define evaluation scheme with k-fold cross-validation
  ubcf_evaluation_scheme <- evaluationScheme(user_item_matrix_real, method = "cross-validation", k = 3, given = 2)
  
  # Define the recommender model using User-Based Collaborative Filtering (UBCF) (Default Cosine similarity)
  ubcf_model <- Recommender(getData(ubcf_evaluation_scheme, "train"), method = "UBCF", parameter = list(method = "pearson"))
  
  # Predict the top N recommendations for the test set
  ubcf_predictions <- predict(ubcf_model, getData(ubcf_evaluation_scheme, "known"), type = "topNList", n = 1)

  # Remove previously interacted hotels from the recommendations
  ubcf_predictions_filtered <- removeKnownItems(ubcf_predictions, getData(ubcf_evaluation_scheme,"known"))
  
  # Get the top 1 recommendation for each user 
  ucbf_top_1_pred <- bestN(ubcf_predictions_filtered, n=1)
    
# 1.1 Evaluate UBCF Model -------------------------------------------------
  
  # Obtain the number of items given from the evaluation scheme
  ubcf_given_items <- getData(ubcf_evaluation_scheme, "given")

  # Calculate the prediction accuracy for both 
  ubcf_error <- calcPredictionAccuracy(ubcf_predictions, getData(ubcf_evaluation_scheme, "unknown"), given = ubcf_given_items)

  # Print error 
  print(ubcf_error )

# 2.0 test IBCF model -----------------------------------------------------
    # Define evaluation scheme with k-fold cross-validation
    ibcf_evaluation_scheme <- evaluationScheme(user_item_matrix_real, method = "cross-validation", k = 2, given = 2)
    
    # Define the recommender model using Item-Based Collaborative Filtering (IBCF) (Default Cosine similarity)
    ibcf_model <- Recommender(getData(ibcf_evaluation_scheme, "train"), method = "IBCF")
    
    # Use cosine similarity
    ibcf_model <- Recommender(getData(ibcf_evaluation_scheme, "train"), method = "IBCF", parameter = list(method = "Jaccard"))
    
    # Predict the top N recommendations for the test set
    ibcf_predictions <- predict(ibcf_model, getData(ibcf_evaluation_scheme, "known"), type = "topNList", n = 1)
    
    # Remove previously interacted hotels from the recommendations
    ibcf_predictions_filtered <- removeKnownItems(ibcf_predictions, getData(ibcf_evaluation_scheme,"known"))
    
    # Get the top 1 recommendation for each user 
    ibcf_top_1_pred <- bestN(ibcf_predictions_filtered, n=1)

# 2.1 Evalute IBCF Model ------------------------------------------------------
   
    # Get Given items 
    ibcf_given_items <- getData(ibcf_evaluation_scheme, "given")
  
    # Calculate the prediction accuracy
    ibcf_error <- calcPredictionAccuracy(ibcf_predictions, getData(ubcf_evaluation_scheme, "unknown"), given = ibcf_given_items)
    
    # Print Error
    print(ibcf_error)
    
# 3. Augmented UBCF model (Gender) ----------------------------------------
    
    # One-hot encode gender and augment matrix 
    # Augment user item matrix by gender 
    user_item_aug_gender_df <- user_item_df %>% 
      left_join(users_df, by = c("user")) %>%  
      select(-c("home.continent")) %>%  
      mutate(gender = case_when(gender =="male"~1, TRUE ~ 0))
    
    # Convert to matrix format suitable for recommenderlab
    user_item_matrix_gender_aug <- as.matrix(user_item_aug_gender_df[, -1])
    user_item_matrix_gender_aug_real <- as(user_item_matrix_gender_aug, "binaryRatingMatrix")
    
    # Define evaluation scheme with k-fold cross-validation
    ubcf_gender_evaluation_scheme <- evaluationScheme(user_item_matrix_gender_aug_real, method = "cross-validation", k = 2, given = 3)
    
    # Train Gender Augmented UBCF Model 
    ubcf_gender_model <- Recommender(getData(ubcf_gender_evaluation_scheme, "train"), method = "UBCF", parameter = list(method = "pearson"))
    
    # Predict the top N recommendations for the test set
    ubcf_gender_predictions <- predict(ubcf_gender_model, getData(ubcf_gender_evaluation_scheme, "known"), type = "topNList", n = 1)
    
    # Get the top 1 recommendation for each user 
    ucbf_gender_top_1_pred <- bestN(ubcf_gender_predictions, n = 1)
    
    # Get Given items 
    ubcf_gender_given_items <- getData(ubcf_gender_evaluation_scheme, "given")
    
    # Calculate the prediction accuracy
    #ubcf_gender_error <- calcPredictionAccuracy(ubcf_gender_predictions, getData(ubcf_gender_evaluation_scheme, "unknown"), given = ubcf_gender_given_items)
    ubcf_gender_error <- calcPredictionAccuracy(ubcf_gender_predictions, getData(ubcf_gender_evaluation_scheme, "unknown"), given = 3)
    
    #Print 
    print(ubcf_gender_error)

# 4. Augmented UBCF model with Gender and Continents ----------------------------------------
    
    # One-hot encode continent  
    continent_one_hot_encode <- users_df%>%
      select(user, home.continent) %>% 
      mutate(home.continent = home.continent+1000) %>% # To differentiate colnames between hotels
      mutate(value = 1) %>%
      group_by(user, home.continent) %>%
      summarise(value = sum(value), .groups = 'drop') %>%
      pivot_wider(names_from = home.continent, values_from = value, values_fill = list(value = 0))  
    
    # Augment gender user-item matrix by home continent information
    user_item_continent_df <- user_item_aug_gender_df %>% 
      left_join(continent_one_hot_encode , by = c("user")) 
    
    # Convert to matrix format suitable for recommenderlab
    user_item_matrix_continent <- as.matrix(user_item_continent_df[, -1])
    user_item_matrix_continent_real <- as(user_item_matrix_continent, "binaryRatingMatrix")
    
    # Define evaluation scheme with k-fold cross-validation
    ubcf_continent_evaluation_scheme <- evaluationScheme(user_item_matrix_continent_real, method = "cross-validation", k = 3, given = 1)
    
    # Train Gender& Continent Augmented UBCF Model 
    ubcf_continent_model <- Recommender(getData(ubcf_continent_evaluation_scheme, "train"), method = "UBCF", parameter = list(method = "pearson"))
    
    # Predict the top N recommendations for the test set
    ubcf_continent_predictions <- predict(ubcf_continent_model, getData(ubcf_continent_evaluation_scheme, "known"), type = "topNList", n = 1)
    
    # Get the top 1 recommendation for each user 
    ucbf_continent_top_1_pred <- bestN(ubcf_continent_predictions, n = 1)
    
    # Get Given items 
    ubcf_continent_given_items <- getData(ubcf_continent_evaluation_scheme, "given")
    
    # Calculate the prediction accuracy
    #ubcf_continent_error <- calcPredictionAccuracy( ubcf_continent_predictions, getData(ubcf_continent_evaluation_scheme, "unknown"), given = ubcf_continent_given_items)
    ubcf_continent_error <- calcPredictionAccuracy( ubcf_continent_predictions, getData(ubcf_continent_evaluation_scheme, "unknown"), given = 1)

    #Print 
    print(ubcf_continent_error)
    

# 5. Test Popular Method --------------------------------------------------
    
    # Define evaluation scheme with 80% train and 20% test split
    #pop_evaluation_scheme <- evaluationScheme(user_item_matrix_real, method = "split", train = 0.9, given = -1)
    
    # Define evaluation scheme with k-fold cross-validation
    pop_evaluation_scheme <- evaluationScheme(user_item_matrix_real, method = "cross-validation", k = 3, given = 1)
    
    # Define the recommender model using Popular method
    popular_model <- Recommender(getData(pop_evaluation_scheme, "train"), method = "POPULAR")
    
    # Predict the top N recommendations for the test set
    popular_predictions <- predict(popular_model, getData(pop_evaluation_scheme, "known"), type = "topNList", n = 1)
    
    # Get the top 1 recommendation for each user
    popular_top_1_pred <- bestN(popular_predictions, n = 1)
    
    # Extract the top 1 recommendation for each user
    top_1_recommendations <- as(popular_top_1_pred, "list")
    
    # Get Given items 
    popular_given_items <- getData(pop_evaluation_scheme, "given")
    
    # Calculate the prediction accuracy
    popular_error <- calcPredictionAccuracy(popular_predictions, getData(pop_evaluation_scheme, "unknown"), given =  popular_given_items )
    popular_error <- calcPredictionAccuracy(popular_predictions, getData(pop_evaluation_scheme, "unknown"), given =  1)
    
    #Print 
    print(popular_error)
    
#  Train Hybrid Recommender ---------------------------------------------
    hybrid_model <- HybridRecommender(
      Recommender(getData(ubcf_evaluation_scheme, "train"), method = "UBCF", parameter = list(method = "cosine")),
      Recommender(getData(pop_evaluation_scheme, "train"), method = "POPULAR"),
      weights = c(0.4, 0.6)
    )

    # Predict the top N recommendations for the test set using hybrid model
    hybrid_predictions <- predict(hybrid_model, getData( ubcf_evaluation_scheme, "known"), type = "topNList", n = 1)
    
    # Get the top 1 recommendation for each user using hybrid model
    hybrid_top_1_pred <- bestN(hybrid_predictions, n = 1)
    
    # Calculate the prediction accuracy for hybrid model
    hybrid_error <- calcPredictionAccuracy(hybrid_predictions, getData( ubcf_evaluation_scheme, "unknown"), given = popular_given_items)
    
    # Print Hybrid Error 
    print(hybrid_error)
    