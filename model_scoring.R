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

# EDA --------------------------------------------------------

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

# Group by user and hotel, and count the number of visits
visit_counts <- activity_df %>%
  group_by(user, hotel) %>%
  summarise(visits = n(), .groups = 'drop')

# Group users by gender and count 
user_gender_count <- users_df %>% 
  group_by(gender) %>%  
  summarise(count = n())

# Group users by continent and count 
user_continent_count <- users_df %>% 
  group_by(home.continent) %>%  
  summarise(count = n())

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

# Test Models 1: User based Collaborative Filtering Recommendation (UCBF), ICBF, Popular and Random  -------------------------------------

  # Create a user-item matrix
  user_item_df <- activity_df %>%
    mutate(value = 1) %>%
    group_by(user, hotel) %>%
    summarise(value = sum(value), .groups = 'drop') %>%
    pivot_wider(names_from = hotel, values_from = value, values_fill = list(value = 0))  
    
  # Handle edge case, if user user_id 4232, replace row value which is equal to 1
    
  # Convert to matrix format suitable for recommenderlab
  user_item_matrix <- as.matrix(user_item_df[, -1])
  user_item_matrix_real <- as(user_item_matrix, "binaryRatingMatrix") # Check this 
  
  # Define evaluation scheme with k-fold cross-validation
  evaluation_scheme <- evaluationScheme(user_item_matrix_real, method = "split", train = 0.9, given = -1)
  
  # Specify algorithms to score
  algorithms <- list(
    "ALS"   = list(name = "ALS_implicit", param = list(n_factors=10))
    "user-based CF" = list(name = "UBCF", param = list(method = "Cosine",nn = 30)),
    "item-based CF" = list(name = "IBCF", param = list(method = "Cosine",k = 30)),
    "random items" = list(name = "RANDOM", param = NULL),
    "popular items" = list(name = "POPULAR", param = NULL)
  )
  
  # Similarity Measures 
  similarity_algorithms <- list(
    "item-based CF - Jaccard" = list(name = "IBCF", param = list(method = "Jaccard",k = 30)),
    "item-based CF - Cosine" = list(name = "IBCF", param = list(method = "cosine",k = 30)),
    "item-based CF - Pearson"  = list(name = "IBCF", param = list(method = "pearson",k = 30)),
    "popular items" = list(name = "POPULAR", param = NULL)
  )
  
  # Train and score algorithms, grabbing top 1 recommendation 
  results <- evaluate(evaluation_scheme, algorithms, type = "topNList",  n=c(1, 2,3))
  simiarlity_results <-evaluate(evaluation_scheme, similarity_algorithms, type = "topNList",  n=c(1, 2,3))
  
  # Plot results for precision and recall
  plot(results, "prec/rec", annotate = 3, legend = "topleft")
  plot( simiarlity_results , "prec/rec", annotate = 3, legend = "topleft")
    
#  4. Train Hybrid Recommender ------------------------------------------------
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
    
# 5. Test: Train gender clustered model - Males ---------------------------------------------
    user_item_aug_gender_df <- user_item_df %>% 
      left_join(users_df, by = c("user")) %>%  
      select(-c("home.continent")) %>%  
      mutate(gender = case_when(gender =="male"~1, TRUE ~ 0)) %>% 
      filter(gender == 1) %>%  
      select(-c("gender"))
    
    # Convert to matrix format suitable for recommenderlab
    user_item_matrix_gender_aug <- as.matrix(user_item_aug_gender_df[, -1])
    user_item_matrix_gender_aug_real <- as(user_item_matrix_gender_aug, "binaryRatingMatrix")
    
    # Define evaluation scheme with k-fold cross-validation
    ubcf_gender_evaluation_scheme <- evaluationScheme(user_item_matrix_gender_aug_real, method = "cross-validation", k = 2, given = -1)
    
    # Train Gender Augmented UBCF Model 
    ubcf_gender_model <- Recommender(getData(ubcf_gender_evaluation_scheme, "train"), method = "UBCF", parameter = list(method = "pearson"))
    
    # Predict the top N recommendations for the test set
    ubcf_gender_predictions <- predict(ubcf_gender_model, getData(ubcf_gender_evaluation_scheme, "known"), type = "topNList", n = 1)
    
    # Get the top 1 recommendation for each user 
    ucbf_gender_top_1_pred <- bestN(ubcf_gender_predictions, n = 1)
    
    # Get Given items 
    ubcf_gender_given_items <- getData(ubcf_gender_evaluation_scheme, "given")
    
    # Calculate the prediction accuracy
    ubcf_gender_error <- calcPredictionAccuracy(ubcf_gender_predictions, getData(ubcf_gender_evaluation_scheme, "unknown"), given = ubcf_gender_given_items)
    
    #Print 
    print(ubcf_gender_error)

    # 5. Test: Continent clustered model - Popular Continent 7 ---------------------------------------------
    user_item_continent_df <- user_item_df %>% 
      left_join(users_df, by = c("user")) %>%  
      select(-c("gender")) %>%  
      filter(home.continent ==2) %>%  
      select(-c("home.continent"))
    
    # Convert to matrix format suitable for recommenderlab
    user_item_matrix_continent <- as.matrix(user_item_continent_df[, -1])
    user_item_matrix_continent_real <- as(user_item_matrix_continent, "binaryRatingMatrix")
    
    # Define evaluation scheme with k-fold cross-validation
    ubcf_continent_evaluation_scheme <- evaluationScheme(user_item_matrix_continent_real, method = "cross-validation", k = 3, given = -1)
    
    # Train Gender& Continent Augmented UBCF Model 
    ubcf_continent_model <- Recommender(getData(ubcf_continent_evaluation_scheme, "train"), method = "UBCF", parameter = list(method = "pearson"))
    
    # Predict the top N recommendations for the test set
    ubcf_continent_predictions <- predict(ubcf_continent_model, getData(ubcf_continent_evaluation_scheme, "known"), type = "topNList", n = 1)
    
    # Get the top 1 recommendation for each user 
    ucbf_continent_top_1_pred <- bestN(ubcf_continent_predictions, n = 1)
    
    # Get Given items 
    ubcf_continent_given_items <- getData(ubcf_continent_evaluation_scheme, "given")
    
    # Calculate the prediction accuracy
    ubcf_continent_error <- calcPredictionAccuracy( ubcf_continent_predictions, getData(ubcf_continent_evaluation_scheme, "unknown"), given = ubcf_continent_given_items)
    
    #Print 
    print(ubcf_continent_error)
    
    
    
    
    
    
    
    
    
    
    
    