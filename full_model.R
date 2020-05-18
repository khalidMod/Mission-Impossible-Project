
library(tidyverse)
library(ggplot2)
library(corrplot)
library(caret)
library(xgboost)
library(parallel)
library(doParallel)
library(randomForest)
library(gbm)
library(ranger)
library(e1071)
library(gridExtra)

# Set Up
rm(list=ls())
dev.off()
path <- "C:/Users/mnelo/Documents/DAM/A2"
setwd(path)
set.seed(999)


########## Functions 
# For feature engineering and data clean up.
# Need to be called multiple times for train/test sets, and final submission.

# Data cleansing that can be applied to datasets before train/test splits 
clean_data <- function(df) {
  
  # Remove unknown release dates (only includes unknown movie_titles)
  df <- df %>% 
    filter(!is.na(release_date)) 
  
  # No data filled
  df <- df %>% 
    select(-video_release_date)
  
  # Filter out remaining "unknown" category. Only single 1* entry for User 181.
  # Will not affect their average
  df <- df %>% 
    filter(unknown!="TRUE") %>% 
    select(-unknown)
  
  # Infer US/Non-US user by assuming 5 digit numerical zip codes are from US
  df$US <- str_detect(df$zip_code, "[0-9]{5}")
  
  # Additional webscrape location data not included in the AT2_prep_students.R
  scrape <- readRDS('scrape.rds')
  scrape_location <- scrape %>% 
    select(movie_id, US_users_votes, US_users_average, 
           non_us_users_votes, non_us_users_average) 
  
  names(scrape_location)[2:ncol(scrape_location)] <- 
    c("US_votes", "US_mean", "NonUS_votes", "NonUS_mean") 
  
  # Reshape and rename
  scrape_location <- scrape_location %>% 
    gather(key=location_score, value=value, -movie_id) %>% 
    separate(col=location_score, into=c("location", "score_type")) %>% 
    spread(key = score_type, value = value) 
  
  names(scrape_location) <- c("item_id", "location",
                              "user_location_item_imdb_mean_rating",
                              "user_location_item_imdb_votes") 
  
  scrape_location <- scrape_location %>% 
    mutate(US=ifelse(location=="US",TRUE,FALSE)) %>% 
    select(-location) %>% 
    mutate(item_id=factor(item_id))
  
  # Merge to main dataset
  df <- left_join(df, scrape_location, by=c("US", "item_id"))
  
  # Weekday of release - to test as a feature
  df$weekday <- weekdays(as.Date(df$timestamp))
  
  # Seasonality - Decade most effective
  #df$month <- as.numeric(format(df$release_date, "%m"))
  df$year <- as.numeric(format(df$release_date, "%y"))
  df$decade <- as.numeric(as.integer(df$year/10)*10)
  df <- select(df, -year)
  
  # Review age - feature to test. Are fans likely to watch a movie earlier?
  df <- df %>% 
    mutate(review_age=as.numeric(as.Date(timestamp)-release_date)) 
  
  return(df)
}

# Calculate User Genre Averages - Run after train/test split to avoid leakage.
# Requires known ratings scores. 
genre_calcs <- function(df){
  
  genre_ratings <- df %>% 
    select(user_id, rating, action:western)
  genres <- names(genre_ratings)[3:length(genre_ratings)]
  
  # Get individual users to later create dataframe of results
  user_genre_details <- genre_ratings %>% 
    select(user_id) %>% 
    unique()
  
  # Loop through genres 
  for(n in 1:length(genres)){
    # Get genre mean ratings per user
    single_genre <- genre_ratings[genre_ratings[genres[n]]==TRUE,] %>% 
      group_by(user_id) %>% 
      summarise(mean=mean(rating))
    
    names(single_genre)[2] <- paste(genres[n], "mean", sep="_")
    
    # Merge all genres 
    user_genre_details <- left_join(user_genre_details, single_genre, by="user_id")
    
    # Mean imputation if missing data
    col <- names(user_genre_details)[1+n]
    user_genre_details[is.na(user_genre_details[col]), col] <- mean(unlist(user_genre_details[col]), na.rm=TRUE)
  }
  
  return(user_genre_details)
}

# Stats specific to each movie to test as features. 
item_level_stats <- function(df) {
  
  # Calculate average age of review and number of ratings relative to age of movie
  item_stats <- df %>% 
    group_by(item_id) %>% 
    summarise(rating_count=n(), 
              item_age=as.numeric(Sys.Date()-min(release_date)),
              item_mean_review_age=mean(review_age)) %>% 
    mutate(item_count_to_age=100*rating_count/item_age) %>% 
    select(item_id, item_mean_review_age, item_count_to_age) %>% 
    ungroup()
  
  df <- left_join(df, item_stats, by="item_id")
  
  return(df)
  
}

# User specific features 
# Requires known ratings scores. 
user_features <- function(df){
  
  # User's number of ratings, average rating score, standard deviation of scores, 
  #   and time since first review. Checking if user's preferences change over time. 
  features <- df %>% 
    group_by(user_id) %>% 
    summarise(user_count=n(), user_mean_rating=mean(rating), 
              user_sd_rating=sd(rating), 
              user_age=as.numeric(round(Sys.time()-min(timestamp),0))) %>% 
    ungroup()
  
  return(features)
}

# Cosine Similarity - to be used by collaborative filter tests 
cosineSim <- function(x){
  as.dist(x%*%t(x)/(sqrt(rowSums(x^2) %*% t(rowSums(x^2)))))
}

# Create User-User and Item-Item Matrices of Cosine Values
# Requires known user ratings
similarity_func <- function(df){
  
  # Item similarity 
  item_matrix <- df %>% 
    select(user_id, item_id, rating) %>% 
    spread(key=user_id, value=rating) %>% 
    mutate_all(function(x) replace_na(x, 0)) %>% 
    ungroup()
  
  # Call cosine function and rename columns
  cos_matrix <- as.matrix(cosineSim(as.matrix(item_matrix[,-1])))
  colnames(cos_matrix) <- paste0("item_cos_", colnames(cos_matrix), sep="")
  cos_matrix <- 1 - cos_matrix
  item_cos <- cbind(item_matrix[, 1], cos_matrix)
  
  # User similarity 
  user_matrix <- df %>% 
    select(user_id, item_id, rating) %>% 
    spread(key=item_id, value=rating) %>% 
    mutate_all(function(x) replace_na(x, 0)) %>% 
    ungroup()
  
  # Call cosine function and rename columns
  cos_matrix <- as.matrix(cosineSim(as.matrix(user_matrix[,-1])))
  colnames(cos_matrix) <- paste0("user_cos_", colnames(cos_matrix), sep="")
  cos_matrix <- 1 - cos_matrix
  user_cos <- cbind(user_matrix[, 1], cos_matrix)
  
  # Merge to main dataset
  df <- left_join(df, user_cos, by="user_id") 
  df <- left_join(df, item_cos, by="item_id") 
  
  return(df)
}

# Use XGBoost to impute missing values 
impute_xgb <- function(df){
  
  # Columns to check for NA's to fill
  cols_na <- c("item_imdb_rating_of_ten", "item_imdb_count_ratings", 
               "item_imdb_length", "item_imdb_staff_votes", "item_imdb_staff_average",
               "item_imdb_top_1000_voters_votes", "item_imdb_top_1000_voters_average", 
               "user_gender_item_imdb_mean_rating", "user_gender_item_imdb_votes", 
               "user_age_band_item_imdb_votes", "user_age_band_item_imdb_mean_rating", 
               "user_gender_age_band_item_imdb_votes", 
               "user_gender_age_band_item_imdb_mean_rating") 
  
  # Information to use in the model
  cols_train <- c("age", "rating", "item_mean_rating", "gender.M", "gender.F", 
                  "action", "adventure", "animation", "childrens", "comedy", 
                  "crime", "documentary", "drama", "fantasy", "film_noir",  
                  "horror", "musical", "mystery", "romance", "sci_fi", "thriller",  
                  "war", "western")
  
  # Train and fill each column in a loop
  for(i in 1:length(cols_na)){
    
    # Isolate score to train/impute
    short_df <- df[, c(cols_na[i], cols_train)] 
    
    # Format data for XGBoost
    train_df <- na.omit(short_df)
    train_df <- xgb.DMatrix(data=as.matrix(train_df[,-1]), label=train_df[,1])
    
    # Train model
    xgb_fit <- xgboost(data=train_df,
                       max_depth=6,
                       eta=0.15, 
                       nthread=2,
                       nrounds=70,
                       objective = "reg:squarederror", 
                       eval_metric="rmse",
                       verbose=0)
    
    # Add Predictions to main dataset
    df$impute <- predict(xgb_fit, as.matrix(short_df[, -1]))
    
    # Fill NA's
    df[is.na(df[cols_na[i]]), cols_na[i]] <- df[is.na(df[cols_na[i]]), "impute"]
    
  }

  # Remove prediction column
  df <- df %>% 
    select(-impute)
  
  return(df)
}

# Item Based Collaborative Filtering Score
cf_item <- function(df){
  
  #Item-User Matrix of scores 
  item_user_matrix <- df %>% 
    mutate(user_id=as.numeric(user_id), item_id=as.numeric(item_id)) %>% 
    arrange(user_id, item_id) %>% 
    select(user_id, item_id, rating) %>% 
    spread(key=user_id, value=rating)
  
  item_ids <- item_user_matrix %>% 
    select(item_id) 
  item_user_matrix <- item_user_matrix %>% 
    select(-item_id)
  
  # Mean replacement (item mean)
  item_user_matrix <- as.matrix(item_user_matrix)
  c <- which(is.na(item_user_matrix), arr.ind=TRUE)
  item_user_matrix[c] <- rowMeans(item_user_matrix, na.rm=TRUE)[c[,1]]
  
  # Create Cosine Matrix
  cos_matrix <- as.matrix(cosineSim(item_user_matrix))
  cos_matrix <- 1 - cos_matrix
  cos_sum <- rowSums(cos_matrix)
  
  # Matrix multiplication to get predicted scores 
  pred_ratings <- as.matrix(cos_matrix) %*% item_user_matrix
  pred_ratings <- pred_ratings / cos_sum
  pred_ratings <- cbind(item_id=item_ids, pred_ratings)
  
  # Reformat
  pred_long <- as.data.frame(pred_ratings) %>% 
    gather(user_id, cf_item, -item_id)
  
  return(pred_long)
}

# User-based Collaborative Filtering Scores 
cf_user <- function(df){
  
  # User-Item Matrix
  user_item_matrix <- df %>% 
    mutate(user_id=as.numeric(user_id), item_id=as.numeric(item_id)) %>% 
    arrange(user_id, item_id) %>% 
    select(user_id, item_id, rating) %>% 
    spread(key=item_id, value=rating)  
  
  user_ids <- user_item_matrix %>% 
    select(user_id)
  user_item_matrix <- user_item_matrix %>% 
    select(-user_id)
  
  # Mean impute NA's
  user_item_matrix <- user_item_matrix %>% 
    mutate_all(~ifelse(is.na(.x), mean(.x, na.rm = TRUE), .x)) 
  user_item_matrix <- as.matrix(user_item_matrix)
  
  # Create cosine matrix
  cos_matrix <- as.matrix(cosineSim(user_item_matrix))
  cos_matrix <- 1 - cos_matrix
  cos_sum <- rowSums(cos_matrix)
  
  # Matrix multiplication to get predicted scores 
  pred_ratings <- as.matrix(cos_matrix) %*% user_item_matrix
  pred_ratings <- pred_ratings / cos_sum
  pred_ratings <- cbind(user_id=user_ids, pred_ratings)
  
  # Reformat 
  pred_long <- as.data.frame(pred_ratings) %>% 
    gather(item_id, cf_user, -user_id)
  
  return(pred_long)
  
}

# Popularity feature - Weighted Mean Score to factor in number of reviews 
popularity <- function(df){
  items <- df %>% 
    select(item_id, rating)
  
  # Aggregate per item
  item_scores <- items %>% 
    group_by(item_id) %>% 
    summarise(count=n(), mean=mean(rating, na.rm=T)) %>% 
    ungroup()
  
  # Calculate global average rating 
  total_mean <- mean(unlist(items[, 2]), na.rm=T)
  
  # Final score 
  item_scores <- item_scores %>% 
    mutate(popularity=mean*count/total_mean) %>% 
    select(item_id, popularity)
  
  return(item_scores)
}


# Impute missing values of IMDB data with column means
imdb_rating_impute <- function(df){
  
  # Film length rounded to integer level for consistency with existing data
  df <- df %>% 
    mutate_at(vars(item_imdb_length), 
              ~ifelse(is.na(.x), round(mean(.x, na.rm = TRUE), 0), .x)) 
  
  # Fill Ratings NA's with column means
  df <- df %>% 
    mutate_at(vars(user_age_band_item_mean_rating, user_gender_item_mean_rating,
                   item_imdb_rating_of_ten, item_imdb_staff_average, 
                   item_imdb_top_1000_voters_average,
                   user_gender_item_imdb_mean_rating, 
                   user_age_band_item_imdb_mean_rating,
                   user_gender_age_band_item_imdb_mean_rating,
                   user_location_item_imdb_mean_rating), 
              ~ifelse(is.na(.x), mean(.x, na.rm = TRUE), .x)) 
  
  # Count/Vote columns rounded to match existing data
  df <- df %>% 
    mutate_at(vars(item_imdb_count_ratings, item_imdb_staff_votes, 
                   item_imdb_top_1000_voters_votes, user_gender_item_imdb_votes, 
                   user_age_band_item_imdb_votes, 
                   user_gender_age_band_item_imdb_votes,
                   user_location_item_imdb_votes),
              ~ifelse(is.na(.x), round(mean(.x, na.rm = TRUE), 0), .x)) 
  
  return(df)
  
}


# Dummy Variable construction for acceptable format into ranger/xgboost
gender_one_hot_encoding <- function(df){
  # Create dummy variable
  dummy <- dummyVars("~.", data=select(df, gender))
  
  # One-hot format (0 or 1)
  hot <- data.frame(predict(dummy, newdata=select(df, gender)))
  
  # Replace in main dataset
  df <- cbind(select(df, -gender), hot)
  
  return(df)
}



########## Data Set Up

# Load Initial File
data <- readRDS("AT2_train_STUDENT.rds")

# Cleaning and tranforms that don't need to wait for train/test split.
data <- clean_data(data)

# Remove other unused columns
data <- data %>% 
  select(-occupation, -zip_code, -movie_title, -imdb_url, -age_band, 
         -item_imdb_mature_rating)

# Split dataset into train/test sets for evaluating models  
indices <- createDataPartition(y=data$rating, p=0.7, list=F)
train_set <- data[indices, ]
test_set <- data[-indices, ]

# Merge Popularity Score to both sets, using only train set to generate  
item_pop <- popularity(train_set)
train_set <- left_join(train_set, item_pop, by="item_id")
test_set <- left_join(test_set, item_pop, by="item_id")

# Fill NA's if train set doesn't cover all of test set
test_set <- test_set %>% 
  mutate_at(vars(popularity), 
            ~ifelse(is.na(.x), mean(.x, na.rm = TRUE), .x)) 

# Genre means
train_genre <- genre_calcs(train_set)
train_set <- left_join(train_set, train_genre, by="user_id")
test_set <- left_join(test_set, train_genre, by="user_id")

# Item level statistics
train_set <- item_level_stats(train_set)
test_set <- item_level_stats(test_set)

# User level statistics stats - requires 'rating' column, only calc with train set
user_stats <- user_features(train_set)
train_set <- left_join(train_set, user_stats, by="user_id")
test_set <- left_join(test_set, user_stats, by="user_id")

# Format genre columns for ranger/xgboost
train_set <- train_set %>% 
  mutate_if(is.logical, as.numeric)
test_set <- test_set %>% 
  mutate_if(is.logical, as.numeric)

# Impute missing IMDB data
train_set <- imdb_rating_impute(train_set)
test_set <- imdb_rating_impute(test_set)

# Calculate User Beta Score - covariance of user vs item mean / variance of user
#   Train set only - requires known ratings scores
beta <- train_set %>% 
  select(user_id, item_id, rating, item_mean_rating) %>% 
  group_by(user_id) %>% 
  mutate(cov=cov(rating, item_mean_rating), var=var(rating), beta=cov/var) %>% 
  select(user_id, beta) %>% 
  unique() %>% 
  ungroup()

# Merge to main datasets and fill NA's (e.g. user has rated all movies same score)
train_set <- left_join(train_set, beta, by="user_id")
test_set <- left_join(test_set, beta, by="user_id")
train_set[is.na(train_set$beta), "beta"] <- 0
test_set[is.na(test_set$beta), "beta"] <- 0


# User counts per Genre - calculate with train set only 
counts <- train_set %>% 
  select(user_id, action:western) %>% 
  group_by(user_id) %>% 
  summarise_all(sum)
names(counts)[-1] <- paste0(names(counts)[-1], "_count", sep="")

train_set <- left_join(train_set, counts, by="user_id")
test_set <- left_join(test_set, counts, by="user_id")

# Check NA data
#sapply(train_set, function(x) sum(is.na(x)))
#sapply(test_set, function(x) sum(is.na(x)))

nrow(train_set) - nrow(na.omit(train_set))
nrow(test_set) - nrow(na.omit(test_set))


########## Other ineffective features tested, but not used in final models

# Cosine Matrices - Ineffective Feature
#train_set <- similarity_func(train_set)

# Use train set data to add matrix to test set
#   Simulates that ratings are unknown on true out-of-sample data

#user_cos <- train_set %>% 
#  select(user_id, user_cos_1:user_cos_943) %>% 
#  group_by(user_id) %>% 
#  unique()

#item_cos <- train_set %>% 
#  select(item_id, item_cos_1:item_cos_1633) %>% 
#  group_by(item_id) %>% 
#  unique()

#test_set <- left_join(test_set, user_cos, by="user_id")
#test_set <- left_join(test_set, item_cos, by="item_id")


# Collaborative filtering - Ineffective Feature 
#cf_item <- cf_item(train_set)
#cf_item$item_id <- as.factor(cf_item$item_id)
#cf_item$user_id <- as.factor(cf_item$user_id)

# Combine with train/test sets - only calculated using train set
#train_set <- left_join(train_set, cf_item, by=c("user_id", "item_id"))
#test_set <- left_join(test_set, cf_item, by=c("user_id", "item_id"))

#cf_user <- cf_user(train_set)
#cf_user$user_id <- as.factor(cf_user$user_id)
#cf_user$item_id <- as.factor(cf_user$item_id)
#train_set <- left_join(train_set, cf_user, by=c("user_id", "item_id"))
#test_set <- left_join(test_set, cf_user, by=c("user_id", "item_id"))

# Convert Gender Column to Dummy Variable 
#train_set <- gender_one_hot_encoding(train_set)
#test_set <- gender_one_hot_encoding(test_set)

# Impute missing values using xgboost predictions
#train_set <- impute_xgb(train_set)
#test_set <- impute_xgb(test_set)


########## Outlier Filtering - Ineffective 

# Attempts that did not improve model results 
#     - filter out top/bottom 10 users by average rating
#     - top/bottom 10 by count 
#     - top only by count (common for users to have a lower count)
#     - absolute value of rating - user mean rating > 3.5 (e.g. all 5 ratings except for single 1 rating)


#lower_bound <- data %>% 
#  group_by(user_id) %>% 
#  summarise(count=n(), mean=mean(rating)) %>% 
#  arrange(count) %>% 
#  head(10) %>% 
#  tail(1) %>% 
#  select(count) %>% 
#  unlist()

#upper_bound <- data %>% 
#  group_by(user_id) %>% 
#  summarise(count=n(), mean=mean(rating)) %>% 
#  arrange(desc(count)) %>% 
#  head(10) %>% 
#  tail(1) %>% 
#  select(count) %>% 
#  unlist()

#train_set <- train_set %>% 
#  filter(user_count < upper_bound)

#train_set %>% 
#  mutate(diff=abs(rating-user_mean_rating)) %>% 
#  select(user_id, item_id, rating, diff) %>% 
#  arrange(desc(diff)) %>% 
#  head(10)

#train_set <- train_set %>% 
#  mutate(diff=abs(rating-user_mean_rating)) %>% 
#  filter(diff<=3.5) %>% 
#  select(-diff)


########## Train model

### Random Forest Model (Best out-of-sample result)

# Remove unused columns 
train_rf <- train_set %>% 
  select(-user_id, -item_id, -timestamp, -release_date, -weekday, -US, -gender)
test_rf <- test_set[, names(train_rf)] 


# Model - REMOVE importance if charts are not required - slow computation
rf <- ranger(formula=rating~., 
             data=train_rf, 
             num.trees=4000, 
             importance="permutation") 

pred_rf <- predict(rf, test_rf)
RMSE(pred_rf$predictions, test_rf$rating) 

# RMSE = 0.8807584

# Below grid search for hyperparameter tuning ineffective 
#   Increased the num.trees instead

# Grid search
#grid <- expand.grid(mtry=seq(20, 30, by=2),
#                    node_size=seq(3, 9, by=2),
#                    sample_size=c(.55, .632, .70, .80),
#                    OOB_RMSE=0)

# Loop for each combination in grid
#for(i in 1:nrow(grid)) {
#  # Train single model
#  model <- ranger(formula=rating~ ., 
#                  data=train_rf, 
#                  num.trees=1000,
#                  mtry=grid$mtry[i],
#                  min.node.size=grid$node_size[i],
#                  sample.fraction=grid$sample_size[i],
#                  seed=999)
#  
#  # OOB error
#  grid$OOB_RMSE[i] <- sqrt(model$prediction.error)
#}

#grid %>% 
#  arrange(OOB_RMSE) %>% 
#  head(10)


# Variable importance
# Top Variables
p <- stack(rf$variable.importance) %>% 
  arrange(desc(values)) %>% 
  head(20) %>% #tail(20)
  ggplot(aes(x=reorder(ind, values),y=values, fill=values)) + 
  geom_col() +
  coord_flip()
p

# View full list
#stack(rf$variable.importance) %>% 
#  arrange(desc(values)) 


########## Retrain model using full dataset to create Kaggle submission

# Reformat full data set
full_df <- data

# Recalculations 
# Popularity score
item_pop_full <- popularity(full_df)
full_df <- left_join(full_df, item_pop_full, by="item_id")

# User means per Genre
genre_full <- genre_calcs(full_df)
full_df <- left_join(full_df, genre_full, by="user_id")

# Item level statistics
full_df <- item_level_stats(full_df)

# User level statistics 
user_stats_full <- user_features(full_df)
full_df <- left_join(full_df, user_stats_full, by="user_id")

# Genre formatting
full_df <- full_df %>% 
  mutate_if(is.logical, as.numeric)

# IMDB imputation
full_df <- imdb_rating_impute(full_df)

# Genre counts per User
counts_full <- full_df %>% 
  select(user_id, action:western) %>% 
  group_by(user_id) %>% 
  summarise_all(sum)
names(counts_full)[-1] <- paste0(names(counts_full)[-1], "_count", sep="")

full_df <- left_join(full_df, counts_full, by="user_id")

# User Beta Scores
beta_full <- full_df %>% 
  select(user_id, item_id, rating, item_mean_rating) %>% 
  group_by(user_id) %>% 
  mutate(cov=cov(rating, item_mean_rating), var=var(rating), beta=cov/var) %>% 
  select(user_id, beta) %>% 
  unique() %>% 
  ungroup()

full_df <- left_join(full_df, beta_full, by="user_id")
full_df[is.na(full_df$beta), "beta"] <- 0

# Remove unused columns 
full_df <- full_df %>% 
  select(-user_id, -gender, -item_id, -timestamp, -release_date, -weekday, -US)

# Check NA's
nrow(full_df) - nrow(na.omit(full_df))

# Retrain model 
rf_final <- ranger(formula=rating~., 
                   data=full_df, 
                   num.trees=4000) 
                   #importance="permutation


########## Kaggle Submission File 

# Load and format new data set 
sub_data <- readRDS("AT2_test_STUDENT.rds")
sub_data <- clean_data(sub_data)

# Merge features engineered from train data
sub_data <- left_join(sub_data, item_pop_full, by="item_id")
sub_data <- left_join(sub_data, genre_full, by="user_id")
sub_data <- left_join(sub_data, user_stats_full, by="user_id")
sub_data <- left_join(sub_data, counts_full, by="user_id")
sub_data <- left_join(sub_data, beta_full, by="user_id")
sub_data[is.na(sub_data$beta), "beta"] <- 0

# Item level stats
sub_data <- item_level_stats(sub_data)

# Genre formatting
sub_data <- sub_data %>% 
  mutate_if(is.logical, as.numeric)

# Impute missing data
sub_data <- imdb_rating_impute(sub_data)

# Check NA's
nrow(sub_data) - nrow(na.omit(sub_data))

# Predictions
pred_sub <-  predict(rf_final, sub_data[, names(full_df)[-2]])

# Merge to main dataset
sub_data$rating <- pred_sub$predictions
sub_data$user_item <- paste(sub_data$user_id, sub_data$item_id, sep="_")

# Create submission file
submission_file <- sub_data %>% 
  select(rating, user_item) 

# Check for missed predictions 
nrow(submission_file) - nrow(na.omit(submission_file))

# Export file
write.csv(submission_file, "kaggle_upload.csv", row.names=FALSE)

# RMSE = 0.90683 on Kaggle 


########## Alternative models of interest tested

### XGBoost
# Reformat train/test sets 
xgb_train <- xgb.DMatrix(data=as.matrix(train_rf[-2]), label=train_rf$rating)
xgb_test <- xgb.DMatrix(data=as.matrix(test_rf[-2]), label=test_rf$rating)

# Cross Validation 
params <- list(booster="gbtree", 
               objective = "reg:squarederror", 
               eta=0.15, 
               gamma=0, 
               max_depth=7)

xgbcv <- xgb.cv(params=params, 
                data=xgb_train,
                nrounds=300, 
                nfold=5, 
                showsd=T, 
                stratified=T, 
                print_every_n=10, 
                early_stopping_rounds=20, 
                maximize=F)

# Model
xgb_fit <- xgboost(data=xgb_train,
                   max_depth=7,
                   eta=0.1, 
                   nthread=2,
                   nrounds=xgbcv$best_iteration,
                   objective = "reg:squarederror", # #"reg:linear"
                   eval_metric="rmse",
                   verbose=0)

# Evaluate
pred_xgb <- predict(xgb_fit, xgb_test)
RMSE(pred_xgb, test_rf$rating) 

# RMSE = 0.8826862  
# 0.93614 on Kaggle submission - significant difference



### Hybrid Model (RF and XGBoost)

RMSE(0.5*pred_xgb + 0.5*pred_rf$predictions, test_rf$rating) 

# RMSE = 0.8742463
# Some improvement on RF but worse on out-of-sample
# 0.91145 on Kaggle submission



