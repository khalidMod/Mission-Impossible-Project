
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

# Open File
path <- "C:/Users/mnelo/Documents/DAM/A2"
setwd(path)
data <- readRDS("AT2_train_STUDENT.rds")
set.seed(999)

## Clean Data
# View NA data
sapply(data, function(x) sum(is.na(x)))

# Remove unknown release dates (all unknown movie_titles)
data <- data %>% 
  filter(!is.na(release_date)) 

# No data filled
data <- data %>% 
  select(-video_release_date)

# Filter out remaining "unknown" category. Only single 1* entry for User 181.
# Will not affect their average
data <- data %>% 
  filter(unknown!="TRUE") %>% 
  select(-unknown)


## Functions for imputing NA values

# Function to estimate missing imdb ratings and
imdb_rating_estimate <- function(df){
  
  # Estimate imdb ratings using ratings
  new_ratings <- df %>% 
    filter(is.na(item_imdb_rating_of_ten)) %>% 
    group_by(item_id) %>% 
    summarise(mean=mean(10/5*rating), sd=sd(rating), item_imdb_count_ratings=n()) %>% 
    ungroup()
  new_ratings[is.na(new_ratings$sd), "sd"] <- 0
  
  new_ratings$item_imdb_rating_of_ten <- new_ratings$mean + rnorm(length(new_ratings$sd)) * new_ratings$sd
  
  # Combine old ratings with imputed
  cols <- c("item_id", "item_imdb_rating_of_ten", "item_imdb_count_ratings")
  old_ratings <- df %>% 
    filter(!is.na(item_imdb_rating_of_ten)) %>% 
    group_by(item_id) %>% 
    select(cols) %>% 
    unique() %>% 
    ungroup()
  
  combined_ratings <- rbind(old_ratings, select(new_ratings, cols))
  
  # Return combined ratings - will be used to impute test/validation sets
  return(combined_ratings)
}

# Calculate Genre Averages
genre_calcs <- function(df){
  
  genre_ratings <- df %>% 
    select(user_id, rating, action:western)
  genres <- names(genre_ratings)[3:length(genre_ratings)]
  
  user_genre_details <- genre_ratings %>% 
    select(user_id) %>% 
    unique()
  
  for(n in 1:length(genres)){
    # Get genre mean ratings per user
    single_genre <- genre_ratings[genre_ratings[genres[n]]==TRUE,] %>% 
      group_by(user_id) %>% 
      summarise(mean=mean(rating))
    
    names(single_genre)[2] <- paste(genres[n], "mean", sep="_")
    
    # Merge all genres 
    user_genre_details <- left_join(user_genre_details, single_genre, by="user_id")
    
    # Mean imputation
    col <- names(user_genre_details)[1+n]
    user_genre_details[is.na(user_genre_details[col]), col] <- mean(unlist(user_genre_details[col]), na.rm=TRUE)
  }
  
  return(user_genre_details)
}

# Item level imputation
item_imputation <- function(df){

  # Fill missing counts/votes with zero
  df <- df %>%  
    mutate(item_imdb_count_ratings=replace_na(item_imdb_count_ratings, 0),
           item_imdb_staff_votes=replace_na(item_imdb_staff_votes, 0),
           item_imdb_top_1000_voters_votes=replace_na(item_imdb_top_1000_voters_votes, 0))

  # Mean reviews
  df$item_imdb_rating_of_ten[is.na(df$item_imdb_rating_of_ten)] <- 
    mean(df$item_imdb_rating_of_ten, na.rm=T) + 
      sd(df$item_imdb_rating_of_ten, na.rm=T) * 
      rnorm(length(df$item_imdb_rating_of_ten[is.na(df$item_imdb_rating_of_ten)]))
      
  df$item_imdb_staff_average[is.na(df$item_imdb_staff_average)] <- 
    mean(df$item_imdb_staff_average, na.rm=T) + 
    sd(df$item_imdb_staff_average, na.rm=T) * 
    rnorm(length(df$item_imdb_staff_average[is.na(df$item_imdb_staff_average)]))
  
  df$item_imdb_top_1000_voters_average[is.na(df$item_imdb_top_1000_voters_average)] <- 
    mean(df$item_imdb_top_1000_voters_average, na.rm=T) + 
    sd(df$item_imdb_top_1000_voters_average, na.rm=T) * 
    rnorm(length(df$item_imdb_top_1000_voters_average[is.na(df$item_imdb_top_1000_voters_average)]))
  
  # Mean length (rounded)
  df$item_imdb_length[is.na(df$item_imdb_length)] <- 
    round(mean(df$item_imdb_length, na.rm=T) + 
            sd(df$item_imdb_length, na.rm=T) * 
            rnorm(length(df$item_imdb_length[is.na(df$item_imdb_length)])), 0)
  
  return(df)
}

# User level imputations
user_imputation <- function(df){
  
  # Subgroup aggregations 
  new_age_band_mean <- df %>% 
    group_by(age_band) %>% 
    summarise(new_age_band_mean=mean(item_mean_rating, na.rm=TRUE),
              new_age_band_sd=sd(item_mean_rating, na.rm=TRUE))
  
  new_gender_mean <- df %>% 
    group_by(gender) %>% 
    summarise(new_gender_mean=mean(item_mean_rating, na.rm=TRUE),
              new_gender_sd=sd(item_mean_rating, na.rm=TRUE))
  
  new_gender_imdb_mean <- df %>% 
    group_by(gender) %>% 
    summarise(new_gender_imdb_mean=mean(item_imdb_rating_of_ten, na.rm=TRUE),
              new_gender_imdb_sd=sd(item_imdb_rating_of_ten, na.rm=TRUE))
  
  new_age_band_imdb_mean <- df %>% 
    group_by(age_band) %>% 
    summarise(new_age_band_imdb_mean=mean(item_imdb_rating_of_ten, na.rm=TRUE),
              new_age_band_imdb_sd=sd(item_imdb_rating_of_ten, na.rm=TRUE))
  
  new_gender_age_band_imdb_mean <- df %>% 
    group_by(gender, age_band) %>% 
    summarise(new_gender_age_band_imdb_mean=mean(item_imdb_rating_of_ten, na.rm=TRUE),
              new_gender_age_band_imdb_sd=sd(item_imdb_rating_of_ten, na.rm=TRUE))
  
  
  # Merge new means/sd with main data set
  df <- left_join(df, new_age_band_mean, by="age_band")
  df <- left_join(df, new_gender_mean, by="gender")
  df <- left_join(df, new_gender_imdb_mean, by="gender")
  df <- left_join(df, new_age_band_imdb_mean, by="age_band")
  df <- left_join(df, new_gender_age_band_imdb_mean, by=c("gender", "age_band"))
  
  
  # Fill NA's using new mean/sd values
  df$user_age_band_item_mean_rating[is.na(df$user_age_band_item_mean_rating)] <- 
    df$new_age_band_mean[is.na(df$user_age_band_item_mean_rating)] + 
    df$new_age_band_sd[is.na(df$user_age_band_item_mean_rating)] * 
    rnorm(length(df$user_age_band_item_mean_rating[is.na(df$user_age_band_item_mean_rating)]))
  
  df$user_gender_item_mean_rating[is.na(df$user_gender_item_mean_rating)] <- 
    df$new_gender_mean[is.na(df$user_gender_item_mean_rating)] + 
    df$new_gender_sd[is.na(df$user_gender_item_mean_rating)] * 
    rnorm(length(df$user_gender_item_mean_rating[is.na(df$user_gender_item_mean_rating)]))
  
  df$user_gender_item_imdb_mean_rating[is.na(df$user_gender_item_imdb_mean_rating)] <- 
    df$new_gender_imdb_mean[is.na(df$user_gender_item_imdb_mean_rating)] + 
    df$new_gender_imdb_sd[is.na(df$user_gender_item_imdb_mean_rating)] * 
    rnorm(length(df$user_gender_item_imdb_mean_rating[is.na(df$user_gender_item_imdb_mean_rating)]))
  

  df$user_age_band_item_imdb_mean_rating[is.na(df$user_age_band_item_imdb_mean_rating)] <- 
    df$new_age_band_imdb_mean[is.na(df$user_age_band_item_imdb_mean_rating)] + 
    df$new_age_band_imdb_sd[is.na(df$user_age_band_item_imdb_mean_rating)] * 
    rnorm(length(df$user_age_band_item_imdb_mean_rating[is.na(df$user_age_band_item_imdb_mean_rating)]))
    
  
  df$user_gender_age_band_item_imdb_mean_rating[is.na(df$user_gender_age_band_item_imdb_mean_rating)] <- 
    df$new_gender_age_band_imdb_mean[is.na(df$user_gender_age_band_item_imdb_mean_rating)] + 
    df$new_gender_age_band_imdb_sd[is.na(df$user_gender_age_band_item_imdb_mean_rating)] * 
    rnorm(length(df$user_gender_age_band_item_imdb_mean_rating[is.na(df$user_gender_age_band_item_imdb_mean_rating)]))
  
  # Fill missing vote/counts with zero
  df <- df %>%  
    mutate(user_age_band_item_imdb_votes=replace_na(user_age_band_item_imdb_votes, 0),
           user_gender_item_imdb_votes=replace_na(user_gender_item_imdb_votes, 0),
           user_gender_age_band_item_imdb_votes=replace_na(user_gender_age_band_item_imdb_votes, 0))
  
  return(df)
}

# User level aggregations:
#   Number of reviews, average rating, variance of ratings, time reviewing
user_features <- function(df){
  # Aggregation
  features <- df %>% 
    group_by(user_id) %>% 
    summarise(user_count=n(), user_mean_rating=mean(rating), 
              user_sd_rating=sd(rating), 
              user_age=as.numeric(round(Sys.time()-min(timestamp),0))) %>% 
    ungroup()
  
  # Return user level stats - final data does not contain ratings column
  return(features)
}


# Train/Test Split 
data <- data[sample(1:nrow(data)), ]

cut_off <- round(nrow(data)*0.7, 0)
train_set <- data[1:cut_off, ]
test_set <- data[-(1:cut_off), ]


# Impute missing values (after train/test split to avoid test data leakage)

# Item and user level aggregations - can be re-used on test/validation sets 
imdb_imputed_ratings <- imdb_rating_estimate(train_set)
user_stats <- user_features(train_set)
user_genre_stats <- genre_calcs(train_set)

# Merge to main data set
train_set <- train_set %>% 
  select(-item_imdb_rating_of_ten, -item_imdb_count_ratings)
train_set <- left_join(train_set, imdb_imputed_ratings, by="item_id")
train_set <- left_join(train_set, user_genre_stats, by="user_id")
train_set <- left_join(train_set, user_stats, by="user_id")

train_set <- item_imputation(train_set)
train_set <- user_imputation(train_set)

train_set <- train_set %>% 
  group_by(item_id) %>% 
  mutate(review_rank=rank(timestamp)) %>% 
  ungroup()

# Set up Test set
test_set <- test_set %>% 
  select(-item_imdb_rating_of_ten, -item_imdb_count_ratings)
test_set <- left_join(test_set, imdb_imputed_ratings, by="item_id")
test_set <- left_join(test_set, user_genre_stats, by="user_id")
test_set <- left_join(test_set, user_stats, by="user_id")

test_set <- item_imputation(test_set)
test_set <- user_imputation(test_set)

test_set <- test_set %>% 
  group_by(item_id) %>% 
  mutate(review_rank=rank(timestamp)) %>% 
  ungroup()



# XGboost model

one_hot_format <- function(df){
  # Gender formatting
  dummy <- dummyVars("~.", data=select(df, gender))
  hot <- data.frame(predict(dummy, newdata=select(df, user_id, gender)))
  boost_df <- cbind(df, hot)
  
  # Genres
  boost_df <- boost_df %>% 
    mutate_if(is.logical, as.numeric)
  
  # Age Band 
  df$age_band <- as.character(df$age_band)
  dummy <- dummyVars("~.", data=select(df, age_band))
  hot <- data.frame(predict(dummy, newdata=select(df, user_id, age_band)))
  boost_df <- cbind(boost_df, hot)
  
  # Genre formatting
  # dummy <- dummyVars("~.", data=select(df, action:western))
  # hot <- data.frame(predict(dummy, newdata=select(df, user_id, action:western)))
  # boost_df <- cbind(df, hot)
  
  return(boost_df)
}

# Set up formatting of data
train_xgb <- one_hot_format(train_set)
train_xgb <- train_xgb %>% 
  select(rating, action:western, 
         item_mean_rating:user_gender_item_mean_rating, 
         item_imdb_length:user_age, review_rank:age_bandunder_18)
train_matrix <- xgb.DMatrix(data=as.matrix(train_xgb[-1]), label=train_xgb$rating)

test_xgb <- one_hot_format(test_set)
test_xgb <- test_xgb %>% 
  select(rating, action:western, 
         item_mean_rating:user_gender_item_mean_rating, 
         item_imdb_length:user_age, review_rank:age_bandunder_18)
test_matrix <- xgb.DMatrix(data=as.matrix(test_xgb[-1]), label=test_xgb$rating)

# Train model
xgb_fit <- xgboost(data=train_matrix,
                   max_depth=6,
                   eta=0.15, 
                   nthread=2,
                   nrounds=70,
                   objective = "reg:squarederror", # #"reg:linear"
                   eval_metric="rmse",
                   verbose=0)

# Evaluate
pred <- predict(xgb_fit, test_matrix)
RMSE(pred, test_set$rating) # 0.8837407


# Cross Validation Version
params <- list(booster="gbtree", 
               objective = "reg:squarederror", 
               eta=0.15, 
               gamma=0, 
               max_depth=6, 
               min_child_weight=1, 
               subsample=1, 
               colsample_bytree=1)

xgbcv <- xgb.cv(params=params, 
                data=train_matrix,
                nrounds=300, 
                nfold=5, 
                showsd=T, 
                stratified=T, 
                print_every_n=10, 
                early_stopping_rounds=20, 
                maximize=F)

xgb_fit <- xgboost(data=train_matrix,
                   max_depth=6,
                   eta=0.15, 
                   nthread=2,
                   nrounds=xgbcv$best_iteration,
                   objective = "reg:squarederror", # #"reg:linear"
                   eval_metric="rmse",
                   verbose=0)

# Evaluate
pred <- predict(xgb_fit, test_matrix)
RMSE(pred, test_set$rating) # 0.8902206


# Predictions and Submission file
# Set up new file
new_data <- readRDS("AT2_test_STUDENT.rds")

# Add user level data from training set
user_stats <- train_set %>% 
  select(user_id, action_mean:western_mean, user_count:user_age) %>% 
  group_by(user_id) %>% 
  unique() %>% 
  ungroup() 
new_data <- left_join(new_data, user_stats, by="user_id")
new_data$user_id <- as.factor(new_data$user_id)

# Add review rank
new_data <- new_data %>% 
  group_by(item_id) %>% 
  mutate(review_rank=rank(timestamp)) %>% 
  ungroup()

# Fill item level data from training set
cols <- names(select(new_data, item_imdb_rating_of_ten:item_imdb_top_1000_voters_average))
for(n in cols){
  if(n %in% c("item_imdb_count_ratings", "item_imdb_staff_votes",
              "item_imdb_top_1000_voters_votes")) {
    # Counting Items
    new_data[is.na(new_data[, n]), n] <- 0
    
  } else if (n %in% c("item_imdb_rating_of_ten", 
                      "item_imdb_staff_average", 
                      "item_imdb_top_1000_voters_average")){
    # Rating Items
    new_data[is.na(new_data[, n]), n] <- mean(unlist(new_data[, n]), na.rm=TRUE) + rnorm(1)* sd(unlist(new_data[, n]), na.rm=TRUE)
    
  } else if (n=="item_imdb_length") {
    new_data[is.na(new_data[, n]), n] <- round(mean(unlist(new_data[, n]), na.rm=TRUE) + rnorm(1)* sd(unlist(new_data[, n]), na.rm=TRUE), 0)
  }
  
  # Skip - not included in model
  # "item_imdb_mature_rating"
  
}


# Calc mean/sd values for user subgroupings
new_age_band_mean <- new_data %>% 
  group_by(age_band) %>% 
  summarise(new_age_band_mean=mean(item_mean_rating, na.rm=TRUE),
            new_age_band_sd=sd(item_mean_rating, na.rm=TRUE))

new_gender_mean <- new_data %>% 
  group_by(gender) %>% 
  summarise(new_gender_mean=mean(item_mean_rating, na.rm=TRUE),
            new_gender_sd=sd(item_mean_rating, na.rm=TRUE))

new_gender_imdb_mean <- new_data %>% 
  group_by(gender) %>% 
  summarise(new_gender_imdb_mean=mean(item_imdb_rating_of_ten, na.rm=TRUE),
            new_gender_imdb_sd=sd(item_imdb_rating_of_ten, na.rm=TRUE))

new_age_band_imdb_mean <- new_data %>% 
  group_by(age_band) %>% 
  summarise(new_age_band_imdb_mean=mean(item_imdb_rating_of_ten, na.rm=TRUE),
            new_age_band_imdb_sd=sd(item_imdb_rating_of_ten, na.rm=TRUE))

new_gender_age_band_imdb_mean <- new_data %>% 
  group_by(gender, age_band) %>% 
  summarise(new_gender_age_band_imdb_mean=mean(item_imdb_rating_of_ten, na.rm=TRUE),
            new_gender_age_band_imdb_sd=sd(item_imdb_rating_of_ten, na.rm=TRUE))

# Merge new means/sd with main data set
new_data <- left_join(new_data, new_age_band_mean, by="age_band")
new_data <- left_join(new_data, new_gender_mean, by="gender")
new_data <- left_join(new_data, new_gender_imdb_mean, by="gender")
new_data <- left_join(new_data, new_age_band_imdb_mean, by="age_band")
new_data <- left_join(new_data, new_gender_age_band_imdb_mean, by=c("gender", "age_band"))

# Fill NA's using new mean/sd values
new_data[is.na(new_data$user_age_band_item_mean_rating), "user_age_band_item_mean_rating"] <- new_data %>% 
  filter(is.na(user_age_band_item_mean_rating)) %>% 
  mutate(user_age_band_item_mean_rating=new_age_band_mean + rnorm(1)*new_age_band_sd) %>% 
  select(user_age_band_item_mean_rating)

new_data[is.na(new_data$user_gender_item_mean_rating), "user_gender_item_mean_rating"] <- new_data %>% 
  filter(is.na(user_gender_item_mean_rating)) %>% 
  mutate(user_gender_item_mean_rating=new_gender_mean + rnorm(1)*new_gender_sd) %>% 
  select(user_gender_item_mean_rating)

new_data[is.na(new_data$user_gender_item_imdb_mean_rating), "user_gender_item_imdb_mean_rating"] <- new_data %>% 
  filter(is.na(user_gender_item_imdb_mean_rating)) %>% 
  mutate(user_gender_item_imdb_mean_rating=new_gender_imdb_mean + rnorm(1)*new_gender_imdb_sd) %>% 
  select(user_gender_item_imdb_mean_rating)

new_data[is.na(new_data$user_age_band_item_imdb_mean_rating), "user_age_band_item_imdb_mean_rating"] <- new_data %>% 
  filter(is.na(user_age_band_item_imdb_mean_rating)) %>% 
  mutate(user_age_band_item_imdb_mean_rating=new_age_band_imdb_mean + rnorm(1)*new_age_band_imdb_sd) %>% 
  select(user_age_band_item_imdb_mean_rating)

new_data[is.na(new_data$user_gender_age_band_item_imdb_mean_rating), "user_gender_age_band_item_imdb_mean_rating"] <- new_data %>% 
  filter(is.na(user_gender_age_band_item_imdb_mean_rating)) %>% 
  mutate(user_gender_age_band_item_imdb_mean_rating=new_gender_age_band_imdb_mean + rnorm(1)*new_gender_age_band_imdb_sd) %>% 
  select(user_gender_age_band_item_imdb_mean_rating)

# Fill missing vote numbers
new_data[is.na(new_data$user_gender_item_imdb_votes), "user_gender_item_imdb_votes"] <- 0
new_data[is.na(new_data$user_age_band_item_imdb_votes), "user_age_band_item_imdb_votes"] <- 0
new_data[is.na(new_data$user_gender_age_band_item_imdb_votes), "user_gender_age_band_item_imdb_votes"] <- 0

# Double check missing data has been filled
sapply(new_data, function(x) sum(is.na(x)))


# XGboost format
new_xgb <- one_hot_format(new_data)
new_xgb <- new_xgb[, names(train_xgb)[-1]]
new_matrix <- xgb.DMatrix(data=as.matrix(new_xgb))

# Predictions and submission file
new_data$rating = predict(xgb_fit, newdata=new_matrix)
new_data$user_item <- paste(new_data$user_id, new_data$item_id, sep="_")

submission_file <- new_data %>% 
  select(rating, user_item) 

# check for missed predictions 
nrow(submission_file)
nrow(na.omit(submission_file))

write.csv(submission_file, "xgb_submission.csv", row.names=FALSE)


