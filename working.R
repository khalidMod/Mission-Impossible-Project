
library(tidyverse)
library(ggplot2)
library(corrplot)
library(caret)
library(randomForest)
library(xgboost)
library(parallel)
library(doParallel)
library(ranger)
library(e1071)
library(gbm)
library(gridExtra)

data <- readRDS("AT2_train_STUDENT.rds")

### EDA

## Check correlations of ratings - use most similar to determine imputation
# Filter Ratings variables only
rate_cols <- c("user_id", "item_mean_rating", "user_age_band_item_mean_rating",             
               "user_gender_item_mean_rating", "item_imdb_rating_of_ten",
               "item_imdb_staff_average", "item_imdb_top_1000_voters_average",         
               "user_gender_item_imdb_mean_rating", 
               "user_age_band_item_imdb_mean_rating",
               "user_gender_age_band_item_imdb_mean_rating")

# Shorten names for plots
short_cols <- c("user_id", "mean_rating", "user_age_band", "user_gender",
                "imdb", "imdb_staff", "imdb_top_1000", 
                "user_gender_imdb", "user_age_imdb",
                "user_gender_age_imdb")
all_ratings <- data %>% 
  select(rate_cols) 
names(all_ratings) <- short_cols

# Plot correlation between mean scores per movie
corr_matrix <- all_ratings %>% 
  select(-user_id) %>% 
  na.omit() %>% 
  cor(method="pearson")
corrplot(corr_matrix, method="color", type="upper", tl.srt=45,
         addCoef.col="black", number.cex=.7)

# Convert 5* scale to 10
all_ratings <- all_ratings %>% 
  mutate(mean_rating=10/5*mean_rating, 
         user_age_band=10/5*user_age_band,
         user_gender=10/5*user_gender)

# Plot original ratings
p <- all_ratings %>% 
  select(user_id, mean_rating, imdb, imdb_staff, imdb_top_1000) %>% 
  pivot_longer(-c("user_id"), names_to="category", values_to="values") %>% 
  ggplot(aes(x=values, group=category, fill=category)) +
  geom_density() +
  facet_wrap(~category, ncol=2)+
  theme(legend.position="none") + 
  labs(title="Distribution of Ratings",
       subtitle="Per User, Normalised to ten-point scale", 
       x="Rating")
p

# Plot derivative ratings 
p <- all_ratings %>% 
  select(-mean_rating, -imdb, -imdb_staff, -imdb_top_1000) %>% 
  pivot_longer(-c("user_id"), names_to="category", values_to="values") %>% 
  ggplot(aes(x=values, group=category, fill=category)) +
  geom_density() +
  facet_wrap(~category, ncol=2)+
  theme(legend.position="none") + 
  labs(title="Distribution of Ratings",
       subtitle="Averaged by User subgroup", 
       x="Rating")
p

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

# Show User 181 - mostly 1* ratings
data %>% 
  filter(user_id==181) %>% 
  ggplot(aes(x=rating, fill=rating)) + 
  geom_bar(fill="steelblue") 


# Look at genre details
genre_ratings <- data %>% 
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
}

# Plot Distributions
p <- user_genre_details %>% 
  pivot_longer(-c("user_id"), names_to="category", values_to="values") %>% 
  ggplot(aes(x=values, group=category, fill=category)) +
  geom_density() +
  facet_wrap(~category, ncol=6, scales="free_y") + 
  theme(legend.position="none") + 
  labs(title="Genre Distributions", 
       subtitle="User Average per Genre", 
       x="Rating", 
       y="")
p  
       
# Rating counts per Genre
genre_counts <- genre_ratings %>% 
  select(-rating) %>% 
  group_by(user_id) %>% 
  summarise_all(sum) %>% 
  ungroup()

# Plot counts
p <- genre_counts %>% 
  pivot_longer(-c("user_id"), names_to="category", values_to="values") %>% 
  ggplot(aes(x=reorder(category, values), y=values, fill=category)) + 
  geom_boxplot() +
  theme(legend.position="none", axis.text.x=element_text(angle=45)) + 
  labs(title="Number of User Reviews", 
       subtitle="Per Genre",
       x="Genres",
       y="Count")
p 
       
# Recalc genre aggregations after train/test split to avoid leakage in model training

# Train/Test Split - before imputation to avoid test set leakage
set.seed(999)
data <- data[sample(1:nrow(data)), ]

cut_off <- round(nrow(data)*0.7, 0)
train_set <- data[1:cut_off, ]
test_set <- data[-(1:cut_off), ]


# Function to estimate missing imdb ratings and
imdb_rating_estimate <- function(df){
  # Estimate imdb ratings using ratings
  new_ratings <- df %>% 
    filter(is.na(item_imdb_rating_of_ten)) %>% 
    group_by(movie_title) %>% 
    summarise(mean=mean(10/5*rating), sd=sd(rating), item_imdb_count_ratings=n()) %>% 
    ungroup()
  new_ratings[is.na(new_ratings$sd), "sd"] <- 0

  # 
  new_ratings$item_imdb_rating_of_ten <- new_ratings$mean + rnorm(1) * new_ratings$sd
  
  # Merge with valid data
  cols <- c("movie_title", "item_imdb_rating_of_ten", "item_imdb_count_ratings")
  old_ratings <- df %>% 
    filter(!is.na(item_imdb_rating_of_ten)) %>% 
    group_by(movie_title) %>% 
    select(cols) %>% 
    unique() %>% 
    ungroup()
  
  combined_ratings <- rbind(old_ratings, select(new_ratings, cols))
  
  return(combined_ratings)
}


# Fill missing imdb ratings from averaging user's ratings
train_ratings <- imdb_rating_estimate(train_set)
test_ratings <- imdb_rating_estimate(test_set)

train_set <- train_set %>% 
  select(-item_imdb_rating_of_ten, -item_imdb_count_ratings)
train_set <- left_join(train_set, train_ratings, by="movie_title")

test_set <- test_set %>% 
  select(-item_imdb_rating_of_ten, -item_imdb_count_ratings)
test_set <- left_join(test_set, test_ratings, by="movie_title")


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

train_genre <- genre_calcs(train_set)
test_genre <- genre_calcs(test_set)

train_set <- left_join(train_set, train_genre, by="user_id")
test_set <- left_join(test_set, test_genre, by="user_id")

       
#train_set %>% 
#  group_by(user_id) %>% 
#  summarise(user_mean=10/5*mean(rating), imdb=mean(item_imdb_rating_of_ten)) 

# PLACEHOLDER - remove remaining NA's so we can train and prediction
nrow(train_set)
nrow(test_set)

train_set <- na.omit(train_set)
test_set <- na.omit(test_set)

nrow(train_set)
nrow(test_set)
#


## Feature Engineering
# User level aggregations:
#   Number of reviews, average rating, variance of ratings, time reviewing
user_features <- function(df){
  # Aggregation
  features <- df %>% 
    group_by(user_id) %>% 
    summarise(user_count=n(), user_mean_rating=mean(rating), 
              user_sd_rating=sd(rating), user_age=as.numeric(round(Sys.time()-min(timestamp),0))) %>% 
    ungroup()
  
  # Merge to original dataframe and return
  df <- left_join(df, features, by="user_id")
  return(df)
}

train_set <- user_features(train_set)
test_set <- user_features(test_set)

# Time between review and release date
train_set <- train_set %>% 
  group_by(movie_title) %>% 
  mutate(review_rank=rank(timestamp)) %>% 
  ungroup()

test_set <- test_set %>% 
  group_by(movie_title) %>% 
  mutate(review_rank=rank(timestamp)) %>% 
  ungroup()


## Train Model

# Random Forest model
train_cols <- c("rating", "age", "gender", 
                "item_mean_rating", "user_age_band_item_mean_rating",
                "user_gender_item_mean_rating", 
                "item_imdb_length", "item_imdb_staff_votes", "item_imdb_staff_average", 
                "item_imdb_top_1000_voters_votes", "item_imdb_top_1000_voters_average", 
                "user_gender_item_imdb_mean_rating", "user_gender_item_imdb_votes", 
                "user_age_band_item_imdb_votes", "user_age_band_item_imdb_mean_rating",
                "user_gender_age_band_item_imdb_votes", 
                "user_gender_age_band_item_imdb_mean_rating", "item_imdb_rating_of_ten",                  
                "item_imdb_count_ratings", "user_count", "user_mean_rating", 
                "user_sd_rating", "user_age", "review_rank",
                "action_mean", "adventure_mean", "animation_mean",
                "childrens_mean", "comedy_mean", "crime_mean",
                "documentary_mean", "drama_mean", "fantasy_mean",
                "film_noir_mean", "horror_mean", "musical_mean",
                "mystery_mean", "romance_mean", "sci_fi_mean",
                "thriller_mean", "war_mean", "western_mean")
# Removed "occupation", "item_imdb_mature_rating", "age_band" and genres

train_rf <- train_set[, train_cols]
test_rf <- test_set[, train_cols]
test_rf[is.na(test_rf$user_sd_rating), "user_sd_rating"] <- 0

rf <- randomForest(formula=rating~., 
                   data=train_rf, 
                   importance=TRUE, 
                   xtest=test_rf[, train_cols[2:length(train_cols)]], 
                   ntree=100)

# Calculate RMSE
RMSE(rf$test$predicted, test_set$rating)
       
## Prediction
new_data <- readRDS("AT2_test_STUDENT.rds")

# Feature engineering on test set using trained amounts
# Merge user level statistics
user_stats <- train_set %>% 
  select(user_id, action_mean:user_age) %>% 
  group_by(user_id) %>% 
  unique() %>% 
  ungroup()

new_data <- left_join(new_data, user_stats, by="user_id")
new_data$user_id <- as.factor(new_data$user_id)

# Add review rank
new_data <- new_data %>% 
  group_by(movie_title) %>% 
  mutate(review_rank=rank(timestamp)) %>% 
  ungroup()

# Check missing data
sapply(new_data, function(x) sum(is.na(x)))

# Get item level stats from training data
item_stats <- train_set %>% 
  select(item_id, item_imdb_rating_of_ten:item_imdb_top_1000_voters_average) %>% 
  group_by(item_id) %>% 
  unique() %>% 
  ungroup()

# Isolate missing imdb data
new_item_stats <- new_data %>% 
  select(item_id, item_imdb_rating_of_ten:item_imdb_top_1000_voters_average) %>% 
  group_by(item_id) %>% 
  unique() %>% 
  ungroup()

missing_items <- new_item_stats %>% 
  filter(is.na(item_imdb_rating_of_ten)) %>% 
  select(item_id)

# Missing values don't existing in training set - use imputation instead
sum(missing_items$item_id %in% item_stats$item_id)

for(n in names(new_item_stats)[2:ncol(new_item_stats)]){
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

# Random Forest Predictions
new_data$rating = predict(rf, newdata=new_data)

new_data$user_item <- paste(new_data$user_id, new_data$item_id, sep="_")

submission_file <- new_data %>% 
  select(rating, user_item) 

write.csv(submission_file, "rf_submission.csv", row.names=FALSE)

       
# Variable importance plot 

# Extract data from default plot and create new one
imp <- varImpPlot(rf)
imp1 <- data.frame(variables=rownames(imp), values=imp[,1])
imp2 <- data.frame(variables=rownames(imp), values=imp[,2])
rownames(imp1) <- NULL
rownames(imp2) <- NULL

# ggplot version
p1 <- imp1 %>% 
  ggplot(aes(x=reorder(variables, values), weight=values)) + 
  geom_bar(fill="blue") +
  coord_flip() + 
  theme(legend.position="none") + 
  labs(title="MeanDecreaseAccuracy", 
       subtitle="For Random Forest",
       x="",
       y="")

p2 <- imp2 %>% 
  ggplot(aes(x=reorder(variables, values), weight=values)) + 
  geom_bar(fill="blue") +
  coord_flip() + 
  theme(legend.position="none") + 
  labs(title="MeanDecreaseGini", 
       subtitle="For Random Forest",
       x="",
       y="")

grid.arrange(p1, p2, ncol=2, nrow=1)


# Gradient Boosting
# Data formatting
train_gbm <- train_rf 
train_gbm$rating <- as.numeric(train_gbm$rating)
train_gbm$age <- as.numeric(train_gbm$age)
train_gbm$item_imdb_length <- as.numeric(train_gbm$item_imdb_length)
train_gbm$item_imdb_staff_votes <- as.numeric(train_gbm$item_imdb_staff_votes)
train_gbm$item_imdb_top_1000_voters_votes <- as.numeric(train_gbm$item_imdb_top_1000_voters_votes)
train_gbm$user_count <- as.numeric(train_gbm$user_count)

# GBM hyper parameters
gbm_depth = 5 
gbm_n_min = 15 
gbm_shrinkage=0.01 
cores_num = detectCores() - 1 
gbm_cv_folds=5 
num_trees = 200

# Train Model
gbm_fit = gbm(train_gbm$rating~.,
              data=train_gbm[, -1],
              distribution='gaussian', 
              n.trees=num_trees,
              interaction.depth= gbm_depth,
              n.minobsinnode = gbm_n_min, 
              shrinkage=gbm_shrinkage, 
              cv.folds=gbm_cv_folds,
              verbose = FALSE, 
              n.cores = cores_num)
summary(gbm_fit)

# Test Set Predictions
pred <- predict(gbm_fit, n.trees = gbm_fit$n.trees, test_set)
RMSE(pred, test_set$rating)


# Out of Sample Predictions and Submission
new_data$rating_gbm <- predict(gbm_fit, n.tree=gbm_fit$n.trees, new_data)

submission_file <- new_data %>% 
  select(rating_gbm, user_item) %>% 
  rename(rating=rating_gbm)

write.csv(submission_file, "gb_submission.csv", row.names=FALSE)
       
       
# Random Forest tuned with Caret/Ranger
# Set up parallel
cluster = makeCluster(detectCores()-1)
registerDoParallel(cluster)


# Format Data
train_caret <- train_set
train_caret <- train_caret[, train_cols]
train_caret$rating <- as.numeric(train_caret$rating)
train_caret$age <- as.numeric(train_caret$age)
train_caret$item_imdb_length <- as.numeric(train_caret$item_imdb_length)
train_caret$item_imdb_staff_votes <- as.numeric(train_caret$item_imdb_staff_votes)
train_caret$item_imdb_top_1000_voters_votes <- as.numeric(train_caret$item_imdb_top_1000_voters_votes)
train_caret$user_count <- as.numeric(train_caret$user_count)

# Control variables
control <- trainControl(method="cv",
                        number=5,
                        allowParallel=TRUE)


# Train model
rf_caret <- train(y=train_caret$rating, 
                  x=train_caret[,-1],
                  method="ranger",
                  trControl=control,
                  verbose=FALSE,
                  metric="RMSE")


# Format Test Set
test_caret <- test_set
test_caret <- test_caret[, train_cols]
test_caret$rating <- as.numeric(test_caret$rating)
test_caret$age <- as.numeric(test_caret$age)
test_caret$item_imdb_length <- as.numeric(test_caret$item_imdb_length)
test_caret$item_imdb_staff_votes <- as.numeric(test_caret$item_imdb_staff_votes)
test_caret$item_imdb_top_1000_voters_votes <- as.numeric(test_caret$item_imdb_top_1000_voters_votes)
test_caret$user_count <- as.numeric(test_caret$user_count)
test_caret[is.na(test_caret$user_sd_rating), "user_sd_rating"] <- 0


# Predictions and RMSE
caret_pred <- predict(rf_caret, newdata=test_caret[,-1])
RMSE(test_set$rating, caret_pred)


# Submission file
new_data$rating = predict(rf_caret, newdata=new_data)
new_data$user_item <- paste(new_data$user_id, new_data$item_id, sep="_")

submission_file <- new_data %>% 
  select(rating, user_item) 

# check for missed predictions 
nrow(submission_file)
nrow(na.omit(submission_file))

write.csv(submission_file, "rf_caret_submission.csv", row.names=FALSE)

