
library(tidyverse)
library(ggplot2)
library(corrplot)
library(caret)
library(randomForest)
library(xgboost)
library(parallel)
library(doParallel)
library(gbm)

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
# Recalc after train/test split to avoid leakage


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
    select(all_of(cols)) %>% 
    unique() %>% 
    ungroup()
  
  combined_ratings <- rbind(old_ratings, select(new_ratings, all_of(cols)))
  
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
                "user_sd_rating", "user_age", "review_rank")
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
errors <- ((rf$test$predicted - test_rf$rating)^2)
RMSE <- sqrt(sum(errors)/length(test_rf$rating))
RMSE 
       
       
# Variable importance plot 
library(gridExtra)

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
gbm_depth = 5 
gbm_n_min = 15 
gbm_shrinkage=0.01 
cores_num = detectCores() - 1 
gbm_cv_folds=5 
num_trees = 200

gbm_fit = gbm(train_rf$rating~.,
              data=train_rf[, -1],
              distribution='gaussian', 
              n.trees=num_trees,
              interaction.depth= gbm_depth,
              n.minobsinnode = gbm_n_min, 
              shrinkage=gbm_shrinkage, 
              cv.folds=gbm_cv_folds,
              verbose = FALSE, 
              n.cores = cores_num)
summary(gbm_fit)

# Predictions and RMSE
pred <- predict(gbm_fit, n.trees = gbm_fit$n.trees, test_set)
RMSE(pred, test_set$rating)
