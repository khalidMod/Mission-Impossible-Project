Martin 6/5:
#### Missing data: 
There's NA's for a bunch of items:
* Release date: only 9 observations with unknown titles - should remove
* Unknown category: only one movie with single rating after removing the above - should remove
* Item_imdb_rating_of_ten: 1665 entries, includes big movies like Titanic, Good Will Hunting, etc. Some of these have hundreds of ratings so should keep. Imputation (e.g. mean)?
The following categories are similar to rating of ten:
  * item_imdb_count_ratings
  * item_imdb_length 
  * item_imdb_top_1000_voters_votes 
  * item_imdb_top_1000_voters_average
* A number of user/gender grouped features are also based on the above and should be recalculated if we impute values 
* item_imdb_staff_votes, item_imdb_staff_average: 4366 entries, similar treatment as above

#### Feature Ideas:
* Base line average vs Movie Average vs User Average (e.g. are some people harsher critics than others)
* Days since first review for user (has user become more/less critical over time)
* Days since first review of the movie (fans tend to watch earlier)
* Number of previous reviews of movie (contrarian or crowd follower?)
* Correlation/cosine between movies/users 
* Number of reviews by user per genre (user preference for particular type of movie?)

On a simple linear regression I tested and their significance:
* Number of review by user: *** 
* User's average rating: ***
* Time since user's first review: ***
* Number of previous reviews on movie: not significant (pval 0.22)
* Days since release: ** 
Haven't tested correlation between movies/users or per genre splits
