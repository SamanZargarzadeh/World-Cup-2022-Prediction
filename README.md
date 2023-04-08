# World Cup 2022 Prediction
Utilizing web scraping techniques to extract data from various websites, creating a soccer dataset for national teams, and implementing machine learning algorithms, in both Python and Weka, to predict World Cup 2022 results with the help of feature engineering.


# Table of Contents
- [World Cup 2022 Prediction](#world-cup-2022-prediction)
  - [Introduction](#introduction)
    - [The 2022 FIFA World Cup](#the-2022-fifa-world-cup)
    - [Why Are We Proposing This Topic?](#why-are-we-proposing-this-topic)
  - [Data Overview](#data-overview)
  - [Feature Engineering](#feature-engineering)
  - [Exploratory Analysis](#exploratory-analysis)
  - [Machine Learning](#machine-learning)
    - [K-nearest-neighbors](#k-nearest-neighbors)
    - [Random forest](#random-forest)
    - [Logistic Regression](#Logistic-Regression)
  - [Final Results](#final-results--group-stage)



## Introduction
Every four years, the sporting world comes together to watch one of the biggest tournaments in the world. Featuring billions of fans, hundreds of players, and 32 nations, the world cup is undoubtedly a spectacle of enormous proportions, perhaps the biggest the world sees. With the pride of fans, the legacies of players, and the reputations of nations hanging in the balance, the legend that is the world cup is impactful in more ways than just status. Over the four-week tournament, FIFA sees an inflow of billions in advertising revenue, millions of clicks in web traffic, and countless views in-stadium and on television. The scale of the world cup is simply unmatched. 

As one of the most trending topics in recent months, we have chosen to build a machine learning prediction model to forecast the outcomes of world cup matches throughout the group stage, knockout rounds, and of course, the final. Using data combined from three sources: historical match data, FIFA rankings data, and national team ratings in EA’s FIFA franchise, we will train, test, and evaluate three machine learning models head-to-head-to-head to predict matches winners and tournament outcomes for all 64 matches in the 2022 FIFA World Cup. 


### The 2022 FIFA World Cup

The 2022 World Cup will feature 32 national team who play a combined 64 matches. Starting with the group stage, each team is partitioned off into a group of four teams, where each squad will play each other once. Winners in the group stage are awarded 3 points, teams that draw are given a single point each, and the loser is awarded nothing. The two teams with the highest point total advance to the knockout round. In the knockout round, the 16 advancing teams will be placed in a standard tournament bracket. From there, things are simple. The winner advances, and the loser goes home. The winner of all games in this grueling 4-week journey is awarded the prestigious World Cup, and crowned world champions.


### Why Are We Proposing This Topic?
We are attempting to predict the match results of the 2022 FIFA World Cup for a few reasons. Firstly, it is a fun exercise in practice for the members of our team because we are all hardcore soccer fans. In addition to rooting for our favourite teams, and cheering on the best players, we have an additional dimension of involvement in the World Cup, and that makes the tournament that much more exciting for us. In addition, we are all really excited to flex our newly acquired machine learning skills to predict the outcome of the matches in the tournament. We are anticipating applying the concepts and skills we have learned in class to model a real-world scenario that is trending around the world. We want to see the success of our model come to fruition in a way that invigorates our hobbies outside of school. This project is a perfect way to accomplish that. Finally, we would like to evaluate whether it would be feasible to make money from betting on these matches through various sports betting websites. If our models are successful and accurate, it may be possible for us to find an edge in a sports book and make some well deserve cash. Our expectations though are not too high, as we have chalked that last point up to wishful thinking.


## Data Overview
Looking through various official and unofficial data sources, we were unable to locate a complete dataset of historical soccer match data that contained statistics of the matches that were played. Therefore, we had to investigate building our own data. After exploring the possibility of web scraping and manual data manipulation, we finally set on combining three different data sources that were obtained from Kaggle. This combined dataset represented a solid frame of what we believed were good predictor columns for forecasting the outcome of soccer games based on our domain knowledge. In terms of data integrity, these three datasets were verified of their accuracy by randomly sampling 10 records from each dataset and cross checking them with different sources to confirm their validity.
Our base dataset was historical match data from previous World Cup and Euro Cup qualifying matches and tournament games. Although we had access to all international football matches spanning back a few decades, we chose to filter our data to contain only records from the aforementioned tournaments due to their similarity in playing style, importance, and match conditions to the games that were to be played in this year’s world cup. Each record in the dataset was a match, and it contained identifying information of the teams playing, and well as their match statistics categorized in status by home and away. This dataset was quite clean; however, it did contain some NA values that were eventually dropped. The number of excluded records was less that 5% of the total dataset and were manually verified to be inconsequential.

<img width="477" alt="image" src="https://user-images.githubusercontent.com/88157400/230743427-474759ff-8fcd-49ab-9626-c6e2e441329f.png">
 
Figure 1 - A brief look at the base dataset we started with

After finalizing the base dataset, we worked to obtain more data that could help our model learn better. One of the best indicators of current team performance is the official men’s national team rankings that is published by FIFA. The world rankings are a ranking system based on game results, with the most successful teams being ranked highest. These rankings are considered the most complete and accurate picture of a current national team’s standing in relation to competing nations. Since FIFA rankings are updated 4-5 times yearly, we chose to select rankings from the first and the last ranking of each team for each year. After creating a table of all national teams and their beginning and end-of-year rankings, we joined it with the initial dataset twice, once each for the home and away team.  

<img width="125" alt="image" src="https://user-images.githubusercontent.com/88157400/230743477-81026510-7ab9-4f94-8a2e-cc36ff272328.png">


After finalizing the base dataset, we worked to obtain more data that could help our model learn better. One of the best indicators of current team performance is the official men’s national team rankings that is published by FIFA. The world rankings are a ranking system based on game results, with the most successful teams being ranked highest. These rankings are considered the most complete and accurate picture of a current national team’s standing in relation to competing nations. Since FIFA rankings are updated 4-5 times yearly, we chose to select rankings from the first and the last ranking of each team for each year. After creating a table of all national teams and their beginning and end-of-year rankings, we joined it with the initial dataset twice, once each for the home and away team.  
Another reliable metric that we added to our dataset was EA sports’ national team ratings from their EA sports video game franchise. EA’s ratings are derived from the real-world performance of the players and teams in their game. Although their rating methods are generally unknown, they employ an entire team of sports enthusiasts and data specialists, who generate player and team ratings from a variety of different sources. Their ratings are generally considered to be very accurate to real world performance. After obtaining a dataset of the national team ratings from 2004-2022, we added it to our master dataset. All data in the master dataset prior to 2004 was dropped, leaving us with approximately 17 years of highly relevant data. 
The next step in building our dataset was encoding the match results from categorical strings to their binary representation. Luckily, one columns of our dataset had match outcomes for the home team, which were categorized as either ‘win’, ‘loss’, or ‘draw’. Although it was understood that “1” would represent a win and “0” denoted a loss, we still had to account for the draw outcome. The way we accounted for this was by encoding draw and loss with the same number, “0”. Since we are only trying to predict wins, adding draws to the loss categorization made sense, since our new classification would technically become win/ not win for the home team. 


## Feature Engineering
With the dataset seemingly complete, we went ahead and trained our ML algorithms only to find out that the classification accuracy was 99%. It was obvious that we did something wrong. After further inspection, we realized that data leakage was the cause of our issue. The matches dataset contained statistics like data like goals, possession stats, and shots, and because of that, the algorithm badly overfit our data. Although we could have eliminated those columns from the dataset, we decided that they were too valuable as predictors to be let go. Instead, we thought outside the box and decided to calculate rolling averages of a team’s past game statistics to use as predictors for their next games. Ideally, a 10-game rolling average of game statistics would have worked well for our model, but that was not possible for two reasons: 1) our matches dataset only contained years in the date columns, without days and months, and 2) the frequency of world cup and Euro games was sparse on off-years. Consequently, we settled on a 2-year rolling average of statistics. This was a nice compromise, since 2-years of prior data would still be relevant to the current performance of a team, and it fit perfectly with our data. The “final” dataset came in at 1476 rows, with 360 rows that contained NA values. After careful consideration, we decided to drop all records that contained NA values, and then re-introduce them and measure their impact in subsequent iterations of model training. After this change, the final, final dataset stood at 1116 rows and 28 columns. 

'YEAR'	Year of the match
'HNAME'	Home team name
'ANAME'	Away team name
'H_RATING',	Home team overall rating
'H_BOY_RANK',	Home team beginning of year ranking
'A_RATING',	Away team rating
'A_BOY_RANK',	Away team beginning of year ranking
'HAVG_GOALS',	2-year rolling average of goals scored per game by home team
'HAVG_POSSESION',	2-year rolling average of possession % per game by home team
'HAVG_SHOTSONTARGET',	2-year rolling average of shots on target per game by home team
'HAVG_SHOTS',	2-year rolling average of total shots per game by home team
'HAVG_YELLOWCARDS',	2-year rolling average of yellow cards per game by home team
'HAVG_REDCARDS',	2-year rolling average of red cards per game by home team
'HAVG_FOULS',	2-year rolling average of total fouls per game by home team
'HAVG_RATING',	2-year rolling average of average rating of home team
'HAVG_EOY_RANK',	2-year rolling average of end of year ranking of home team
'HAVG_BOY_RANK',	2-year rolling average of beginning of year ranking of home team
'AAVG_GOALS',	2-year rolling average of goals scored per game by away team
'AAVG_POSSESION',	2-year rolling average of possession % per game by away team
'AAVG_SHOTSONTARGET',	2-year rolling average of shots on target per game by away team
'AAVG_SHOTS',	2-year rolling average of total shots per game by away team
'AAVG_YELLOWCARDS',	2-year rolling average of yellow cards per game by away team
'AAVG_REDCARDS',	2-year rolling average of red cards per game by away team
'AAVG_FOULS',	2-year rolling average of total fouls per game by away team
'AAVG_RATING',	2-year rolling average of average rating of away team
'AAVG_EOY_RANK',	2-year rolling average of end of year ranking of away team
'AAVG_BOY_RANK',	2-year rolling average of beginning of year ranking of away team
'RESULTS'	Result of the match (1 - Home team win, 0 - Home team draw/loss)



## Exploratory Analysis
 
<img width="468" alt="image" src="https://user-images.githubusercontent.com/88157400/230743503-0f12a84a-ee2a-4733-b5f8-7730ec5bed5d.png">

Figure 4 - Team ratings plotted by team, with the size of each line indicating a 2-year rolling average goals per game per team
 
With the dataset containing so much different data on different teams in different matches in different records, it was sometimes hard to conceptualize the patterns in the underlying dataset. Luckily, with the help of data visualization, Tableau, and Python, we were able to derive some very useful insights into our data. As shown above, we see a general correlation between team rankings, and team ratings. Considering that the data sources for these two measures were completely data, we can say with confidence that there is no anomolies in the rating and rankings datasets. Figure 4 also shows that there is a steady distributon of average goals, which validates our concerns of data integrity.
Figure 5 below shows a correlation matrix of our predictor variables. As we can see, there many of highly correlated variable, and many variables with zero correlation. It is nice to be able to visualize the correlation matrix because it tells us that we will need to conduct feature selection in order to make an accurate model, and reduce the underlying noise in the dataset.

<img width="463" alt="image" src="https://user-images.githubusercontent.com/88157400/230743512-d40f1a01-ec38-4830-ab92-87263ea7024c.png">

Figure 5 - Correlation matrix of the predictor variables

 
<img width="241" alt="image" src="https://user-images.githubusercontent.com/88157400/230743514-9699c77e-5e43-4073-ae51-a90ec8b3f68e.png">
Figure 6 - Distribution of national team ratings

With figures 6 and 7, we can see that the data is mostly true to scale. Therefore, we may not have to normalize the data because it is already in order. Even though the scale of the data is normal, there may still be an advantage in terms of model improve if the data is in fact scale. It is only by conducting these sorts of pre-processing steps, and experimenting with data and model tweaks, that will really determine whether scaling and normalization helps improve the predictive capability of the models. In addition, by visualizing the data through diagrams like the histogram in figure 6 and the boxplot in figure 7, we can visually verify that our dataset doesn’t contain any bad values in it. Bad values may throw off the accuracy of a model, cause variation in the results, and may cause overfitting of the training data.

<img width="242" alt="image" src="https://user-images.githubusercontent.com/88157400/230743521-1e12842b-869b-4616-9ee9-9791477bc167.png">
Figure 7 - Highest ranked are #1, lowest are in the 80 range

## Machine Learning 
After finalizing our dataset, the next step was to split our dataset into test and training samples. Since our data was time series, there was serious consideration to separate the test and train samples based off date. For example, our training sample would contain all games from before 2020, and our test sample would contain those from 2020 and onwards. The logic was that data leakage would occur since past games may have an impact on subsequent games in terms of winning/cold streaks, improved chemistry between teams, and other factors that could influence the model. We, however, decided against splitting based on date, due to the independent nature of each record, and the time constraint and scope of the project. Instead, our data was split randomly into a test and training sample with an 80/20 allocation. 

Since our target classes were binary, we decided to compare three supervised classification algorithms: 1) K-nearest neighbors, 2) Random Forest, and 3) Logistic regression. To reduce variance in the results, we used k-folds cross-validation on each model. K-folds cross validation is a sampling technique in which the training set is separated into k “banks”, where k-1 banks are combined and trained, and the remaining bank is left as the validation sample. The evaluation score of the first run is recorded, and then k-1 iterations of training the model are processed with each validation bank being used as a validation sample, and the others being used as the training sample. At the end of the k-1 iterations, the evaluation scores are averaged and that becomes the final evaluation score. 
In our case, the evaluation metric that was used to compare the three models was accuracy score. Accuracy score is calculated by dividing the true positives and true negatives with the total dataset count. Baseline accuracy scores of each model were generated by running the data through the algorithms without pre-processing. This baseline accuracy score was evaluated against each iteration of a data processing step in an effort to improve the final model’s predictive ability. 


### K-nearest-neighbors 
K-nearest neighbors is supervised learning algorithm for both classification and regression problems. Simply put, it takes data points, places them into k clusters, calculates the mean of each cluster, and iterates placement of datapoints to different clusters in an effort to decrease the distance between datapoints and their closest cluster mean. 
 
  ![image](https://user-images.githubusercontent.com/88157400/230743543-c88eab99-ff39-4520-9b63-7a5f3b7cc5e1.png)

Figure 9 - Source: https://vitalflux.com/k-nearest-neighbors-explained-with-python-examples/

The reason we chose KNN is because it is a good all-around algorithm for those who are new to machine learning. When first starting our project, we did not have a concrete dataset and we were experimenting with adding more than two classes as our target variable. Because of this uncertainty, we needed to test and learn with a robust algorithm. KNN was exactly that. KNN is simple to understand, it can be used for multiclass classification problems, it makes no assumptions about the underlying dataset’s distribution, and it is easy to optimize. These reasons taken together was why we chose it. 

Our first run of the KNN algorithm involved feeding a raw dataset into the model to establish a benchmark accuracy. The algorithm was benchmarked using the mean accuracy score of 10-folds cross-validation. A code snippet of the run and the resulting accuracy score of the training set is summarized below in figure 10. 

<img width="474" alt="image" src="https://user-images.githubusercontent.com/88157400/230743548-beba51f2-fb2b-4bda-873d-4b0562227ebf.png">
 
Figure 10

Over the 10 runs of the KNN cross validation model, a mean accuracy score of 66.48% was the result. This means that, on average, 66.48% of the classifications were correct. To illustrate this more simply, suppose that the initial class proportions of the training set are 50 wins and 50 losses/draws. Therefore, the KNN model above would have correctly classified 66% * 100 = 66 of the match results in the training set. 

In thinking of improvements to our model, we realized that the team names themselves were not accounted for. We looked at experimenting with attribute expansion. When teams are playing head-to-head, at times, one team may have a particularly bad record against another team for an unexplained reason. To many, this may seem like a “curse” that one team has against another. We wanted to model this and control for it in our data. Therefore, we transformed team names to dummy variables in a process called ‘one hot encoding’. After encoding the home and away teams to dummy variables, the number of our columns increased to 170. The output of the mean cross-validation score for the KNN algorithm was 0.6626, and when comparing it to the benchmark score, we can see that the score went down instead of improving. Although we were surprised at the outcome, it was clear that we needed to revert to the previous dataset without the encoded team columns. 

<img width="469" alt="image" src="https://user-images.githubusercontent.com/88157400/230743555-fb769230-5416-4af5-baa4-8120ad76db86.png">

Figure 11 - The accuracy score experienced a surprising drop after one hot encoding
Next, we attempted to improve accuracy by exposing the model to more data. As mentioned in the data section of the report, we initially had a dataset shape of 1476 rows, which was then reduced to 1116 after dropping the rows that contained NA values. Instead of dropping those NA values, we decided to impute the blanks by replacing them with the means of the columns. 

<img width="474" alt="image" src="https://user-images.githubusercontent.com/88157400/230743563-80a7cd7a-80f5-41c0-8c9d-6bf457236b8d.png">

Figure 12 - This code snippet transforms NA values to their columns means

<img width="468" alt="image" src="https://user-images.githubusercontent.com/88157400/230743572-2f8d6d31-a563-4b6a-a761-43651059692a.png">

Figure 13 - Another drop in accuracy score
Again, we see that the model accuracy actually declined instead of improving. Obviously, attribute expansion did not work for our model, so we tried one final thing: normalizing the data. Since KNN is not suitable for large dimensional data, it is recommended that the data is normalized in values between 0 and 1. By transforming the data to the same scale, we were hoping that a rebalancing of feature importance would occur, and eventually improve the model. We were wrong again. 
 
<img width="468" alt="image" src="https://user-images.githubusercontent.com/88157400/230743578-d6f58bd5-60e5-4721-ab68-e851bd5d49ba.png">

Figure 14 - Normalizing the data in Python

<img width="467" alt="image" src="https://user-images.githubusercontent.com/88157400/230743581-5b9a2492-b1d3-40f5-a823-2d26dd54844f.png">

Figure 15 - Trending in the opposite direction
An accuracy of 63.89% was the worst score so far. We reverted the dataset back to the initial state, and instead tried something different. Instead of tweaking the data first, we decided to tweak the algorithm. By default, KNN’s parameters have 5 neighbors. This however can be changed to optimize the model and potentially improve accuracy. We experimented with changing the number of neighbors by iterating the k-values from 1 to 40. After plotting the accuracy scores for each iteration, we see that the optimal number of neighbors is 19, yeilding a max score of 69.73%.

<img width="430" alt="image" src="https://user-images.githubusercontent.com/88157400/230743588-3e3ae5b8-b2ac-491f-b2f9-0464e1527124.png">

Figure 16 - Our first model improvement
Finally, after many failed attempts, we see an increase in the accuracy score. In an effort to further improve the accuracy score, we tried to implement all three data preprocessing steps again with k-neighbors equal to 19. 

After many iterations, we were happy report that our accuracy score after changing the k nearest neighbours to 19 and imputing the NA data, we had a final model accuracy of 70.33%. Now, this was put up against the accuracy score of 2 other models. 


### Random forest
Random forest is an ensemble learning algorithm that works with both classification and regression problems. Random forests are essentially a collection of decision trees that run on random subsets of data. Since each decision tree runs independently, Random forests reduce bias and increase accuracy. Radom forests function by averaging the results of all the decision trees it contains, therefore reducing overfitting of the data through its voting mechanism.  Its built-in feature selection method is also very useful, and it can be used to see which attributes are the most impactful in a given dataset.

After using the KNN algorithm in the previous section, we realized that our data had a lot of columns (at 28 columns), and it was hard to differentiate importance between them. Since random forests have feature selection built into it, it was the logical next choice for our use case. To start our analysis of random forests off, we computed the accuracy score of the model without any changes or tweaks to the data and parameters. The benchmark accuracy score was computed as before, namely, it was an average of the 10 accuracy scores of the cross validation run. To compare, it’s initial accuracy score came in at slightly above that of the KNN model.

<img width="461" alt="image" src="https://user-images.githubusercontent.com/88157400/230743659-1cbe989c-54a6-45a1-83ed-9d7c2634ad51.png">
 
Figure 18 - Another benchmark accuracy run

In order to accurately evaluate this model against the previous KNN model, we ran the same data preprocessing steps as in the prior section. The preprocessing steps were: one hot encoding the team names, Imputing NA values, and finally, normalizing the data. In isolation, these three steps did not yield much improvement. Only attribute expansion by means of one hot encoding made a difference to the accuracy score, and that too only by approximately 1%. It was also interesting to note that the accuracy score remained exactly the same after normalization. After some further research, it was quickly apparent that scaling is not necessary for tree-base models because they are not affected by the absolute values taken by features. 

Misleading data and noise in a model may sometimes lead to decreased accuracy in that model. Random Forests have built in ways of determining feature importance and the model parameters can be changed to select n number of top features in the model. As our dataset currently stands, there are 24 predictor columns that we are using to make predictions. Some variables have much more predictive influence than others, and it is clear once you see a visualization of it. 

<img width="468" alt="image" src="https://user-images.githubusercontent.com/88157400/230743671-d107b140-ca2e-48fa-b5a9-642b1a710334.png">

Figure 20 - Rankings of feature importance of the predictor variables in the model

As we did with the KNN model, the next steps involve tweaking model parameters to come up with the most optimal model for the data. Instead of just tweaking one parameter like we did in the last model, we will be using randomized parameter search to come up with the most optimal parameters for our model. As the name suggests, the randomized parameter search tries random combinations of parameters and return the most optimal set of parameters for the training set. Using randomized parameter search saves computational resources since attempting every permutation and combination of parameters would be very resource intensive. In contrast, the randomized parameter search that we ran only took 5 minutes to return optimal parameters. 

<img width="293" alt="image" src="https://user-images.githubusercontent.com/88157400/230743687-17d87e13-91c5-4add-b461-3092c09ee92d.png">
Figure 21 - The optimized parameters after random parameter search

<img width="468" alt="image" src="https://user-images.githubusercontent.com/88157400/230743697-3f48d04c-ccca-4bc0-a34f-772d74eb1a6e.png">

Figure 22 - Code snippet for randomize grid search. Adapted from: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74


Although appearing to be promising, the final accuracy score with tuned parameters was only 68.95%. After such extensive tuning and manipulation, our team expected a higher accuracy score, however, that was not to be had. Curious for more, our group decided to turn to one final algorithm to make the comparison more exciting. 

### Logistic Regression
Logistic regression is a binary classification algorithm that takes a function and maps it to a value between 0 and 1. Visually, the logistic function is shaped like a slanted ‘S’ curve. The tails are effectively polarized – one goes towards positive infinity and the other negative infinity. After running the algorithm, if the predicted y-value is above 0.5, the output will be classified as 1, and if the predicted y-value is less than 0.5, the output will be classified as 0. Logistic regression is one of the simplest and commonly used machine learning algorithms. It is easily interpretable and is computationally efficient. 

As with the prior two sections, we ran the logistic regression algorithm with the unprocessed data to establish a baseline accuracy score to evaluate our data and model tweaks against. The logistics regression cross validation accuracy score benchmark came in the highest among all three models we evaluated at 70.06%. This was enough to beat all tweaked and tuned random forest models, and only fell slightly short to the highest accuracy score achieved by the KNN model. My team were hopeful that we could improve the accuracy score of this model since it looked so optimistic right off the bat.

Following standard preprocessing procedure, we again ran the three processing steps and recorded their accuracies.  As expected, we were able to increase the accuracy score of the logistic regression model by imputing NA values in our dataset. With the KNN model responding kindly to imputation of missing values as well, it was quite clear to us that increasing the size of the training set had a positive impact on the learning ability of the models. Curious to say the least, we want to explore imputation further, perhaps by using an alternative method of imputation. This however would have to wait until the final model evaluation phase.


One final processing tweak that our team wanted to explore was feature selection. Although there are different types of feature selection methods such as backwards selection, forwards selection, and step selection, we chose to take data generated from the run of our random forest analysis and consider selection of a combination of the top variables ranked by random forest algorithm. 

We ran a modified version of forward selection by starting with the most impactful attribute, and then subsequently added additional attributes until we observed an optimal result. Note that the model run was combined with imputation of NA values since that yielded the best results as seen above in figure 23.  The visualized results of the feature selection were plotted on the graph below. We achieved a final model accuracy score of 70.59%. The highest score achieved in all runs!

 <img width="426" alt="image" src="https://user-images.githubusercontent.com/88157400/230743710-6c1a6762-de16-4781-b310-c1c606121a9e.png">

Figure 24 - The accuracy score of 0.70593 was maximized at the first 12 variables

Happy to end things off with the highest accuracy score for any model, our final model was logistic regression with the top 12 features selected, and the NA columns in the original dataset imputed with mean values. It was now time to test our trained model on the test set.


## From Training to Testing
After we had evaluated between the three algorithms, pre-processed data, and tweaked the model parameters, it was time to tie everything together, and evaluate the model on unseen data. As mentioned previously, the model that had the highest accuracy score was the model that we would select for the test data. That model was logistic regression with NA values in the dataset imputed with their means, and with only the top 12 features selected.

<img width="405" alt="image" src="https://user-images.githubusercontent.com/88157400/230743751-8916cd9b-105d-42ce-a1dc-4a127cb05e43.png">

Figure 26 - We achieve an accuracy score of 69.19%

<img width="391" alt="image" src="https://user-images.githubusercontent.com/88157400/230743756-b02405d2-a980-4e61-b407-9f08888575d6.png">

Figure 27 - The final confusion matrix shows false and true positives and negatives

	Our results on the testing dataset were quite good. With an accuracy score of 69.19%, this lined up quite closely with the 70.59% maximum accuracy score we achieve with the training data. This means that our training data was not overfit, and with unseen data, that proved true. Upon further analysis of the confusion matrix, we see that the precision score, which is calculated by dividing true positives by all positives, is (55/55+23 = 70.51%). The precision is a ratio of correctly predictive positive observations to the total positive values. For the case of predicting soccer outcomes with our first implementation of a machine learning model, our group is quite pleased with our result. Sports matches experience so much variability and randomness, that is why even with a precision score that may be unacceptable in many other contexts, we can take pride in knowing that our predictions were quite good. Now, to wrap things all up, we decided to use our final model and apply it to predicting the world cup outcomes and winners.



## Final Results – Group Stage
<img width="458" alt="image" src="https://user-images.githubusercontent.com/88157400/230743765-3a6b7f5f-5aa2-4820-8606-940034685dd0.png">

Figure 28
Applying our model to predict the World Cup outcomes gives us new teams to cheer for. We ran the training model to simulate the results of each match, and after 10 runs, we averaged out the number of wins in each of the 3 games each team played. The table above shows us who advanced to the knockout stage, and which teams failed to qualify. Although we see that the model predicted a lot of the pre-tournament favorites to advance, we did see a few upsets occur. Namely, we see that Spain, Uruguay, Croatia were upset by Costa Rica, South Korea, and Morocco in our model. As there are always surprising results in these kinds of tournaments, this seems in line with what could happen during the actual tournament.
 
<img width="474" alt="image" src="https://user-images.githubusercontent.com/88157400/230743771-44e4bf41-93f1-492d-b005-93f093fe3de8.png">
<img width="474" alt="image" src="https://user-images.githubusercontent.com/88157400/230743779-057cdcd4-23c9-435c-bf47-42b3092dec77.png">

 
Figure 29

Ranked 7th in betting odds during the pre-tournament phase, Portugal pull off the upset against number 1 ranked Brazil to take the 2022 World Cup. This result would not shock anyone, and it is sufficient to say that we are happy with our predictions, and we now all have a new favorite team to root for. Go Portugal!

Colleague in this project: Aneesh Riar
