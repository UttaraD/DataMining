# Analysis and prediction of the characteristics of Amazon Top 50 Bestselling Books 2009 - 2019 
- Recently Amazon released it’s bestsellers list of books from 2009-2019. I have applied the ML algorithms MLR, KNN and Regression Trees Analyze the factors that affect the price of the bestsellers books and predict the average price
## Dataset
- Amazon Top 50 Bestselling Books 2009 - 2019 (Source - www.kaggle.com)
- The data set contains a total of 550 books written by various authors from 2009-2019 along with price. There are Reviews and User Ratings given to the books. 
## Initial Analysis
- There were duplicate authors because of spacing issue. They were merged into one.
- Author, gives life to a book. But how to derive meaning out of Author? For this i have categorized the author as Low,Medium and high based on the number of times author appeared in best sellers list
- Added a new variable Words_Name thinking that number of words in title of the book could impact on the reader
## Algorithms and Error Metrics
- I have used Multiple Linear regression, KNN and Regression trees algorithms to predict the outcome variable Price. I have calculated the accuracy metrics to choose which model fits in the best to predict the average price of the best selling book in amazon.
- Regression Trees gives lesser RMSE which is more accuracy.
## Conclusion
- Regression trees is a better model which gives a better accuracy. The variables popularity, genre, user rating are the main predictors for price



