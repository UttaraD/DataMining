library(dummies)
library(forecast)
library(reshape)
library(leaps)
library(FNN)
library(rpart)
library(rpart.plot)
library(glmulti)
library(gains)
library(neuralnet)
library(FNN)
library(scales)
library(RColorBrewer)
library(dplyr)
library(tidyr)

#Load data
books.main <- read.csv("bestsellers with categories.csv", header = TRUE)  # load data
books.df <-books.main
summary(books.df)
dim(books.df)
#Creating additional column Words_Name that contains the number of words used to name the book
books.df$Words_Name <- sapply(strsplit(books.df$Name, " "), length)

#Merge duplicate authors into one
books.df$Author <- ifelse(books.df$Author=='George R. R. Martin', 'George R.R. Martin', books.df$Author)
books.df$Author <- ifelse(books.df$Author=='J. K. Rowling', 'J.K. Rowling', books.df$Author)
#Creating additional column Popularity that contains levels (Low, Medium, High) based on the number of times an author has appeared in the best sellers list
plot(table(books.df$Author)[books.df$Author])
books.df$count <- as.numeric(table(books.df$Author)[books.df$Author])
books.df$popularity <- ifelse(books.df$count==1,'Low',ifelse(books.df$count==2,'Medium','High'))

#Replaced Books with price=0 with mean of price
books.df$Price <- ifelse(books.df$Price==0, mean(books.df$Price), books.df$Price)
books.df[,c(7,10)] <- lapply(books.df[,c(7,10)] , factor)
summary(books.df)
books_new.df <- books.df[,c(-1,-2,-6,-9)]
summary(books_new.df)

round(cor(books_new.df[,c(1,2,3,5)]),2)
#Negative correlation between Words_Name & Reviews
#Negative correlation between Reviews & Price
#Negative correlation between User.Rating & Price
#User Rating and Reviews have a negative correlation with Price.
#Price and Words_Name have a negative correlation with Reviews.

#Heatmap to visualize the correlation
plot(books_new.df[,c(1,2,3,5)])
plot(books_new.df$Reviews~books_new.df$Words_Name)
plot(books_new.df$Reviews~books_new.df$Price)
plot(books_new.df$Price~books_new.df$User.Rating)

##Partitioning the data:
set.seed(1)
test.rows <- sample(c(1:dim(books_new.df)[1]),50)
test.df <- books_new.df[test.rows,]
dim(test.df)
train.rows <- sample(setdiff(rownames(books_new.df), test.rows),dim(books_new.df[-test.rows,])[1]*0.6)
train.df <- books_new.df[train.rows,]
dim(train.df)
valid.rows <- setdiff(rownames(books.df), union(train.rows, test.rows))
valid.df <- books_new.df[valid.rows,]
dim(valid.df)

#==================================================================#
# stacking Price values for each combination of Year and Genre
mlt <- melt(books.main, id=c("Year", "Genre"), measure=c("Price"))
head(mlt, 5)
# use cast() to reshape data and generate pivot table
GenrevsPrice<- cast(mlt, Year ~ Genre, subset=variable=="Price",
                    mean)
#plot price trend throughout the years for fiction and non-fiction books
plot(GenrevsPrice$Year, GenrevsPrice$Fiction, type = "b", col = 'Blue',
     ylim = c(0,25), xlim = c(2009,2019),
     xlab = 'Year', ylab='Price')
lines(GenrevsPrice$Year, GenrevsPrice$`Non Fiction`, type = "b", col = 'Red')
legend("topleft", inset=c(0,0),
       legend = c("Fiction","Non Fiction"),
       col = c("blue","red"),
       pch = 1, cex = 0.5)

#Run LM on all variables
books.mvr <- lm(Price ~  ., data=train.df)
options(scipen=999) # avoid scientific notation
summary(books.mvr)
books.pred <- predict(books.mvr, newdata = valid.df)
accuracy(books.pred, valid.df$Price)

# Adjusted R sqaure 0.03825, Residual standard error: 11.85
#Important variables
#User.Rating , Genre

names(books_new.df)
search <- regsubsets(Price ~ ., data =train.df, nbest = 1, nvmax = dim(train.df)[2], method = "exhaustive")
sum <- summary(search)
# show models
sum$which
sum$adjr2
names(train.df)

books.lm.step <- step(books.mvr, direction = "backward")
summary(books.lm.step)  # Which variables were dropped? -- Reviews and Words_Name
books.lm.step.pred <- predict(books.lm.step, valid.df)
accuracy(books.lm.step.pred, valid.df$Price)

#final model
booksfinal.mvr<-lm(Price ~  ., data=train.df[,c(-1,-2,-5)])
options(scipen=999) # avoid scientific notation
summary(booksfinal.mvr)
booksfinal.pred <- predict(booksfinal.mvr, newdata = valid.df)
accuracy(booksfinal.pred, valid.df$Price) # RMSE - 8.62617
mean(booksfinal.pred)


#==================================================================#
#Using Knn

summary(books_new.df)
books2.df <- books_new.df
#Making dummies for genre and popularity:
library(dummies)
Genre <- dummy(books2.df$Genre, sep = "_")
popularity <- dummy(books2.df$popularity, sep = "_")
books2.df <- cbind(books2.df[,c(-4,-6)], Genre,popularity)
names(books2.df)[names(books2.df)=="Genre_Non Fiction"] <- "Genre_Non_Fiction"

summary(books2.df)


#Partitioning the data  into 3 parts:
set.seed(1)
test.rows <- sample(c(1:dim(books2.df)[1]),50)
test.df <- books2.df[test.rows,]
train.rows <- sample(setdiff(rownames(books2.df), test.rows),dim(books2.df[-test.rows,])[1]*0.6)
train.df <- books2.df[train.rows,]
valid.rows <- setdiff(rownames(books2.df), union(train.rows, test.rows))
valid.df <- books2.df[valid.rows,]

#Dimensions of the three data partitions: training, validation and testing.
dim(train.df)
dim(valid.df)
dim(test.df)

# initialize normalized training, validation data, assign (temporarily) data frames to originals
train.norm.df <- train.df
valid.norm.df <- valid.df
books2.norm.df <- books2.df
test.norm.df <- test.df

head(train.norm.df)
summary(train.norm.df)

#Normalize the data for kNN

# use preProcess() from the caret package to normalize the variables:
library(lattice)
library(ggplot2)
library(caret)
norm.values <- preProcess(train.df[,-3], method=c("center", "scale"))
head(norm.values)
train.norm.df <- predict(norm.values, train.df[,-3])
head(train.norm.df)
##Similarly scale valid data,bank2 data and the new data
valid.norm.df <- predict(norm.values, valid.df[,-3])
books2.norm.df <- predict(norm.values, books2.df[,-3])
test.norm.df <- predict(norm.values, test.df)
head(books2.norm.df)
summary(books2.norm.df)

# kNN  regression model to predict the Price
# Initial model with k=3
library(FNN)
book_knnreg <- knn.reg(train = train.norm.df, test = valid.norm.df,
                       train.df[, 3], k = 3)

print(book_knnreg)

#Check the accuracy of the kNN regression model through plotting

plot(valid.df[,3], book_knnreg$pred, xlab="Actual price", ylab="Predicted price")

#The regression model with k=3 does not look very efficient. We need to calculate the RMSE and MAE values


#RMSE on Validation set:
sqrt(mean((valid.df[,3] - book_knnreg$pred) ^ 2))# RMSE - 7.63

#Finding a better value of k:
book.accuracy.df <- data.frame(k = seq(1, 20, 1), RMSE = rep(0, 20))
book.accuracy.df

for(i in 1:20) {
  books_knn_chk.pred <- knn.reg(train= train.norm.df, test= valid.norm.df,
                                train.df[,3], k = i)
  book.accuracy.df[i, 2] <- sqrt(mean((valid.df[,3] - books_knn_chk.pred$pred) ^ 2))
  #book.accuracy.df[i, 3] <- mean(abs(valid.df[,2] - books_knn_chk.pred$pred))
}
book.accuracy.df

#Best accuracy is achieved for k=4.As observed from the above developed table.

# Predicting min. number of reviews for new data using the kNN model.

library(FNN)
book_knnreg_prednew <- knn.reg(train = train.norm.df, test = test.norm.df[,-3],
                               train.df[, 3], k = 4)
#mean price for new data
print(book_knnreg_prednew)
mean(book_knnreg_prednew$pred)#14.204
#RMSE
sqrt(mean((valid.df[,3] - book_knnreg_prednew$pred) ^ 2))# RMSE - 12.3983

#For predicting the Price as earlier we shall be using Regression Trees:

library(rpart)
library(rpart.plot)
library(ISLR)
library(dplyr)
library(tree)
library(tibble)

summary(books2.df)
books3.df<-books_new.df[,]
dim(books3.df)

set.seed(1)
test.rows <- sample(c(1:dim(books3.df)[1]),50)
testtree.df <- books3.df[test.rows,]

train.rows <- sample(setdiff(rownames(books3.df), test.rows),dim(books3.df[-test.rows,])[1]*0.6)
traintree.df <- books3.df[train.rows,]

valid.rows <- setdiff(rownames(books3.df), union(train.rows, test.rows))
validtree.df <- books3.df[valid.rows,]


#Dimensions of the three data partitions: training, validation and testing.
dim(traintree.df)
dim(validtree.df)
dim(testtree.df)

#Generating default Regression tree:
tree.books <- rpart(Price ~., data = traintree.df)
options(scipen=999)
summary(tree.books)

#Default Regression Tree structure
rpart.plot(tree.books)

#Importance of different predictors on the outcome variable (Reviews) based on Default Tree:
tree.books$variable.importance

#Using the default Regression Tree for predicting Price

pred_tree=predict(tree.books,newdata=validtree.df)

#Calculating the RMSE and MAE for the default Regression Tree

#RMSE
sqrt(mean((validtree.df$Price-pred_tree)^2)) #RMSE=8.972496

#Pruning the tree based on best cp:
Bookscv.ct <- rpart(Price ~ ., data = traintree.df,
                    cp = 0.00001, minsplit = 10,xval=5)
rpart.plot(Bookscv.ct)
printcp(Bookscv.ct)

Bookspruned.ct <- prune(Bookscv.ct,
                        cp = 0.02517981)
length(Bookspruned.ct$frame$var[Bookspruned.ct$frame$var == "<leaf>"])

rpart.plot(Bookspruned.ct)
pred_bestpr=predict(Bookspruned.ct,newdata=validtree.df)

#RMSE for Tree with best cp:
sqrt(mean((validtree.df$Price-pred_bestpr)^2))#RMSE=8.684689

#Random Forest
library(randomForest)
## random forest
set.seed(500)
Booksrf <- randomForest(Price ~ ., data = traintree.df, ntree = 4,
                        mtry = 4, nodesize = 5, importance = TRUE)

#Plot showing influential variables based on Random Forest:
varImpPlot(Booksrf, type = 1)

pred_rf=predict(Booksrf,newdata=validtree.df)
#RMSE for Random Forest:
sqrt(mean((validtree.df$Price-pred_rf)^2)) #RMSE=9.01801

#Optimizing number of trees:

Booksrf.accuracy.df <- data.frame(k = seq(10, 5000, 100), rmse = rep(0, 50))
Booksrf.accuracy.df

for(i in 1:50) {
  set.seed(500)
  Booksrf <- randomForest(Price ~ ., data = traintree.df, ntree = i,
                          mtry = 4, nodesize = 5, importance = TRUE)
  pred_opt=predict(Booksrf,newdata=validtree.df)
  Booksrf.accuracy.df[i, 2] <- sqrt(mean((validtree.df$Price - pred_opt)^2))

}

Booksrf.accuracy.df

min(Booksrf.accuracy.df$rmse)#6.725443
#least RMSE is achieved for k=3710

#Randomforest with 3710 trees:
library(randomForest)
## random forest
set.seed(500)
Booksrf_best <- randomForest(Price ~ ., data = traintree.df, ntree = 3710,
                             mtry = 4, nodesize = 5, importance = TRUE)

#Plot showing influential variables:
varImpPlot(Booksrf_best, type = 1)

pred_rfbest=predict(Booksrf_best,newdata=validtree.df)
#RMSE for optimized Randome Forest:
sqrt(mean((validtree.df$Price - pred_rfbest)^2))#RMSE - 6.806961

#Predicting for test set based on just Random Forest:
testpred_rf=predict(Booksrf,newdata=testtree.df)
#RMSE for Random Forest:
sqrt(mean((testtree.df$Price-pred_rf)^2))# RMSE - 12.43962

#Average Prediction on test set:
mean(testpred_rf)#13.74839

