
### C3_T3
## Big Trouble in Some Departments
#  I would like for you to do the analysis with the goals of:
# Predicting sales of four different product types: PC, Laptops, Netbooks and Smartphones
# Assessing the impact services reviews and customer reviews have on sales of different product types

# Import the data
library(readr)
df1 <- read_csv("~/Documents/''Nastaran''/Analytics@UTexas/Courses/Course 3/Task 3/productattributes/existingproductattributes2017.csv")
View(df1)


# Quick EDA 
install.packages("funModeling")
library(funModeling)
library(Hmisc)

basic_eda <- function(df1)
{
  glimpse(df1)
  print(status(df1))
  freq(df1) 
  print(profiling_num(df1))
  plot_num(df1)
  describe(df1)
}

basic_eda(df1)

status(df1)

# Some more plotting
ggplot(data = df1) +
  geom_bar(mapping = aes(x = ProductType))

ggplot(data = df1, mapping = aes(x = PositiveServiceReview, colour = ProductType)) +
  geom_freqpoly(binwidth = 100)

ggplot(data = df1, mapping = aes(x = NegativeServiceReview, colour = ProductType)) +
  geom_freqpoly(binwidth = 100)

ggplot(data = df1, mapping = aes(x = Price, colour = ProductType)) +
  geom_freqpoly(binwidth = 100)

ggplot(df1) + 
  geom_bar(mapping = aes(x = ProductType))  

ggplot(data = df1, mapping = aes(x = ProductType, y = Price)) +
  geom_boxplot()


# Dummify the data

library(caret)

newDataFrame <- dummyVars(" ~ .", data = df1)
readyData <- data.frame(predict(newDataFrame, newdata = df1))
View(readyData)

str(readyData)
summary(readyData)

# Remove NAs
readyData <- na.omit(readyData)
readyData$attributeWithMissingData <- NULL

# Drop the columns of the data frame

library(dplyr)

readyData <- select (readyData,-c(BestSellersRank,ProductNum,ProfitMargin))
View(readyData)

# Correlation
corrData <- cor(readyData)
corrData

install.packages("corrplot")
library(corrplot)
corrplot(corrData)


# Linear Combos: to identify features with high Collinearity
comboInfo <- findLinearCombos(readyData)
comboInfo

# Comment:
# It tells us that volume (dependent variable) is highly correlated to 5XReview
# so we should delete one of them which is going to be 5XReview

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

### Linear Regression Modeling

install.packages("mlbench")
library(mlbench)
library(caTools)

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

# Fit linear model (version 1.)
lm1 <-lm(Volume~.,data=readyData)
summary(lm1)

# Comment:
# We see in the summary of this linear regression model that we got 1 for R-squared
# which means we have perfect fitting model (over-fitted)

# Let's omit feature 14 (5X Review), and try the linear model again:

# omit 5XReview
readyData <- select (readyData,-c(x5StarReviews))
View(readyData)

# Fit linear model (version 2.)


#10 fold cross validation
set.seed(532)
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

#train model: LM
mod_lm <- train(Volume~PositiveServiceReview,
                data = train.data,
                method="lm",
                trControl = fitControl)
mod_lm



lm2 <-lm(Volume~.,data=readyData)
summary(lm2)
print(lm2)

# Comment:
# We got 0.93 for R-squared and 0.89 for Adjusted R-squared. This linear model definitely works better.

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

#10 fold cross validation
set.seed(532)
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

mod_lm5 <- train(Volume~PositiveServiceReview+x4StarReviews+ProductTypeGameConsole,
                 data = train.data,
                 method="lm",
                 trControl = fitControl)
mod_lm5


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

### Support Vector Machines (SVM)

# Split the data into training and test set
set.seed(123)
training.samples <- readyData$Volume %>%
  createDataPartition(p = 0.8, list = FALSE)
train.data  <- readyData[training.samples, ]
test.data <- readyData[-training.samples, ]


# SVM (version 1.)

train_control <- trainControl(method="repeatedcv", number=10, repeats=3)

set.seed(123)
svm1 <- train(Volume ~., data = readyData, method = "svmLinear2",
              trControl = train_control,  preProcess = c("center","scale"))
svm1
summary(svm1)
plot(svm1)

# Make predictions on the test data
pred1 <- predict(svm1, newdata = test.data)
plot(pred1)
head(pred1)

# Compute model accuracy rate
mean(pred1 == test.data$Volume)

# Plot the model
plot(svm1)
svm1$bestTune

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
### Random Forest

#import the package
library(randomForest)

# Perform training:
rf_classifier = randomForest(Volume ~ ., data=train.data, ntree=100, mtry=2, importance=TRUE)
summary(rf_classifier)
varImpPlot(rf_classifier)

pred2 <- rf_classifier %>% predict(test.data)
plot(pred2)
head(pred2)
postResample(pred2, test.data$Volume)




#reduce number of folds to 5
set.seed(999)

#5 fold cross validation
fitControl <- trainControl(method = "repeatedcv", number = 5, repeats = 1)

mod_rf2 <- train(Volume~PositiveServiceReview+x4StarReviews+ProductTypeGameConsole,
                 data = train.data,
                 method="rf",
                 trControl = fitControl,
                 preProcess = c("center","scale"),
                 Tunelength = 20)

mod_rf2

#Results on testset
test_results_rf2 <- predict(object = mod_rf2, 
                            newdata = test.data)
postResample(test.data$Volume, test_results_rf2)

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
### Gradient Boosting

library(tidyverse)
library(caret)

install.packages("xgboost")
library(xgboost)

# Fit the model on the training set
set.seed(123)
gb1 <- train(Volume ~., data = train.data, method = "xgbTree",
  trControl = trainControl("cv", number = 10))

gb1
summary(gb1)
plot(gb1)

# Best tuning parameter
gb1$bestTune

# Make predictions on the test data
predicted.classes <- gb1 %>% predict(test.data)
head(predicted.classes)

# Compute model prediction accuracy rate
mean(predicted.classes)
mean(test.data$Volume)

postResample(predicted.classes, test.data$Volume)
summary(predicted.classes)
plot(predicted.classes)

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

# Compare Models



# To Save the current work space
save.image()

#.............................................................................#
# Now, we want to apply the best model on the new product dataset:

# Import the new data
library(readr)
df2 <- read_csv("~/Documents/''Nastaran''/Analytics@UTexas/Courses/Course 3/Task 3/productattributes/newproductattributes2017.csv")
View(df2)

# Dummify the data

library(caret)

newDataFrame2 <- dummyVars(" ~ .", data = df2)
newData <- data.frame(predict(newDataFrame, newdata = df2))
View(newData)

str(newData)
summary(newData)

# Remove NAs
newData <- na.omit(newData)
newData$attributeWithMissingData <- NULL

# Drop the columns of the data frame

library(dplyr)
newData <- select (newData,-c(BestSellersRank,ProductNum,ProfitMargin,x5StarReviews))
View(newData)
summary(newData)

names(newData)<-c("Accessories","Display","ExtendedWarranty", "GameConsole", "Laptop",
                  "Netbook", "PC", "Printer", "PrinterSupplies", "Smartphone", "Software",
                  "Tablet", "Price", "x4StarReviews", "x3StarReviews", "x2StarReviews",
                  "x1StarReviews", "PositiveServiceReview", "NegativeServiceReview", "Recommendproduct",
                  "ShippingWeight", "ProductDepth", "ProductWidth", "ProductHeight", "Volume") 

view(newData)
str(newData)

comboInfo <- findLinearCombos(newData)
comboInfo

### Prediction by the best model:

#Linear Model
lm3 <-lm(Volume~.,newData)
lm3
summary(lm3)
print(lm3)


# SVM

library(caret)
train_control2 <- trainControl(method="repeatedcv", number=10, repeats=3)

set.seed(123)
svm1_new <- train(Volume ~., data = newData, method = "svmLinear2",
                  trControl = train_control2,  preProcess = c("center","scale"))
svm1_new
summary(svm1_new)
plot(svm1_new)

pred_svm <- predict(svm1, test.data)
pred_svm
plot(pred_svm)
head(pred_svm)


# By Random Forest
pred4 <- rf_classifier %>% predict(newData)
plot(pred2)
head(pred2)
postResample(pred2, newData$Volume)
pred2



#### Final Predictions


final_pred <- predict(lm3, newData=data.frame(newData$Laptop))
final_pred



final_pred2 <- predict(svm1, newData=data.frame(newData$Netbook))
final_pred2


finalPred <- predict(mod_rf2, newdata=data.frame(newData$PC))



#Add predictions to the new products data set 
output3 <- newData 
output3$predictions <- pred_svm

write.csv(output3, file="C2.T3output.csv", row.names = TRUE)



